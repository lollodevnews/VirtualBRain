import torch
import os
import glob
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

print("==========================================")
print(" VIRTUALBRAIN V35: PERPLEXITY BENCHMARK   ")
print("==========================================")

# Update these paths if necessary
#MODEL_ID = "meta-llama/Llama-2-7b-hf"

MODEL_ID = os.path.expanduser("~/models/quant/qwen25_7b")
COMPRESSED_DIR = "./quant_qwen"
DEVICE = "cuda:0"

def dequantize_vbr_matrix_v35(weight_dict, device="cuda"):
    """ Natively unpacks the V35 SWAR Superblocks (Fully Vectorized) """
    out_features, in_features = weight_dict["original_shape"]
    vbr_data = weight_dict["vbr_data"].to(device)
    offsets = weight_dict["vbr_offsets"].to(device)
    bitrates = weight_dict["bitrates"].to(device)
    
    # V35 Math Parameters
    param_a = weight_dict["param_a"].to(device).float()
    param_c = weight_dict["param_c"].to(device).float()
    param_m = weight_dict["param_m"].to(device).float()
    row_max = weight_dict["row_max"].to(device).float()
    
    decoded_weights = torch.zeros((out_features, in_features), dtype=torch.float16, device=device)
    
    for current_D in range(2, 9):
        mask = (bitrates == current_D)
        if not mask.any(): continue
        
        row_idx = mask.nonzero().squeeze(-1)
        num_rows = row_idx.shape[0]
        
        num_chunks = in_features // 8
        row_bytes = num_chunks * current_D
        
        # --- VECTORIZED ROW FETCH ---
        starts = offsets[row_idx].unsqueeze(1)
        byte_offsets = torch.arange(row_bytes, device=device).unsqueeze(0)
        indices = starts + byte_offsets
        
        raw_bytes = vbr_data[indices].to(torch.int32) # [num_rows, row_bytes]
        blocks = raw_bytes.view(num_rows, num_chunks, current_D)
        
        idx_unpacked = torch.zeros((num_rows, num_chunks, 8), dtype=torch.int32, device=device)
        
        # 1. Unpack SWAR Chunks
        for bit_idx in range(current_D):
            byte_col = blocks[:, :, bit_idx]
            for bit_pos in range(8):
                bit_val = (byte_col >> bit_pos) & 1
                idx_unpacked[:, :, bit_pos] |= (bit_val << bit_idx)
                
        idx_flat = idx_unpacked.view(num_rows, in_features)
        
        # --- V35 MAGNITUDE & SIGN RECOVERY ---
        K_bins = 2 ** (current_D - 1)
        is_negative = idx_flat >= K_bins
        mag_idx = idx_flat % K_bins
        
        # 2. Rebuild Base Bins
        if current_D == 2:
            base_bins = torch.tensor([0.20, 1.0], device=device, dtype=torch.float32)
        elif current_D == 3:
            base_bins = torch.tensor([0.25, 0.50, 0.75, 1.0], device=device, dtype=torch.float32)
        else:
            divisor = float(K_bins - 1)
            base_bins = torch.arange(K_bins, device=device).float() / divisor
            
        # 3. Vectorized Topology Math
        a = param_a[row_idx].unsqueeze(1)
        c = param_c[row_idx].unsqueeze(1)
        m = param_m[row_idx].unsqueeze(1)
        b = base_bins.unsqueeze(0) # Broadcastable base bins
        
        inner = (1.0 - a) * b + a * (b ** m)
        warped = torch.clamp(inner, 1e-6, 1.0) ** c # [num_rows, K_bins]
        
        # 4. Gather Physical Bins (Hardware accelerated lookup)
        gathered = torch.gather(warped, 1, mag_idx.long()) # [num_rows, in_features]
        
        # 5. Apply Scale and Fused Sign
        w = gathered * row_max[row_idx].unsqueeze(1)
        w[is_negative] *= -1.0
        
        decoded_weights[row_idx] = w.half()
            
    return decoded_weights

def get_nested_module(m, name):
    for p in name.split("."): m = getattr(m, p)
    return m

@torch.inference_mode()
def main():
    print("[1] Loading Base Model and Tokenizer (FP16)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=DEVICE
    )
    
    print(f"[2] Injecting V35 Weights from {COMPRESSED_DIR}...")
    pt_files = sorted(glob.glob(os.path.join(COMPRESSED_DIR, "*.pt")))
    
    for pt_file in pt_files:
        print(f"    -> Inflating {os.path.basename(pt_file)}...")
        chunk_data = torch.load(pt_file, map_location="cpu")["experts_cold"]
        for key, weight_dict in chunk_data.items():
            module_path = key.replace(".weight", "") if key.endswith(".weight") else key
            try: target_module = get_nested_module(model, module_path)
            except AttributeError: continue

            if weight_dict.get("is_vbr_compressed", False):
                fp16_weight = dequantize_vbr_matrix_v35(weight_dict, device=DEVICE)
                target_module.weight.data.copy_(fp16_weight)
                if "bias" in weight_dict and weight_dict["bias"] is not None:
                    if hasattr(target_module, "bias") and target_module.bias is not None:
                        target_module.bias.data.copy_(weight_dict["bias"].to(DEVICE))
            
            elif "raw_data" in weight_dict:
                target_module.weight.data.copy_(weight_dict["raw_data"].to(DEVICE))
                if "bias" in weight_dict and weight_dict["bias"] is not None:
                    if hasattr(target_module, "bias") and target_module.bias is not None:
                        target_module.bias.data.copy_(weight_dict["bias"].to(DEVICE))
                        
        del chunk_data
        torch.cuda.empty_cache()
        gc.collect()

    print("[3] Loading WikiText-2 Dataset...")
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    max_length = model.config.max_position_embeddings
    max_length = min(2048, max_length) 
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    
    print(f"[4] Calculating Perplexity (Context: {max_length}, Stride: {stride})...")
    
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc 
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    print("\n==========================================")
    print(f" V35 FINAL PERPLEXITY (WikiText-2): {ppl.item():.4f}")
    print("==========================================\n")

if __name__ == "__main__":
    main()
