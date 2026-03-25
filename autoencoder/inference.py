import torch
import os
import glob
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

print("====================================================")
print(" VIRTUALBRAIN V35: PYTHON INFERENCE EMULATOR")
print("====================================================")

# Update these paths to match your local setup
#MODEL_ID = "meta-llama/Llama-2-7b-hf"
MODEL_ID = os.path.expanduser("~/models/quant/qwen25_7b")
COMPILED_DIR = "./quant_qwen"
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
        
        raw_bytes = vbr_data[indices].to(torch.int32) 
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
        b = base_bins.unsqueeze(0) 
        
        inner = (1.0 - a) * b + a * (b ** m)
        warped = torch.clamp(inner, 1e-6, 1.0) ** c 
        
        # 4. Gather Physical Bins
        gathered = torch.gather(warped, 1, mag_idx.long()) 
        
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
    print(f"[*] Loading FP16 Skeleton from {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    chunk_files = sorted(glob.glob(os.path.join(COMPILED_DIR, "*.pt")))
    if not chunk_files:
        print(f"[!] No compressed files found in {COMPILED_DIR}. Did you run the Autoencoder?")
        return
        
    print("\n[*] Unpacking V35 SWAR Files & Rebuilding Geometry...")
    for chunk_file in chunk_files:
        print(f" -> Reading {os.path.basename(chunk_file)}...")
        data = torch.load(chunk_file, map_location="cpu")["experts_cold"]
        
        for key, payload in tqdm(data.items(), leave=False, desc="Decoding Matrices"):
            module_path = key.replace(".weight", "") if key.endswith(".weight") else key
            try: 
                target_module = get_nested_module(model, module_path)
            except AttributeError: 
                continue
            
            if payload.get("is_vbr_compressed", False):
                # Decompress natively on GPU for speed, pull the FP16 matrix to CPU RAM
                W_recon = dequantize_vbr_matrix_v35(payload, device=DEVICE)
                target_module.weight.data.copy_(W_recon.cpu())
                del W_recon
            elif "raw_data" in payload:
                target_module.weight.data.copy_(payload["raw_data"].cpu())
                
            if "bias" in payload and payload["bias"] is not None:
                if hasattr(target_module, "bias") and target_module.bias is not None:
                    target_module.bias.data.copy_(payload["bias"].cpu())
                    
        del data
        torch.cuda.empty_cache()
        gc.collect()
                
    print("\n[*] Geometry Rebuilt. Pushing 7B model to GPU...")
    model = model.to(DEVICE)
    
    prompt = "The architecture of a neural network can be described as"
    print(f"\n[Prompt] {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    print("[*] Generating (Hugging Face FP16 Native)...")
    outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
        
    print(f"\n[Response]\n{tokenizer.decode(outputs[0], skip_special_tokens=True)}\n")

if __name__ == "__main__":
    main()
