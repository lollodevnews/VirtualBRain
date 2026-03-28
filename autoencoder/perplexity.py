import os
# 1. HARDWARE CONFIGURATION
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
os.environ["HSA_ENABLE_SDMA"] = "0"
#os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "max_split_size_mb:128"

HARDWARE_TARGET = "mi50"
from vbr_options import setup_hardware_profile
setup_hardware_profile(HARDWARE_TARGET)

# 2. STANDARD IMPORTS
import torch
import glob
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ==========================================
# HYPERPARAMETERS
# ==========================================
EARLY_EXIT_50 = False   # <--- EARLY EXIT TOGGLE
CONTEXT_LENGTH = 2048  # If it still OOMs, drop this to 1024 or 512
STRIDE = 512
# ==========================================

print("==========================================")
print(" VIRTUALBRAIN V36: PERPLEXITY BENCHMARK   ")
print("==========================================")

MODEL_ID = os.path.expanduser("~/models/quant/qwen25_7b")
COMPRESSED_DIR = "./quant_qwen"
DEVICE = "cuda:0"


# ==========================================
# HIPBLAS WARM-UP HACK
# Force PyTorch to allocate the math library context 
# while the GPU VRAM is 100% empty.
# ==========================================
print("[*] Warming up HIPBLAS context...")
_dummy_a = torch.randn(16, 16, device=DEVICE)
_dummy_b = torch.randn(16, 16, device=DEVICE)
_ = _dummy_a @ _dummy_b
del _dummy_a, _dummy_b, _
torch.cuda.empty_cache()
# ==========================================



def dequantize_vbr_v36(weight_dict, device="cuda"):
    """ Natively unpacks the V36 4-2-1-1 Superblocks (Hardware Safe) """
    out_features, in_features = weight_dict["original_shape"]
    
    vbr_data_32 = weight_dict["vbr_data"].to(device).view(torch.int32)
    offsets_32 = weight_dict["vbr_offsets"].to(device) // 4
    headers = weight_dict["bitrates"].to(device).squeeze()    
    
    scales = weight_dict["row_max"].to(device).float()      
    alphas = weight_dict["param_a"].to(device).float()      
    cs = weight_dict["param_c"].to(device).float()          
    ms = weight_dict["param_m"].to(device).float()          
    
    recon_matrix = torch.empty((out_features, in_features), dtype=torch.float16, device=device)
    num_sb = in_features // 32
    
    for D_val in range(2, 9): 
        mask = (headers == D_val)
        if not mask.any(): continue
            
        row_indices = torch.nonzero(mask).squeeze(-1)
        num_rows = row_indices.shape[0]
        
        mag_bits = D_val - 1
        sb_words = 0
        if mag_bits >= 4: sb_words += 4; mag_bits -= 4
        if mag_bits >= 2: sb_words += 2; mag_bits -= 2
        if mag_bits == 1: sb_words += 1
        sb_words += 1 
        
        row_word_len = num_sb * sb_words
        starts = offsets_32[row_indices].unsqueeze(1)
        steps = torch.arange(row_word_len, device=device).unsqueeze(0)
        
        row_words = vbr_data_32[starts + steps].view(num_rows, num_sb, sb_words)
        mag_accumulator = torch.zeros((num_rows, num_sb, 32), device=device, dtype=torch.int32)
        
        ptr, m_bits, shift_up = 0, D_val - 1, 0
        
        if m_bits >= 4:
            w = row_words[:, :, ptr : ptr+4].unsqueeze(-1)
            sh = (torch.arange(8, device=device) * 4).view(1, 1, 1, 8)
            mag_accumulator |= (((w >> sh) & 0xF).view(num_rows, num_sb, 32) << shift_up)
            ptr += 4; m_bits -= 4; shift_up += 4
            
        if m_bits >= 2:
            w = row_words[:, :, ptr : ptr+2].unsqueeze(-1)
            sh = (torch.arange(16, device=device) * 2).view(1, 1, 1, 16)
            mag_accumulator |= (((w >> sh) & 0x3).view(num_rows, num_sb, 32) << shift_up)
            ptr += 2; m_bits -= 2; shift_up += 2
            
        if m_bits == 1:
            w = row_words[:, :, ptr : ptr+1].unsqueeze(-1)
            sh = torch.arange(32, device=device).view(1, 1, 1, 32)
            mag_accumulator |= (((w >> sh) & 0x1).view(num_rows, num_sb, 32) << shift_up)
            ptr += 1
            
        sw = row_words[:, :, ptr : ptr+1].unsqueeze(-1)
        sh = torch.arange(32, device=device).view(1, 1, 1, 32)
        signs = ((sw >> sh) & 0x1).view(num_rows, in_features)

        # ==========================================
        # TOPOLOGY MATH (The V35 LUT Architecture)
        # ==========================================
        K_bins = 1 << (D_val - 1)
        
        # 1. Grab the raw integer magnitudes (e.g., 0, 1, 2, 3)
        mag_idx = mag_accumulator.view(num_rows, in_features).long()
        
        # 2. Rebuild the V35 Base Bins (The raised noise floors!)
        if D_val == 2:
            base_bins = torch.tensor([0.20, 1.0], device=device, dtype=torch.float32)
        elif D_val == 3:
            base_bins = torch.tensor([0.25, 0.50, 0.75, 1.0], device=device, dtype=torch.float32)
        else:
            divisor = float(K_bins - 1)
            base_bins = torch.arange(K_bins, device=device).float() / divisor
            
        a = alphas[row_indices].unsqueeze(1).float()
        c = cs[row_indices].unsqueeze(1).float()
        m = ms[row_indices].unsqueeze(1).float()
        s = scales[row_indices].unsqueeze(1).float()
        b = base_bins.unsqueeze(0) # Broadcastable base bins [1, K_bins]

        # 3. LUT Math (Evaluate ONLY K_bins times per row, bypass 0^0 safely)
        safe_b = torch.where(b == 0.0, 1e-9, b)
        m_term = torch.where(m == 0.0, 1.0, 
                    torch.where(b == 0.0, 0.0, safe_b ** m))
        
        inner = (1.0 - a) * b + a * m_term
        inner_clamped = torch.clamp(inner, 0.0, 1.0)
        
        safe_inner = torch.where(inner_clamped == 0.0, 1e-9, inner_clamped)
        warped_lut = torch.where(c == 0.0, 1.0, 
                    torch.where(inner_clamped == 0.0, 0.0, safe_inner ** c)) # [num_rows, K_bins]

        # 4. Gather Physical Bins (Hardware accelerated lookup mapping 4096 columns)
        gathered = torch.gather(warped_lut, 1, mag_idx)
        
        # 5. Apply Scale and Sign
        magnitudes = gathered * s
        sign_flip = torch.where(signs == 1, -1.0, 1.0).float()
        
        recon_matrix[row_indices] = (magnitudes * sign_flip).half()

    return recon_matrix

def get_module(m, name):
    for p in name.split("."): m = getattr(m, p)
    return m

@torch.inference_mode()
def main():
    print("[1] Loading Base Model and Tokenizer (FP16)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=DEVICE # Load directly to GPU
        # REMOVED attn_implementation="eager"
    )
    
    print(f"\n[2] Injecting V36 Weights from {COMPRESSED_DIR}...")
    # ... keep your injection loop exactly as it is ...
    # Make sure W = dequantize_vbr_v36(...) returns .half() (which it does!)
    
    # REMOVED model = model.to(DEVICE).float()
    
    print("\n[4] Loading WikiText-2 Dataset...")
    # ... run your dataset loop exactly as it is ...

    print(f"\n[2] Injecting V36 Weights from {COMPRESSED_DIR}...")
    chunk_files = sorted(glob.glob(os.path.join(COMPRESSED_DIR, "*.pt")))
    total_updated = 0
    total_found = 0
    
    for cf in chunk_files:
        data = torch.load(cf, map_location="cpu")["experts_cold"]
        total_found += len(data)
        
        for key, payload in tqdm(data.items(), leave=False, desc=f"Unpacking {os.path.basename(cf)}"):
            module_name = key.replace(".weight", "")
            try: target = get_module(model, module_name)
            except AttributeError:
                try: target = get_module(model, module_name.replace("model.", ""))
                except AttributeError:
                    continue

            if payload.get("is_vbr_compressed", False):
                W = dequantize_vbr_v36(payload, device=DEVICE)
                target.weight.data.copy_(W.cpu())
                total_updated += 1
                del W
            elif "raw_data" in payload:
                target.weight.data.copy_(payload["raw_data"].cpu())
                total_updated += 1

            if "bias" in payload and payload["bias"] is not None:
                if hasattr(target, "bias") and target.bias is not None:
                    target.bias.data.copy_(payload["bias"].cpu())
                    
        del data; gc.collect()

    print(f"\n[*] AUDIT: Updated {total_updated} linear layers out of {total_found} found.")

    print("\n[3] Pushing Model to GPU Device (FP32 Accumulation)...")
    #model = model.to(DEVICE).float()

    print("\n[4] Loading WikiText-2 Dataset...")
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    max_length = min(CONTEXT_LENGTH, model.config.max_position_embeddings) 
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    iterations = 0
    
    print(f"\n[5] Calculating Perplexity (Context: {max_length}, Stride: {STRIDE})...")
    
    pbar = tqdm(range(0, seq_len, STRIDE))
    for begin_loc in pbar:
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc 
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        
        # AGGRESSIVE GARBAGE COLLECTION FOR FP32 OOM
        del outputs
        del input_ids
        del target_ids
        torch.cuda.empty_cache()

        prev_end_loc = end_loc
        iterations += 1
        
        if EARLY_EXIT_50 and iterations >= 50:
            print("\n[*] Early Exit Triggered (50 iterations reached).")
            break
            
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / prev_end_loc)
    print("\n==========================================")
    print(f" V36 FINAL PERPLEXITY (WikiText-2): {ppl.item():.4f}")
    print("==========================================\n")

if __name__ == "__main__":
    main()
