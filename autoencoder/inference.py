import os
import glob
import gc
import math
from tqdm import tqdm

# 1. HARDWARE LOCK (Must be at the absolute top for MI50)
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
os.environ["HSA_ENABLE_SDMA"] = "0"

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# 2. CONFIGURATION
DEVICE = "cuda:0"
MODEL_ID = os.path.expanduser("~/models/quant/qwen25_7b")
COMPILED_DIR = "./quant_qwen"

# ==========================================
# THE V36 DEQUANTIZER (UNIFIED LUT EDITION)
# ==========================================
def dequantize_vbr_v36(weight_dict, device="cuda"):
    """
    Decodes 4-2-1-1 Superblocks with Unified Desmos Continuous Topology.
    Branchless LUT architecture optimized for hardware.
    """
    out_features, in_features = weight_dict["original_shape"]
    
    # Cast to int32 words immediately
    vbr_data_32 = weight_dict["vbr_data"].to(device).view(torch.int32)
    offsets_32 = weight_dict["vbr_offsets"].to(device) // 4
    headers = weight_dict["bitrates"].to(device).squeeze()    
    
    # Metadata parameters in FP32 for precision
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
        
        # Calculate words per superblock for this bitrate
        mag_bits = D_val - 1
        sb_words = 0
        if mag_bits >= 4: sb_words += 4; mag_bits -= 4
        if mag_bits >= 2: sb_words += 2; mag_bits -= 2
        if mag_bits == 1: sb_words += 1
        sb_words += 1 # Sign word
        
        row_word_len = num_sb * sb_words
        starts = offsets_32[row_indices].unsqueeze(1)
        steps = torch.arange(row_word_len, device=device).unsqueeze(0)
        
        # Vectorized gather of all superblock words for this batch
        row_words = vbr_data_32[starts + steps].view(num_rows, num_sb, sb_words)
        mag_accumulator = torch.zeros((num_rows, num_sb, 32), device=device, dtype=torch.int32)
        
        ptr, m_bits, shift_up = 0, D_val - 1, 0
        
        # 1. Unpack 4-Bit Base
        if m_bits >= 4:
            w = row_words[:, :, ptr : ptr+4].unsqueeze(-1)
            sh = (torch.arange(8, device=device) * 4).view(1, 1, 1, 8)
            mag_accumulator |= (((w >> sh) & 0xF).view(num_rows, num_sb, 32) << shift_up)
            ptr += 4; m_bits -= 4; shift_up += 4
            
        # 2. Unpack 2-Bit Residual
        if m_bits >= 2:
            w = row_words[:, :, ptr : ptr+2].unsqueeze(-1)
            sh = (torch.arange(16, device=device) * 2).view(1, 1, 1, 16)
            mag_accumulator |= (((w >> sh) & 0x3).view(num_rows, num_sb, 32) << shift_up)
            ptr += 2; m_bits -= 2; shift_up += 2
            
        # 3. Unpack 1-Bit Residual
        if m_bits == 1:
            w = row_words[:, :, ptr : ptr+1].unsqueeze(-1)
            sh = torch.arange(32, device=device).view(1, 1, 1, 32)
            mag_accumulator |= (((w >> sh) & 0x1).view(num_rows, num_sb, 32) << shift_up)
            ptr += 1
            
        # 4. Unpack Signs
        sw = row_words[:, :, ptr : ptr+1].unsqueeze(-1)
        sh = torch.arange(32, device=device).view(1, 1, 1, 32)
        signs = ((sw >> sh) & 0x1).view(num_rows, in_features)

        # ==========================================
        # TOPOLOGY MATH (Unified Branchless LUT)
        # ==========================================
        K_bins = 1 << (D_val - 1)
        divisor = float(K_bins - 1)
        
        # 1. Generate Uniform Base Bins (No if/else branching!)
        base_bins = torch.arange(K_bins, device=device).float() / divisor
        
        # 2. Grab the raw integer magnitudes (e.g., 0, 1, 2, 3)
        mag_idx = mag_accumulator.view(num_rows, in_features).long()
        
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

# ==========================================
# MAIN EXECUTION
# ==========================================
@torch.inference_mode()
def main():
    print(f"[*] Initializing Qwen Skeleton: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 🚨 FIX: Removed eager attention, loading natively in FP16 to GPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map=DEVICE 
    )
    
    chunk_files = sorted(glob.glob(os.path.join(COMPILED_DIR, "*.pt")))

    print("\n[*] Injecting V36 Quantized Geometry...")
    total_updated = 0
    total_found_in_files = 0

    for cf in chunk_files:
        data = torch.load(cf, map_location="cpu")["experts_cold"]
        total_found_in_files += len(data)
        
        for key, payload in tqdm(data.items(), leave=False, desc=f"Unpacking {os.path.basename(cf)}"):
            # 1. STRIP EVERYTHING TO BE SAFE
            module_name = key.replace(".weight", "")
            
            try:
                target = get_module(model, module_name)
            except AttributeError:
                # TRY FALLBACK: Some versions of HF Qwen don't have the 'model.' prefix in named_modules
                try:
                    alt_name = module_name.replace("model.", "")
                    target = get_module(model, alt_name)
                except AttributeError:
                    print(f"\n[!] WARNING: Could not find module {module_name} in the model skeleton!")
                    continue

            # 2. LOAD LOGIC
            if payload.get("is_vbr_compressed", False):
                W = dequantize_vbr_v36(payload, device=DEVICE)
                target.weight.data.copy_(W.cpu())
                total_updated += 1
                del W
            elif "raw_data" in payload:
                target.weight.data.copy_(payload["raw_data"].cpu())
                total_updated += 1
            else:
                print(f"\n[!] WARNING: Layer {module_name} found but has no recognizable data!")

            # 3. BIAS HANDLING
            if "bias" in payload and payload["bias"] is not None:
                if hasattr(target, "bias") and target.bias is not None:
                    target.bias.data.copy_(payload["bias"].cpu())
        
        del data; gc.collect()

    print(f"\n[*] AUDIT COMPLETE: Updated {total_updated} linear layers out of {total_found_in_files} found in files.")

    # 🚨 FIX: Removed model = model.to(DEVICE).float() which destroyed RoPE embeddings
    print("\n[*] Preparing Model for Generation...")
    torch.cuda.empty_cache()

    prompt = "The future of bare-metal GPU programming is"
    print(f"\n[Prompt] {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    print("[*] Generating...")
    # Using stable sampling parameters
    outputs = model.generate(
        **inputs, 
        max_new_tokens=100, 
        do_sample=True, 
        temperature=0.7, 
        top_k=50, 
        repetition_penalty=1.15
    )
        
    print(f"\n[Response]\n{tokenizer.decode(outputs[0], skip_special_tokens=True)}\n")

if __name__ == "__main__":
    main()
