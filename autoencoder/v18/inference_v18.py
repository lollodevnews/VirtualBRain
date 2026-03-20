import torch
import os
import glob
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

print("==========================================")
print(" VIRTUALBRAIN V18 INFERENCE VALIDATOR     ")
print("==========================================")

# --- CONFIGURATION ---
MODEL_ID = os.path.expanduser("~/models/quant/qwen25_7b")
VBR_DIR = os.path.expanduser("~/models/quant/vbr_qwen25_v18")
DEVICE = "cuda:0"

def decompress_vbr_matrix(vbr_dict, device="cpu"):
    """Reverses the V18 Bit-Packing entirely on the CPU to protect GPU VRAM"""
    out_features, in_features = vbr_dict["original_shape"]
    
    vbr_data = vbr_dict["vbr_data"].to(device)
    offsets = vbr_dict["vbr_offsets"].to(device)
    headers_D = vbr_dict["vbr_headers"].to(device)
    scales = vbr_dict["vbr_scales"].float().to(device)
    dusts = vbr_dict["dust_anchors"].float().to(device)
    alpha_a = vbr_dict["alpha_a"].float().to(device)
    alpha_b = vbr_dict["alpha_b"].float().to(device)
    power_m = vbr_dict["power_m"].float().to(device)
    power_n = vbr_dict["power_n"].float().to(device)
    
    reconstructed = torch.zeros((out_features, in_features), dtype=torch.float16, device=device)
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.int32, device=device)
    
    num_groups = in_features // 8
    
    for i in range(out_features):
        D = int(headers_D[i].item())
        mag_bits = D - 1
        bytes_per_group = mag_bits + 1
        
        offset = offsets[i].item()
        row_bytes = vbr_data[offset : offset + (num_groups * bytes_per_group)].view(num_groups, bytes_per_group)
        
        mag_ints = torch.zeros((num_groups, 8), dtype=torch.int32, device=device)
        for bit_idx in range(mag_bits):
            byte_col = row_bytes[:, bit_idx].unsqueeze(1)
            bits = (byte_col & powers) > 0
            mag_ints += bits.int() << bit_idx
            
        sign_byte = row_bytes[:, -1].unsqueeze(1)
        signs = (sign_byte & powers) > 0
        
        mag_ints = mag_ints.flatten()
        signs = signs.flatten()
        
        max_int = (2 ** mag_bits) - 1
        norm_w = mag_ints.float() / max_int
        
        a, b, m, n = alpha_a[i], alpha_b[i], power_m[i], power_n[i]
        scale, dust = scales[i], dusts[i]
        
        linear_term = (1.0 - a - b) * norm_w
        curve = linear_term + (a * (norm_w ** m)) + (b * (norm_w ** n))
        
        mags = (curve * scale) + dust
        weights = mags * torch.where(signs, -1.0, 1.0)
        
        reconstructed[i] = weights.half()
        
    return reconstructed

def main():
    print("[*] Loading Qwen 2.5 7B Architecture...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cpu")
    
    pt_files = sorted(glob.glob(os.path.join(VBR_DIR, "compressed_*.pt")))
    if not pt_files:
        print("[-] Error: No compressed .pt files found!")
        return
        
    print(f"[*] Found {len(pt_files)} VBR chunks. Decompressing purely on CPU...")
    
    for pt_file in pt_files:
        print(f"    -> Processing {os.path.basename(pt_file)}...")
        chunk_data = torch.load(pt_file, map_location="cpu")["experts_cold"]
        
        for name, vbr_dict in tqdm(chunk_data.items(), desc="Unpacking Matrices", leave=False):
            submodule = model.get_submodule(name)
            
            if vbr_dict["is_vbr_compressed"]:
                # Force strictly to CPU. GPU remains pristine at 0 bytes.
                recon_weight = decompress_vbr_matrix(vbr_dict, device="cpu")
                submodule.weight.data = recon_weight
            else:
                submodule.weight.data = vbr_dict["raw_data"].cpu()
                
            if "bias" in vbr_dict and vbr_dict["bias"] is not None:
                submodule.bias.data = vbr_dict["bias"].cpu()
                
        # Aggressive garbage collection after each chunk
        del chunk_data
        gc.collect()

    print("\n[+] Injection Complete. Moving cleanly to GPU 0 for inference...")
    torch.cuda.empty_cache() # Final sweep before the big move
    model.to(DEVICE)
    model.eval()
    
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain the mathematical concept of a polynomial curve in one simple paragraph.<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    print("\n==========================================")
    print(" GENERATION TEST")
    print("==========================================\n")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150, 
            temperature=0.7, 
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("\n==========================================")

if __name__ == "__main__":
    main()
