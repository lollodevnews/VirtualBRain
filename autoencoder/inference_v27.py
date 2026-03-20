import torch
import os
import glob
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

print("====================================================")
print(" VIRTUALBRAIN V27: PYTHON INFERENCE EMULATOR")
print("====================================================")

MODEL_ID = os.path.expanduser("~/models/quant/qwen25_7b")
COMPILED_DIR = os.path.expanduser("~/models/quant/vbr_qwen25_v27") 

def unpack_vbr_matrix(payload, device="cuda:0"):
    """
    Natively unpacks the VBR SWAR bytes and evaluates the 
    Continuous S-Curve (a, b, m, n) back into an FP16 matrix.
    """
    out_features, in_features = payload["original_shape"]
    
    # Move required vectors to GPU for extremely fast bitwise decoding
    vbr_data = payload["vbr_data"].to(device)
    vbr_offsets = payload["vbr_offsets"].to(device)
    headers = payload["vbr_headers"].to(device)
    divisors = payload["row_divisors"].to(device)
    scales = payload["vbr_scales"].to(device)
    a = payload["alpha_a"].to(device)
    b = payload["alpha_b"].to(device)
    m = payload["power_m"].to(device)
    n = payload["power_n"].to(device)
    dust = payload["dust_anchors"].to(device)
    
    powers_of_2 = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=device, dtype=torch.uint8)
    W_recon = torch.zeros((out_features, in_features), dtype=torch.float16, device=device)
    
    for i in range(out_features):
        final_d = int(headers[i].item())
        mag_bits = final_d - 1
        offset = int(vbr_offsets[i].item())
        
        # Calculate exactly how many bytes belong to this row
        num_chunks = in_features // 8
        row_bytes_count = num_chunks * (mag_bits + 1)
        row_data = vbr_data[offset:offset+row_bytes_count].view(num_chunks, mag_bits + 1)
        
        # 1. Unpack Magnitudes from SWAR Bytes
        mag_ints = torch.zeros((num_chunks, 8), dtype=torch.int32, device=device)
        for bit_idx in range(mag_bits):
            byte_col = row_data[:, bit_idx].unsqueeze(1)
            bit_matrix = (byte_col & powers_of_2) > 0
            mag_ints += bit_matrix.to(torch.int32) << bit_idx
        mag_ints = mag_ints.flatten()
        
        # 2. Unpack Signs from the final byte column
        sign_byte_col = row_data[:, mag_bits].unsqueeze(1)
        sign_matrix = (sign_byte_col & powers_of_2) > 0
        signs = sign_matrix.flatten()
        
        # 3. Evaluate the Continuous Polynomial S-Curve!
        norm = mag_ints.float() / divisors[i].float()
        linear = (1.0 - a[i] - b[i]) * norm
        curve = linear + a[i] * (norm ** m[i]) + b[i] * (norm ** n[i])
        curve = torch.clamp(curve, 0.0, 1.0)
        
        # 4. Apply Dust Anchor, Scale, and Sign
        M = (curve * scales[i]) + dust[i]
        row_weights = torch.where(signs, -M, M) # True meant negative during compression
        
        W_recon[i] = row_weights.half()
        
    return W_recon

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"[*] Loading FP16 Skeleton from {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    chunk_files = sorted(glob.glob(os.path.join(COMPILED_DIR, "compressed_*.pt")))
    if not chunk_files:
        print(f"[!] No compressed files found in {COMPILED_DIR}. Did you run the Autoencoder?")
        return
        
    print("\n[*] Unpacking V28 SWAR Files & Rebuilding Geometry...")
    # We do this one chunk at a time to prevent CPU RAM spikes
    for chunk_file in chunk_files:
        print(f" -> Reading {os.path.basename(chunk_file)}...")
        data = torch.load(chunk_file, map_location="cpu")["experts_cold"]
        
        for name, payload in tqdm(data.items(), leave=False, desc="Decoding Matrices"):
            module = model.get_submodule(name)
            
            if payload["is_vbr_compressed"]:
                # Decompress natively on GPU for speed, then pull the FP16 matrix to CPU RAM
                W_recon = unpack_vbr_matrix(payload, device=device)
                module.weight.data = W_recon.cpu()
            else:
                module.weight.data = payload["raw_data"].cpu()
                
            if "bias" in payload and module.bias is not None:
                module.bias.data = payload["bias"].cpu()
                
    print("\n[*] Geometry Rebuilt. Pushing 7B model to GPU...")
    model = model.to(device)
    
    prompt = "The architecture of a neural network can be described as"
    print(f"\n[Prompt] {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print("[*] Generating (Hugging Face FP16 Native)...")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
        
    print(f"\n[Response]\n{tokenizer.decode(outputs[0], skip_special_tokens=True)}\n")

if __name__ == "__main__":
    main()
