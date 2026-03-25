import torch
import os
import glob
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

print("==========================================")
print(" VIRTUALBRAIN V34: WEIGHT VALIDATION TEST ")
print("==========================================")

MODEL_ID = os.path.expanduser("~/models/quant/qwen25_7b")
COMPRESSED_DIR = os.path.expanduser("~/models/quant/vbr_qwen25_v34")
DEVICE = "cuda:0"

def get_nested_module(m, name):
    for p in name.split("."): 
        m = getattr(m, p)
    return m

def dequantize_vbr_v34_matrix(weight_dict, device="cuda"):
    """ Natively unpacks the V34 Superblocks (Fully Vectorized) """
    out_features, in_features = weight_dict["original_shape"]
    vbr_data = weight_dict["vbr_data"].to(device)
    offsets = weight_dict["vbr_offsets"].to(device)
    headers = weight_dict["vbr_headers"].to(device) 
    row_divisors = weight_dict["row_divisors"].to(device)
    scales = weight_dict["vbr_scales"].to(device)
    
    a = weight_dict["alpha_a"].to(device)
    b = weight_dict["alpha_b"].to(device)
    m = weight_dict["power_m"].to(device)
    n = weight_dict["power_n"].to(device)
    
    decoded_weights = torch.zeros((out_features, in_features), dtype=torch.float16, device=device)
    
    for current_D in range(2, 9):
        mask = (headers == current_D)
        if not mask.any(): continue
        
        row_idx = mask.nonzero().squeeze(-1)
        num_rows = row_idx.shape[0]
        mag_bits = current_D - 1
        row_bytes = (current_D * in_features) // 8
        
        # --- VECTORIZED ROW FETCH ---
        starts = offsets[row_idx].unsqueeze(1)
        byte_offsets = torch.arange(row_bytes, device=device).unsqueeze(0)
        indices = starts + byte_offsets
        raw_bytes = vbr_data[indices] # Shape: [num_rows, row_bytes]
        
        # Reshape into [num_rows, blocks_per_row, bytes_per_block]
        blocks = raw_bytes.view(num_rows, -1, current_D * 4) 
        num_blocks = blocks.shape[1]
        
        r_ints = torch.zeros(num_rows, num_blocks, 32, dtype=torch.int32, device=device)
        byte_offset = 0
        m_bits = mag_bits
        shift = 0
        
        # 1. Unpack 4-bit chunk
        if m_bits >= 4:
            pack4 = blocks[:, :, byte_offset:byte_offset+16]
            b4 = torch.empty(num_rows, num_blocks, 32, dtype=torch.int32, device=device)
            b4[:, :, 0::2] = pack4 & 0x0F
            b4[:, :, 1::2] = (pack4 >> 4) & 0x0F
            r_ints |= (b4 << shift)
            byte_offset += 16
            shift += 4
            m_bits -= 4
            
        # 2. Unpack 2-bit chunk
        if m_bits >= 2:
            pack2 = blocks[:, :, byte_offset:byte_offset+8]
            b2 = torch.empty(num_rows, num_blocks, 32, dtype=torch.int32, device=device)
            b2[:, :, 0::4] = pack2 & 0x03
            b2[:, :, 1::4] = (pack2 >> 2) & 0x03
            b2[:, :, 2::4] = (pack2 >> 4) & 0x03
            b2[:, :, 3::4] = (pack2 >> 6) & 0x03
            r_ints |= (b2 << shift)
            byte_offset += 8
            shift += 2
            m_bits -= 2
            
        # 3. Unpack 1-bit chunk
        if m_bits == 1:
            pack1 = blocks[:, :, byte_offset:byte_offset+4]
            b1 = torch.empty(num_rows, num_blocks, 32, dtype=torch.int32, device=device)
            for bit in range(8):
                b1[:, :, bit::8] = (pack1 >> bit) & 0x01
            r_ints |= (b1 << shift)
            byte_offset += 4
            shift += 1
            
        # 4. Unpack Signs
        pack_sign = blocks[:, :, byte_offset:byte_offset+4]
        r_signs = torch.empty(num_rows, num_blocks, 32, dtype=torch.int32, device=device)
        for bit in range(8):
            r_signs[:, :, bit::8] = (pack_sign >> bit) & 0x01
            
        # Vectorized Math Execution
        mag = r_ints.view(num_rows, -1).float()
        
        # --- V34: SYMMETRIC NO-ZERO SHIFT ---
        if current_D <= 4:
            mag += 1.0
            
        x_norm = mag / row_divisors[row_idx].unsqueeze(1)
        
        a_v = a[row_idx].unsqueeze(1)
        b_v = b[row_idx].unsqueeze(1)
        m_v = m[row_idx].unsqueeze(1)
        n_v = n[row_idx].unsqueeze(1)
        scale_v = scales[row_idx].unsqueeze(1)
        
        lin = (1.0 - a_v - b_v) * x_norm
        curve = lin + a_v * (x_norm ** m_v) + b_v * (x_norm ** n_v)
        
        w = curve * scale_v
        w = torch.where(r_signs.view(num_rows, -1) > 0, -w, w)
        
        decoded_weights[row_idx] = w.half()
            
    return decoded_weights

@torch.inference_mode()
def main():
    print("[1] Loading Base Model and Tokenizer (FP16)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=DEVICE
    )
    
    print(f"[2] Finding V34 Compressed Chunks in {COMPRESSED_DIR}...")
    pt_files = sorted(glob.glob(os.path.join(COMPRESSED_DIR, "*.pt")))
    if not pt_files:
        print("ERROR: No .pt files found!")
        return
        
    print("[3] Dequantizing and Injecting V34 Weights...")
    for pt_file in pt_files:
        print(f"    -> Processing {os.path.basename(pt_file)}...")
        chunk_data = torch.load(pt_file, map_location="cpu")["experts_cold"]
        
        for key, weight_dict in chunk_data.items():
            module_path = key.replace(".weight", "") if key.endswith(".weight") else key
            
            try:
                target_module = get_nested_module(model, module_path)
            except AttributeError:
                continue

            if weight_dict.get("is_vbr_compressed", False):
                fp16_weight = dequantize_vbr_v34_matrix(weight_dict, device=DEVICE)
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

    print("\n[4] Injection Complete. Running Standard HuggingFace Generation...")
    model.eval()
    prompt = "The future of bare-metal GPU programming is"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        temperature=0.7,
        repetition_penalty=1.1,
        do_sample=True,
        top_k=50
    )
    
    print("\n--- OUTPUT ---")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("--------------\n")

if __name__ == "__main__":
    main()
