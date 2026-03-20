import torch
import math
import os
import glob
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

print("==========================================")
print(" VIRTUALBRAIN V18 PERPLEXITY BENCHMARK    ")
print("==========================================")

# --- CONFIGURATION ---
MODEL_ID = os.path.expanduser("~/models/quant/qwen25_7b")
VBR_DIR = os.path.expanduser("~/models/quant/vbr_qwen25_v18")
DEVICE = "cuda:0"
SEQ_LENGTH = 2048
STRIDE = 512

def decompress_vbr_matrix(vbr_dict, device="cpu"):
    """Reverses the V18 Bit-Packing purely on the CPU"""
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

def calculate_perplexity(model, encodings):
    """Calculates sliding-window perplexity"""
    max_length = model.config.max_position_embeddings
    seq_len = min(SEQ_LENGTH, max_length)
    
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in tqdm(range(0, encodings.input_ids.size(1), STRIDE), desc="Evaluating", leave=False):
        end_loc = min(begin_loc + seq_len, encodings.input_ids.size(1))
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100 # Ignore loss for context tokens
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
            
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == encodings.input_ids.size(1):
            break
            
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

def main():
    print("[*] Loading Tokenizer and Dataset...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    print(f"[*] Loading Base Qwen 2.5 7B to {DEVICE}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE)
    model.eval()

    print("\n--- MEASURING BASELINE (FP16) ---")
    base_ppl = calculate_perplexity(model, encodings)
    print(f"[+] Base Model Perplexity: {base_ppl:.4f}")

    print("\n[*] Moving model to CPU to safely inject V18 compressed matrices...")
    model.to("cpu")
    torch.cuda.empty_cache()

    pt_files = sorted(glob.glob(os.path.join(VBR_DIR, "compressed_*.pt")))
    for pt_file in pt_files:
        print(f"    -> Unpacking {os.path.basename(pt_file)}...")
        chunk_data = torch.load(pt_file, map_location="cpu")["experts_cold"]
        
        for name, vbr_dict in tqdm(chunk_data.items(), desc="Injecting", leave=False):
            submodule = model.get_submodule(name)
            if vbr_dict["is_vbr_compressed"]:
                recon_weight = decompress_vbr_matrix(vbr_dict, device="cpu")
                submodule.weight.data = recon_weight
            else:
                submodule.weight.data = vbr_dict["raw_data"].cpu()
                
            if "bias" in vbr_dict and vbr_dict["bias"] is not None:
                submodule.bias.data = vbr_dict["bias"].cpu()
                
        del chunk_data
        gc.collect()

    print("\n[*] Moving V18 Compressed model back to GPU...")
    model.to(DEVICE)
    
    print("\n--- MEASURING V18 COMPRESSED ---")
    v18_ppl = calculate_perplexity(model, encodings)
    
    degradation = v18_ppl - base_ppl
    print("\n==========================================")
    print(f" BASE FP16 PERPLEXITY: {base_ppl:.4f}")
    print(f" V18 SWARM PERPLEXITY: {v18_ppl:.4f}")
    print(f" DEGRADATION:          +{degradation:.4f}")
    print("==========================================")

if __name__ == "__main__":
    main()
