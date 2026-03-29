import os

# 1. HARDWARE CONFIGURATION
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
os.environ["HSA_ENABLE_SDMA"] = "0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "max_split_size_mb:128"

# 2. STANDARD IMPORTS
import torch
import glob
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ELEUTHERAI HARNESS IMPORTS
import lm_eval
from lm_eval.models.huggingface import HFLM

## Applying GPU quirks
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ==========================================
# HYPERPARAMETERS
# ==========================================
USE_COMPRESSED_V36 = True  # <--- TOGGLE: True = Custom V36 | False = Standard FP16 Qwen
LIMIT_EVALS = None         # <--- TOGGLE: Set to 50 for a fast test, or None for the full official benchmark
# ==========================================

print("==========================================")
print("  LM-EVAL HARNESS BENCHMARK   ")
print(f" MODE: {'V36 COMPRESSED' if USE_COMPRESSED_V36 else 'STANDARD FP16 BASELINE'}")
print("==========================================")

MODEL_ID = os.path.expanduser("~/models/quant/qwen25_7b")
COMPRESSED_DIR = "./quant_qwen"
DEVICE = "cuda:0"

# ==========================================
# HIPBLAS WARM-UP HACK
# ==========================================
print("[*] Warming up HIPBLAS context...")
_dummy_a = torch.randn(16, 16, device=DEVICE)
_dummy_b = torch.randn(16, 16, device=DEVICE)
_ = _dummy_a @ _dummy_b
del _dummy_a, _dummy_b, _
torch.cuda.empty_cache()


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
        
        mag_idx = mag_accumulator.view(num_rows, in_features).long()
        
        divisor = float(K_bins - 1)
        base_bins = torch.arange(K_bins, device=device).float() / divisor
            
        a = alphas[row_indices].unsqueeze(1).float()
        c = cs[row_indices].unsqueeze(1).float()
        m = ms[row_indices].unsqueeze(1).float()
        s = scales[row_indices].unsqueeze(1).float()
        b = base_bins.unsqueeze(0)

        safe_b = torch.where(b == 0.0, 1e-9, b)
        m_term = torch.where(m == 0.0, 1.0, 
                    torch.where(b == 0.0, 0.0, safe_b ** m))
        
        inner = (1.0 - a) * b + a * m_term
        inner_clamped = torch.clamp(inner, 0.0, 1.0)
        
        safe_inner = torch.where(inner_clamped == 0.0, 1e-9, inner_clamped)
        warped_lut = torch.where(c == 0.0, 1.0, 
                    torch.where(inner_clamped == 0.0, 0.0, safe_inner ** c))

        gathered = torch.gather(warped_lut, 1, mag_idx)
        
        magnitudes = gathered * s
        sign_flip = torch.where(signs == 1, -1.0, 1.0).float()
        
        recon_matrix[row_indices] = (magnitudes * sign_flip).half()

    return recon_matrix

def get_module(m, name):
    for p in name.split("."): m = getattr(m, p)
    return m

@torch.inference_mode()
def main():
    print(f"\n[1] Loading Base Tokenizer & HF Architecture from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=DEVICE 
    )
    
    if USE_COMPRESSED_V36:
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
            torch.cuda.empty_cache()

        print(f"[*] AUDIT: Updated {total_updated} linear layers out of {total_found} found.")
    else:
        print("\n[2] Skipping V36 Injection -> Running Standard FP16 Baseline.")

    print("\n[3] Wrapping Model in EleutherAI HFLM Interface...")
    # Wrap your custom/restored model so lm-eval can talk to it
    lm_eval_wrapper = HFLM(
        pretrained=model, 
        tokenizer=tokenizer, 
        batch_size=1
    )

    # Standard polygraph tasks: ARC (Reasoning), HellaSwag (Common Sense), MMLU (Knowledge)
    tasks = ["arc_challenge", "hellaswag", "mmlu"]
    
    print(f"\n[4] Firing up the harness for tasks: {tasks}")
    if LIMIT_EVALS:
        print(f"[*] WARNING: Fast-run mode enabled. Limited to {LIMIT_EVALS} questions per task.")

    results = lm_eval.simple_evaluate(
        model=lm_eval_wrapper,
        tasks=tasks,
        num_fewshot=0,
        limit=LIMIT_EVALS
    )

    print("\n" + "="*60)
    print(f" RESULTS: {'V36 COMPRESSED' if USE_COMPRESSED_V36 else 'FP16 BASELINE'}")
    print("="*60)
    # This automatically formats a beautiful Markdown table in your terminal
    print(lm_eval.utils.make_table(results))

if __name__ == "__main__":
    main()
