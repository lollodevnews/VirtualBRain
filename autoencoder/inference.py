import torch
import os
import glob
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from process_prefill import dequantize_vbr_matrix

print("==========================================")
print(" VIRTUALBRAIN V30: WEIGHT VALIDATION TEST ")
print("==========================================")

MODEL_ID = os.path.expanduser("~/models/quant/qwen25_7b")
COMPRESSED_DIR = os.path.expanduser("~/models/quant/vbr_qwen25_v30")
DEVICE = "cuda:0"

def get_nested_module(m, name):
    for p in name.split("."): 
        m = getattr(m, p)
    return m

@torch.inference_mode()
def main():
    print("[1] Loading Base Model and Tokenizer (FP16)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # We load the full standard FP16 model first (Fits easily in MI50's 32GB)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=DEVICE
    )
    
    print(f"[2] Finding V30 Compressed Chunks in {COMPRESSED_DIR}...")
    pt_files = sorted(glob.glob(os.path.join(COMPRESSED_DIR, "*.pt")))
    if not pt_files:
        print("ERROR: No .pt files found!")
        return
        
    print("[3] Dequantizing and Injecting V30 Weights...")
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
                # Dequantize back to pure FP16 using your Python logic
                fp16_weight = dequantize_vbr_matrix(weight_dict, device=DEVICE)
                
                # Overwrite the standard HuggingFace weight
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
    
    # Vanilla HuggingFace Generate
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
lolollo@node-01:~/models/vbr_main$ 
