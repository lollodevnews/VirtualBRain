import torch
import os
import glob
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from process_prefill import dequantize_vbr_matrix

print("==========================================")
print(" VIRTUALBRAIN V30: PERPLEXITY BENCHMARK   ")
print("==========================================")

MODEL_ID = os.path.expanduser("~/models/quant/qwen25_7b")
COMPRESSED_DIR = os.path.expanduser("~/models/quant/vbr_qwen25_v30")
DEVICE = "cuda:0"

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
    
    print(f"[2] Injecting V30 Weights from {COMPRESSED_DIR}...")
    pt_files = sorted(glob.glob(os.path.join(COMPRESSED_DIR, "*.pt")))
    
    for pt_file in pt_files:
        chunk_data = torch.load(pt_file, map_location="cpu")["experts_cold"]
        for key, weight_dict in chunk_data.items():
            module_path = key.replace(".weight", "") if key.endswith(".weight") else key
            try: target_module = get_nested_module(model, module_path)
            except AttributeError: continue

            if weight_dict.get("is_vbr_compressed", False):
                fp16_weight = dequantize_vbr_matrix(weight_dict, device=DEVICE)
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
    # Cap stride/max_len to 2048 to keep evaluation fast but accurate
    max_length = min(2048, max_length) 
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    
    print(f"[4] Calculating Perplexity (Context: {max_length}, Stride: {stride})...")
    
    # Standard sliding window PPL evaluation
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # Loss is calculated using CrossEntropyLoss which averages over valid labels
            # Multiply by trg_len to get the total log-likelihood for the chunk
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    print("\n==========================================")
    print(f" V30 FINAL PERPLEXITY (WikiText-2): {ppl.item():.4f}")
    print("==========================================\n")

if __name__ == "__main__":
    main()
