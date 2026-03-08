import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from VirtualBRainEngine import load_virtualBrain_graph, detect_hardware

MODEL_ID = "Qwen/Qwen1.5-0.5B"
COMPILED_DIR = "./vbr_compiled_qwen_0.5B"
TEST_TEXT = """
The history of artificial intelligence (AI) began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain.
"""

def calculate_perplexity(model, tokenizer, text, device, compute_dtype):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    # Ensure inputs are correctly formatted for loss calculation
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        # Passing labels=input_ids forces the HF wrapper to calculate CrossEntropyLoss
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss)
        
    return perplexity.item()

def measure_throughput(model, tokenizer, device, num_tokens=50):
    prompt = "In the future, the development of quantum computing will"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print(f"Generating {num_tokens} tokens...")
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=num_tokens, do_sample=False)
        
    end_time = time.time()
    generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    total_time = end_time - start_time
    tps = generated_tokens / total_time
    
    return tps, tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    device, compute_dtype = detect_hardware()
    print(f"--- VIRTUALBRAIN BENCHMARK SUITE ---")
    print(f"Hardware: {device.upper()} | Precision: {compute_dtype}\n")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 1. BASELINE FP16 TEST
    print("Loading Baseline FP16 Model (Hugging Face Original)...")
    baseline_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
    baseline_ppl = calculate_perplexity(baseline_model, tokenizer, TEST_TEXT, device, compute_dtype)
    print(f"Baseline FP16 Perplexity: {baseline_ppl:.4f}")
    del baseline_model # Free RAM
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print("\n" + "="*50 + "\n")
    
    # 2. VIRTUALBRAIN TEST
    print("Loading VBR Compiled Model...")
    vbr_model = load_virtualBrain_graph(MODEL_ID, COMPILED_DIR, device, compute_dtype)
    vbr_ppl = calculate_perplexity(vbr_model, tokenizer, TEST_TEXT, device, compute_dtype)
    print(f"VBR Engine Perplexity: {vbr_ppl:.4f}")
    
    # 3. DEGRADATION CALCULATION
    degradation = vbr_ppl - baseline_ppl
    print(f"Absolute Perplexity Degradation: +{degradation:.4f}")
    
    print("\n" + "="*50 + "\n")
    
    # 4. THROUGHPUT TEST
    print("Running Throughput Speed Test (VBR Engine)...")
    tps, text = measure_throughput(vbr_model, tokenizer, device, num_tokens=50)
    print(f"\nThroughput: {tps:.2f} Tokens / Second")
    print(f"Sample Output:\n{text}")
