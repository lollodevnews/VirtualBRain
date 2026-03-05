# ==============================================================================
# VirtualBRain (VBR) - A LISP-style virtual machine for LLM brains
# 
# Copyright (c) 2026 [lollodevnews]
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# ==============================================================================
# VirtualBRain (VBR) - Runtime Execution Engine
# ==============================================================================

class VirtualBrainLinear:
    def __init__(self, packed_data_dict):
        """
        Loads the pre-compiled VBR matrix.
        """
        self.W_packed = packed_data_dict["W_packed_4bit"]
        self.col_mins = packed_data_dict["col_mins"]
        self.vbr_scales = packed_data_dict["vbr_scales"] # The pre-divided multiplier
        self.inverse_indices = packed_data_dict["inverse_indices"]
        
        # Determine shapes
        self.out_features = self.W_packed.shape[0]
        self.in_features = self.W_packed.shape[1] * 2

    def forward(self, x):
        """
        Executes the forward pass with zero division overhead.
        """
        # 1. Unpack the 4-bit containers into flat integers
        W_flat = torch.zeros((self.out_features, self.in_features), dtype=torch.uint8, device=x.device)
        W_flat[:, 0::2] = (self.W_packed >> 4) & 0x0F
        W_flat[:, 1::2] = self.W_packed & 0x0F
        
        # 2. Reconstruct the FP16 matrix (Multiplication ONLY)
        # We multiply the integer directly by the pre-divided vbr_scale, then add the dust anchor.
        W_reconstructed = (W_flat.to(torch.float16) * self.vbr_scales) + self.col_mins
        
        # 3. Unsort the matrix using the inverse indices (The Router)
        W_unsorted = W_reconstructed[self.inverse_indices]
        
        # 4. Standard FP16 Matrix Multiplication against the token stream
        return torch.matmul(x, W_unsorted.t())


# =============================================================================
# 2. THE DECOMPILER (ENCODER) - 4-BIT BASELINE WITH FIXED SIEVE
# =============================================================================
def pack_virtualBrain_vbr(W_fp16, entropy_threshold=0.10):
    out_features, in_features = W_fp16.shape
    
    # --- 1. Dust Anchor ---
    magnitudes = torch.abs(W_fp16)
    k_bottom = max(1, int(in_features * 0.05))
    _, lowest_mag_indices = torch.topk(magnitudes, k_bottom, dim=1, largest=False)
    lowest_weights = torch.gather(W_fp16, 1, lowest_mag_indices)
    col_mins = lowest_weights.mean(dim=1, keepdim=True)
    
    W_shifted = W_fp16 - col_mins
    
    max_vals = W_shifted.max(dim=1, keepdim=True).values
    min_vals = W_shifted.min(dim=1, keepdim=True).values
    is_unsigned = (torch.abs(min_vals) < (0.05 * max_vals))
    col_ranges = torch.max(torch.abs(max_vals), torch.abs(min_vals)).clamp_(min=1e-9)
    
    # --- 2. Baseline Quantization (Everything to 0-15 for the 4-bit baseline) ---
    W_scaled = (W_shifted / col_ranges) * 15.0
    W_int4 = torch.round(torch.abs(W_scaled)).to(torch.uint8)
    W_int4 = torch.clamp(W_int4, 0, 15)
    
    # --- 3. Entropy Sieve (The XOR Fold Fix) ---
    test_matrix = W_int4.clone()
    mask = test_matrix >= 8
    test_matrix[mask] = test_matrix[mask] ^ 0b0111 # Fold the upper half to quiet the LSBs

    bit_depths = torch.zeros(out_features, dtype=torch.uint8)
    for i in range(out_features):
        row = test_matrix[i] # Use the folded test matrix!
        bit2_active = ((row & 0b0100) >> 2).float().mean()
        bit1_active = ((row & 0b0010) >> 1).float().mean()
        
        if bit1_active > entropy_threshold: bit_depths[i] = 3
        elif bit2_active > entropy_threshold: bit_depths[i] = 2
        else: bit_depths[i] = 1
            
    # --- 4. VBR Sorting (Using the clean Sieve Map) ---
    sorted_indices = torch.argsort(bit_depths)
    inverse_indices = torch.argsort(sorted_indices) 
    
    W_sorted = W_int4[sorted_indices] # Use the untouched, actual W_int4!
    col_mins_sorted = col_mins[sorted_indices]
    col_ranges_sorted = col_ranges[sorted_indices]
    is_unsigned_sorted = is_unsigned[sorted_indices]
    bit_depths_sorted = bit_depths[sorted_indices]

    # --- 5. Physical Packing (Uniform 4-bit for v1.0) ---
    assert in_features % 2 == 0, "Input features must be even for 4-bit packing."
    W_paired = W_sorted.view(out_features, in_features // 2, 2)
    W_packed = (W_paired[:, :, 0] << 4) | W_paired[:, :, 1]
    
    return {
        "W_packed_4bit": W_packed,           # Physically 50% smaller!
        "col_mins": col_mins_sorted.half(),
        "col_ranges": col_ranges_sorted.half(),
        "is_unsigned": is_unsigned_sorted,
        "bit_depths": bit_depths_sorted,     # Sieve Map ready for Phase 2
        "inverse_indices": inverse_indices   # The Router Key
    }

# =============================================================================
# 3. THE SURGEON (INJECTOR)
# =============================================================================
def replace_with_virtualBrain(module, name_prefix, packed_dict):
    """
    Recursively walks the Hugging Face model tree. If it finds an nn.Linear layer
    that exists in our compressed dictionary, it surgically replaces it with the
    VirtualBRain Virtual Machine.
    """
    for child_name, child in module.named_children():
        full_name = f"{name_prefix}.{child_name}" if name_prefix else child_name
        
        # If this is a linear layer AND we have packed data for it
        if isinstance(child, nn.Linear) and f"{full_name}.W_packed_4bit" in packed_dict:
            # Create the Virtual Machine emulator
            virtualBrain_layer = VirtualBRainLinear(
                in_features=child.in_features, 
                out_features=child.out_features, 
                packed_dict=packed_dict, 
                prefix=full_name
            )
            # Perform the surgery
            setattr(module, child_name, virtualBrain_layer)
        else:
            # Keep digging down the tree
            replace_with_virtualBrain(child, full_name, packed_dict)

def load_virtualBrain_model(model_id, virtualBrain_file):
    """
    Loads the base HuggingFace model skeleton, loads your custom VBR weights,
    and runs the injection sequence.
    """
    print(f"Loading base skeleton for {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cpu")
    
    print(f"Loading Project VirtualBRain VBR Engine from {virtualBrain_file}...")
    packed_dict = torch.load(virtualBrain_file, map_location="cpu", weights_only=True)
    
    print("Executing structural injection (Swapping CISC for RISC layers)...")
    replace_with_virtualBrain(model, "", packed_dict)
    
    # Load the remaining uncompressed parts (like embeddings and lm_head)
    print("Restoring uncompressed analog pathways...")
    missing, unexpected = model.load_state_dict(packed_dict, strict=False)
    
    print("Injection Complete. LISP Machine Online.")
    return model

# =============================================================================
# 4. EXECUTION (THE TEST)
# =============================================================================
if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen1.5-0.5B"
    BRAIN_FILE = "qwen_virtualBrain_test.pt"
    
    # Check if the user already ran the compressor (which you did!)
    if not os.path.exists(BRAIN_FILE):
        print("VirtualBRain file not found. Run your importer.py first!")
        exit()

    # 1. Boot up the Virtual Machine
    model = load_virtualBrain_model(MODEL_ID, BRAIN_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Move the grafted model to your GPU for testing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    print(f"\nModel successfully mounted on {device.upper()}.")
    
    # 2. The First Token Generation
    prompt = "The concept of a Turing machine can be explained as"
    print(f"\nUser: {prompt}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print("\nVirtualBRain Engine generating...")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nResponse:\n{response}")
