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

# =============================================================================
# 1. THE VIRTUAL MACHINE (DECODER)
# =============================================================================
class VirtualBRainLinear(nn.Module):
    """
    The LISP Machine Emulator. 
    Replaces standard nn.Linear. Unpacks 4-bit VBR routing masks on the fly,
    applies FP16 Dust Anchors, and unscrambles the dimensional space.
    """
    def __init__(self, in_features, out_features, packed_dict, prefix=""):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Load the decoupled physical components
        self.W_packed = nn.Parameter(packed_dict[f"{prefix}.W_packed_4bit"], requires_grad=False)
        self.col_mins = nn.Parameter(packed_dict[f"{prefix}.col_mins"], requires_grad=False)
        self.col_ranges = nn.Parameter(packed_dict[f"{prefix}.col_ranges"], requires_grad=False)
        self.inverse_indices = nn.Parameter(packed_dict[f"{prefix}.inverse_indices"], requires_grad=False)

    def forward(self, x):
        # 1. Bitwise Unpacking (Split the byte into two 4-bit integers)
        W_high = (self.W_packed >> 4) & 0x0F
        W_low = self.W_packed & 0x0F
        
        # Reconstruct the logic gates
        W_int4 = torch.empty((self.out_features, self.in_features), dtype=torch.uint8, device=x.device)
        W_int4[:, 0::2] = W_high
        W_int4[:, 1::2] = W_low
        
        # Cast to match the input activation precision (FP16)
        W_float = W_int4.to(x.dtype)
        
        # 2. Analog Amplification (The Resistors & Dust Anchor)
        W_shifted = (W_float / 15.0) * self.col_ranges
        W_reconstructed = W_shifted + self.col_mins
        
        # 3. Unscramble the Spatial Geometry
        W_unscrambled = W_reconstructed[self.inverse_indices]
        
        # 4. Fire the Token
        return F.linear(x, W_unscrambled)

# =============================================================================
# 2. THE DECOMPILER (ENCODER)
# =============================================================================
def pack_virtualBrain_vbr(W_fp16, entropy_threshold=0.10):
    """
    Calculates the 5% magnitude Dust Anchor and bit-packs the matrix into uint8.
    """
    out_features, in_features = W_fp16.shape
    
    # 1. Dust Anchor (Average of lowest 5% magnitude)
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
    
    # 2. Baseline Quantization
    W_scaled = (W_shifted / col_ranges) * 15.0
    W_int4 = torch.round(torch.abs(W_scaled)).to(torch.uint8)
    W_int4 = torch.clamp(W_int4, 0, 15)
    
    # 3. Entropy Sieve (Determines VBR Bit-Depth)
    bit_depths = torch.zeros(out_features, dtype=torch.uint8)
    for i in range(out_features):
        row = W_int4[i]
        bit2_active = ((row & 0b0100) >> 2).float().mean()
        bit1_active = ((row & 0b0010) >> 1).float().mean()
        if bit1_active > entropy_threshold: bit_depths[i] = 3
        elif bit2_active > entropy_threshold: bit_depths[i] = 2
        else: bit_depths[i] = 1
            
    # 4. VBR Sorting (Group by entropy)
    sorted_indices = torch.argsort(bit_depths)
    inverse_indices = torch.argsort(sorted_indices) 
    
    W_sorted = W_int4[sorted_indices]
    col_mins_sorted = col_mins[sorted_indices]
    col_ranges_sorted = col_ranges[sorted_indices]
    is_unsigned_sorted = is_unsigned[sorted_indices]
    bit_depths_sorted = bit_depths[sorted_indices]

    # 5. Physical Packing (Halve the filesize)
    W_paired = W_sorted.view(out_features, in_features // 2, 2)
    W_packed = (W_paired[:, :, 0] << 4) | W_paired[:, :, 1]
    
    return {
        "W_packed_4bit": W_packed,
        "col_mins": col_mins_sorted.half(),
        "col_ranges": col_ranges_sorted.half(),
        "is_unsigned": is_unsigned_sorted,
        "bit_depths": bit_depths_sorted,
        "inverse_indices": inverse_indices
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
    BRESCIA_FILE = "qwen_virtualBrain_test.pt"
    
    # Check if the user already ran the compressor (which you did!)
    if not os.path.exists(BRESCIA_FILE):
        print("VirtualBRain file not found. Run your run_compressor_vbr.py first!")
        exit()

    # 1. Boot up the Virtual Machine
    model = load_virtualBrain_model(MODEL_ID, BRESCIA_FILE)
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
        outputs = model.generate(**
