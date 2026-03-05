# ==============================================================================
# VirtualBRain (VBR) - Offline VBR Packer
# ==============================================================================
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
import os
from transformers import AutoModelForCausalLM

def compress_llm_to_virtualBrain(model_id, output_file, entropy_threshold):
    print(f"=== Initiating Project VirtualBRain Decompiler ===")
    print(f"Target Model: {model_id}")
    print(f"Loading FP16 weights into System RAM (This may take a minute)...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="cpu"
    )
    
    packed_state_dict = {}
    total_original_bytes = 0
    total_packed_bytes = 0
    
    print("\n--- Starting Matrix Entropy Sieve ---")
    
    for name, param in model.named_parameters():
        original_bytes = param.element_size() * param.nelement()
        total_original_bytes += original_bytes
        
        if "weight" in name and param.dim() == 2 and "embed" not in name and "lm_head" not in name:
            print(f"Compressing: {name} | Shape: {list(param.shape)}")
            
            packed_data = pack_virtualBrain_vbr(param.data, entropy_threshold)
            
            packed_bytes = (
                packed_data["W_packed_4bit"].element_size() * packed_data["W_packed_4bit"].nelement() +
                packed_data["col_mins"].element_size() * packed_data["col_mins"].nelement() +
                packed_data["vbr_scales"].element_size() * packed_data["vbr_scales"].nelement() +
                packed_data["bit_depths"].element_size() * packed_data["bit_depths"].nelement() +
                packed_data["inverse_indices"].element_size() * packed_data["inverse_indices"].nelement()
            )
            total_packed_bytes += packed_bytes
            
            packed_state_dict[f"{name}.W_packed_4bit"] = packed_data["W_packed_4bit"]
            packed_state_dict[f"{name}.col_mins"] = packed_data["col_mins"]
            packed_state_dict[f"{name}.vbr_scales"] = packed_data["vbr_scales"] # THE NEW PRE-DIVIDED ANCHOR
            packed_state_dict[f"{name}.is_unsigned"] = packed_data["is_unsigned"]
            packed_state_dict[f"{name}.bit_depths"] = packed_data["bit_depths"]
            packed_state_dict[f"{name}.inverse_indices"] = packed_data["inverse_indices"]
            
        else:
            print(f"Bypassing:   {name} (Preserving exact FP16)")
            packed_state_dict[name] = param.data
            total_packed_bytes += original_bytes

    print("\n=== Compression Complete ===")
    print(f"Saving virtual machine state to: {output_file}")
    torch.save(packed_state_dict, output_file)
    
    orig_gb = total_original_bytes / (1024**3)
    pack_gb = total_packed_bytes / (1024**3)
    print(f"\nOriginal Model Size: {orig_gb:.2f} GB")
    print(f"VirtualBRain VBR Size:    {pack_gb:.2f} GB")
    print(f"Total Reduction:     {((orig_gb - pack_gb) / orig_gb) * 100:.1f}%")


def pack_virtualBrain_vbr(W_fp16, entropy_threshold):
    out_features, in_features = W_fp16.shape
    
    # --- 1. THE DUST ANCHOR ---
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
    
    # --- 2. THE ENTROPY SIEVE (Find the optimal bit-depth) ---
    W_scaled_test = (W_shifted / col_ranges) * 15.0
    W_int4_test = torch.round(torch.abs(W_scaled_test)).to(torch.uint8).clamp(0, 15)
    
    test_matrix = W_int4_test.clone()
    mask = test_matrix >= 8
    test_matrix[mask] = test_matrix[mask] ^ 0b0111 

    bit_depths = torch.zeros(out_features, dtype=torch.uint8)
    for i in range(out_features):
        row = test_matrix[i]
        bit2_active = ((row & 0b0100) >> 2).float().mean()
        bit1_active = ((row & 0b0010) >> 1).float().mean()
        
        if bit1_active > entropy_threshold:
            bit_depths[i] = 4
        elif bit2_active > entropy_threshold:
            bit_depths[i] = 3
        else:
            # We skip 1-bit here to allow for 2-bit (states 0, 1, 2, 3) as the baseline minimum
            bit_depths[i] = 2 
            
    # --- 3. THE DIVISION-FREE VBR SCALING ---
    # Calculate the discrete denominators: (2^b - 1) -> 3, 7, or 15
    states_max = (2 ** bit_depths.float().unsqueeze(1)) - 1
    
    # Pre-divide the amplitude by the states. This is the magic VBR anchor.
    vbr_scales = col_ranges / states_max
    
    # Re-quantize the actual weights to their exact integer states using the new anchor
    W_vbr_int = torch.round(torch.abs(W_shifted) / vbr_scales).to(torch.uint8)
            
    # --- 4. THE VBR SORTING ---
    sorted_indices = torch.argsort(bit_depths)
    inverse_indices = torch.argsort(sorted_indices) 
    
    W_sorted = W_vbr_int[sorted_indices]
    col_mins_sorted = col_mins[sorted_indices]
    vbr_scales_sorted = vbr_scales[sorted_indices] # Sorted anchors
    is_unsigned_sorted = is_unsigned[sorted_indices]
    bit_depths_sorted = bit_depths[sorted_indices]

    # --- 5. THE PHYSICAL PACKING ---
    assert in_features % 2 == 0, "Input features must be even for packing."
    W_paired = W_sorted.view(out_features, in_features // 2, 2)
    W_packed = (W_paired[:, :, 0] << 4) | W_paired[:, :, 1]
    
    return {
        "W_packed_4bit": W_packed,
        "col_mins": col_mins_sorted.half(),
        "vbr_scales": vbr_scales_sorted.half(), # The pre-divided FP16 multiplier
        "is_unsigned": is_unsigned_sorted,
        "bit_depths": bit_depths_sorted,
        "inverse_indices": inverse_indices
    }

if __name__ == "__main__":
    compress_llm_to_virtualBrain(model_id="Qwen/Qwen1.5-0.5B", output_file="qwen_virtualBrain_test.pt", entropy_threshold=0.05)
