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
# Make sure your pack_virtualBrain_vbr function is either in this file or imported
# from pack_qwen_vbr import pack_virtualBrain_vbr 
# for the ease of use of nontechnical people i've included the packing function at the end

def compress_llm_to_virtualBrain(model_id, output_file, entropy_threshold):
    print(f"=== Initiating Project VirtualBRain Decompiler ===")
    print(f"Target Model: {model_id}")
    print(f"Loading FP16 weights into System RAM (This may take a minute)...")
    
    # Load the model strictly on CPU to save VRAM during the packing phase
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="cpu"
    )
    
    packed_state_dict = {}
    total_original_bytes = 0
    total_packed_bytes = 0
    
    print("\n--- Starting Matrix Entropy Sieve ---")
    
    # Iterate through every parameter in the neural network
    for name, param in model.named_parameters():
        original_bytes = param.element_size() * param.nelement()
        total_original_bytes += original_bytes
        
        # We only want to compress the massive 2D dense matrices (up_proj, down_proj, etc.)
        # We leave 1D structures (LayerNorm, biases, embeddings) in native precision
        if "weight" in name and param.dim() == 2 and "embed" not in name and "lm_head" not in name:
            print(f"Compressing: {name} | Shape: {list(param.shape)}")
            
            # Run the matrix through your VBR Engine
            packed_data = pack_virtualBrain_vbr(param.data, entropy_threshold)
            
            # Calculate new byte size (uint8 matrix + FP16 vectors + metadata)
            packed_bytes = (
                packed_data["W_packed_4bit"].element_size() * packed_data["W_packed_4bit"].nelement() +
                packed_data["col_mins"].element_size() * packed_data["col_mins"].nelement() +
                packed_data["col_ranges"].element_size() * packed_data["col_ranges"].nelement() +
                packed_data["bit_depths"].element_size() * packed_data["bit_depths"].nelement() +
                packed_data["inverse_indices"].element_size() * packed_data["inverse_indices"].nelement()
            )
            total_packed_bytes += packed_bytes
            
            # Save the decoupled RISC components into the new dictionary
            packed_state_dict[f"{name}.W_packed_4bit"] = packed_data["W_packed_4bit"]
            packed_state_dict[f"{name}.col_mins"] = packed_data["col_mins"]
            packed_state_dict[f"{name}.col_ranges"] = packed_data["col_ranges"]
            packed_state_dict[f"{name}.is_unsigned"] = packed_data["is_unsigned"]
            packed_state_dict[f"{name}.bit_depths"] = packed_data["bit_depths"]
            packed_state_dict[f"{name}.inverse_indices"] = packed_data["inverse_indices"]
            
        else:
            # Pass through uncompressed components
            print(f"Bypassing:   {name} (Preserving exact FP16)")
            packed_state_dict[name] = param.data
            total_packed_bytes += original_bytes

    print("\n=== Compression Complete ===")
    print(f"Saving virtual machine state to: {output_file}")
    torch.save(packed_state_dict, output_file)
    
    # --- The Moment of Truth ---
    orig_gb = total_original_bytes / (1024**3)
    pack_gb = total_packed_bytes / (1024**3)
    print(f"\nOriginal Model Size: {orig_gb:.2f} GB")
    print(f"VirtualBRain VBR Size:    {pack_gb:.2f} GB")
    print(f"Total Reduction:     {((orig_gb - pack_gb) / orig_gb) * 100:.1f}%")


def pack_virtualBrain_vbr(W_fp16, entropy_threshold):
    """
    Decompiles, sorts, and physically byte-packs a dense matrix.
    """
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
    
    # --- 2. THE 4-BIT BASELINE ---
    # Scale to 0-15
    W_scaled = (W_shifted / col_ranges) * 15.0
    W_int4 = torch.round(torch.abs(W_scaled)).to(torch.uint8)
    W_int4 = torch.clamp(W_int4, 0, 15)
    
    # --- 3. THE ENTROPY SIEVE (Bitwise Fold Test Matrix) ---
    # Create a test matrix to evaluate resolution independently of amplitude
    test_matrix = W_int4.clone()
    
    # If MSB is 1 (value >= 8), negate the lower 3 bits using XOR (0b0111 is 7)
    # This folds 15 (1111) down to 8 (1000), hushing the lower bits for extremes.
    mask = test_matrix >= 8
    test_matrix[mask] = test_matrix[mask] ^ 0b0111 

    bit_depths = torch.zeros(out_features, dtype=torch.uint8)
    
    for i in range(out_features):
        row = test_matrix[i] # Evaluate the folded test matrix!
        
        # Check activity on the interior bits
        bit2_active = ((row & 0b0100) >> 2).float().mean()
        bit1_active = ((row & 0b0010) >> 1).float().mean()
        
        if bit1_active > entropy_threshold:
            bit_depths[i] = 3
        elif bit2_active > entropy_threshold:
            bit_depths[i] = 2
        else:
            bit_depths[i] = 1
            
    # --- 4. THE PACKER ---
    # Now you proceed to pack using the REAL values in W_int4, 
    # guided by the accurate bit_depths array.
            
    # --- 4. THE VBR SORTING (Reordering the Matrix) ---
    # Group all 1-bit, 2-bit, and 3-bit logic gates together
    sorted_indices = torch.argsort(bit_depths)
    
    # We MUST save the inverse indices so the Engine can unsort the output token later!
    inverse_indices = torch.argsort(sorted_indices) 
    
    W_sorted = W_int4[sorted_indices]
    col_mins_sorted = col_mins[sorted_indices]
    col_ranges_sorted = col_ranges[sorted_indices]
    is_unsigned_sorted = is_unsigned[sorted_indices]
    bit_depths_sorted = bit_depths[sorted_indices]

    # --- 5. THE PHYSICAL 4-BIT PACKING (Halving the filesize) ---
    # W_sorted is currently [out_features, in_features] in uint8 (1 byte per weight)
    # We will pack pairs of weights: [W0, W1] -> (W0 << 4) | W1
    # This physically cuts the tensor size in half.
    
    # Ensure in_features is even (it always is in Qwen, e.g., 4096, 11008)
    assert in_features % 2 == 0, "Input features must be even for 4-bit packing."
    
    # Reshape to pair adjacent weights: [out_features, in_features // 2, 2]
    W_paired = W_sorted.view(out_features, in_features // 2, 2)
    
    # Shift the first weight 4 bits left, and bitwise OR it with the second weight
    W_packed = (W_paired[:, :, 0] << 4) | W_paired[:, :, 1]
    
    return {
        "W_packed_4bit": W_packed,           # Physically 50% smaller!
        "col_mins": col_mins_sorted.half(),  # FP16 Anchor
        "col_ranges": col_ranges_sorted.half(), # FP16 Amplitude
        "is_unsigned": is_unsigned_sorted,   # 1-bit Flags
        "bit_depths": bit_depths_sorted,     # Sieve Map
        "inverse_indices": inverse_indices   # The Router Key
    }

# Example Usage:
# packed_layer = pack_virtualBrain_vbr(qwen_up_proj_weight, entropy_threshold=0.05)
# torch.save(packed_layer, "qwen_layer_10_up_proj.virtualBrain")

if __name__ == "__main__":
    # Point this to your local llm directory or a HuggingFace repo ID
    # Adjust the model_id to exactly what you have on your drive
    compress_llm_to_virtualBrain(model_id="Qwen/Qwen1.5-0.5B", output_file="qwen_virtualBrain_test.pt", entropy_threshold=0.05)
