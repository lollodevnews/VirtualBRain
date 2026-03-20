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
import json
import shutil
from transformers import AutoModelForCausalLM
from tqdm import tqdm

# ==============================================================================
# 🎛️ SYSTEM CONTROL DECK (PHASE 4: ADAPTIVE GAMMA HYBRID)
# ==============================================================================
MODEL_ID = "Qwen/Qwen1.5-0.5B"
OUTPUT_DIR = "./vbr_compiled_qwen_0.5B"

# Architecture Routing
COMPRESS_ATTENTION_FP_HACK = True  
TARGET_HACK_BITS = 8               

COMPRESS_MLP_VBR = True            
TARGET_PRECISION = 0.05
# ==============================================================================

def pack_poor_mans_fp(W_fp16, target_bits=8):
    """Zero-math bit truncation for Attention geometry."""
    drop_bits = 16 - target_bits
    mask = (1 << target_bits) - 1 
    
    W_raw_int16 = W_fp16.contiguous().view(torch.int16)
    W_packed = ((W_raw_int16 >> drop_bits) & mask).to(torch.uint8)
    
    return {
        "W_packed_fp_hack": W_packed,
        "target_bits": target_bits
    }

def pack_true_vbr_signed_magnitude(W_fp16):
    """Phase 9: True Mixed-Precision VBR (1-bit, 2-bit, 4-bit Magnitude)"""
    out_features, in_features = W_fp16.shape
    
    W_f32 = W_fp16.to(torch.float32)
    M = W_f32.abs()
    S = torch.where(W_f32 >= 0, torch.tensor(1, dtype=torch.int8, device=W_f32.device), 
                                torch.tensor(-1, dtype=torch.int8, device=W_f32.device))
    
    # Dynamic Alpha
    ceiling = torch.quantile(M, 0.995, dim=1, keepdim=True)
    col_maxs = M.max(dim=1, keepdim=True).values.clamp(min=1e-9)
    alphas = (ceiling / col_maxs).clamp(min=0.01, max=1.0)
    
    # Tournament Trackers
    best_recon_error = torch.full((out_features,), float('inf'), device=W_f32.device)
    best_M_int = torch.zeros((out_features, in_features), dtype=torch.uint8, device=W_f32.device)
    best_powers = torch.zeros((out_features,), dtype=torch.float32, device=W_f32.device)
    
    # NEW: Tracks the exact scalar divisor for the engine (defaults to 4-bit / 15.0)
    best_divisors = torch.full((out_features,), 15.0, dtype=torch.float32, device=W_f32.device) 
    
    # NEW: Boolean mask to lock rows once they achieve < 5% error
    row_satisfied = torch.zeros((out_features,), dtype=torch.bool, device=W_f32.device)
    
    powers = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0]
    
    # Test ascending bit-depths!
    bit_configs = [1, 2, 4] 
    
    for bits in bit_configs:
        states_max = (2 ** bits) - 1
        
        for p in powers:
            # Generate Curve
            x_norm = torch.linspace(0, 1, states_max + 1, device=W_f32.device, dtype=torch.float32).view(1, -1)
            curve_norm = (alphas * x_norm) + ((1.0 - alphas) * torch.pow(x_norm, p))
            bin_values = col_maxs * curve_norm
            
            # LUT Mapping
            distances = torch.abs(M.unsqueeze(-1) - bin_values.unsqueeze(1))
            M_int = torch.argmin(distances, dim=-1)
            M_recon = torch.gather(bin_values, 1, M_int)
            
            # Relative Error Judge
            relative_error = torch.mean( ((M - M_recon) / (M + 1e-4)) ** 2, dim=1 )
            
            # Only update rows that haven't hit the 5% target at a lower bit-depth
            valid_mask = ~row_satisfied
            better_mask = (relative_error < best_recon_error) & valid_mask
            
            # Update metrics
            best_recon_error[better_mask] = relative_error[better_mask]
            best_M_int[better_mask] = M_int[better_mask].to(torch.uint8)
            best_powers[better_mask] = p
            best_divisors[better_mask] = float(states_max)
        
        # End of bit-depth tournament: Did any new rows achieve < 5% Mean Relative Error?
        # 5% squared is 0.0025.
        threshold_mask = (best_recon_error < 0.0025)
        row_satisfied = row_satisfied | threshold_mask

    # Optional: Print out the distribution so you can see the VBR in action!
    print(f"1-Bit Rows: {(best_divisors == 1.0).sum().item()} | 2-Bit Rows: {(best_divisors == 3.0).sum().item()} | 4-Bit Rows: {(best_divisors == 15.0).sum().item()}")

    return {
        "W_mag_int": best_M_int,              
        "W_sign": S,                          
        "vbr_scales": col_maxs.half(),        
        "alphas": alphas.half(),
        "row_powers": best_powers.unsqueeze(1).half(),
        "row_divisors": best_divisors.unsqueeze(1).half() # Passes the exact step-size!
    }

def compile_model_stream():
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    print(f"Loading Source Model {MODEL_ID} into CPU RAM...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cpu")
    
    attn_name = "self_attn"
    mlp_name = "mlp"
    ln1_name = "input_layernorm"
    ln2_name = "post_attention_layernorm"

    config_data = {
        "attention_module": attn_name,
        "mlp_module": mlp_name,
        "input_layernorm": ln1_name,
        "post_attention_layernorm": ln2_name
    }
    with open(os.path.join(OUTPUT_DIR, "vbr_config.json"), "w") as f:
        json.dump(config_data, f)

    system_tensors = {
        "model.embed_tokens.weight": model.model.embed_tokens.weight.data,
        "model.norm.weight": model.model.norm.weight.data,
        "lm_head.weight": model.lm_head.weight.data
    }
    torch.save(system_tensors, os.path.join(OUTPUT_DIR, "system_analog.pt"))

    for i, layer in enumerate(tqdm(model.model.layers, desc="Compiling Layers")):
        attention_hot = {}
        experts_cold = {}
        
        for name, param in layer.named_parameters():
            is_2d_weight = "weight" in name and param.dim() == 2
            is_attn = attn_name in name
            is_mlp = mlp_name in name
            
            if is_2d_weight and is_attn and COMPRESS_ATTENTION_FP_HACK:
                attention_hot[name] = pack_true_vbr_signed_magnitude(param.data)
                #attention_hot[name] = pack_poor_mans_fp(param.data, TARGET_HACK_BITS)
            elif is_2d_weight and is_mlp and COMPRESS_MLP_VBR:
                experts_cold[name] = pack_true_vbr_signed_magnitude(param.data)
                #experts_cold[name] = pack_poor_mans_fp(param.data, TARGET_HACK_BITS)
            else:
                # Bypass / Layernorms / Biases
                if is_attn: attention_hot[name] = param.data
                else: experts_cold[name] = param.data

        torch.save({"attention_hot": attention_hot, "experts_cold": experts_cold}, os.path.join(OUTPUT_DIR, f"layer_{i:02d}.pt"))

if __name__ == "__main__":
    compile_model_stream()
    print("=== Compilation Complete ===")
