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

import os
import gc
import math
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

# ==========================================
# HYPERPARAMETERS & SETTINGS
# ==========================================
MODEL_ID = os.path.expanduser("~/models/quant/qwen25_7b") #"meta-llama/Llama-2-7b-hf" # <-- Update to your exact model path
OUTPUT_DIR = "./quant_qwen"

# VBR TOLERANCE SETTINGS - FIDELITY RUN
C_MAX = 4
ATTENTION_ERROR = 0.02  # Strict tolerance for Q, K, V, O
EXPERT_ERROR    = 0.08  # Loose tolerance for FFN Gate, Up, Down
DEFAULT_ERROR   = 0.01

LENIENCY_MASK_DEFAULT =  {2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0}
LENIENCY_MASK_EXPERT  =  {2: 2.0, 3: 1.5, 4: 1.2, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0}
LENIENCY_MASK_ATTN    =  {2: 2.0, 3: 1.5, 4: 1.2, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0}

'''
# VBR TOLERANCE SETTINGS - EXTREME COMPRESSION RUN
C_MAX = 4
ATTENTION_ERROR = 0.04  # Strict tolerance for Q, K, V, O
EXPERT_ERROR    = 0.17  # Loose tolerance for FFN Gate, Up, Down
DEFAULT_ERROR   = 0.01

LENIENCY_MASK_DEFAULT =  {2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0}
LENIENCY_MASK_EXPERT  =  {2: 1.5, 3: 1.3, 4: 1.1, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0}
LENIENCY_MASK_ATTN    =  {2: 1.5, 3: 1.3, 4: 1.1, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0}
'''
os.makedirs(OUTPUT_DIR, exist_ok=True)

REFINEMENT_AREA_1 = 0.10  # 10% of the global range
REFINEMENT_AREA_2 = 0.04  # 3% of the global range

N_stage0 = 2048
N_stage1 = 2048
N_stage2 = 2048

@torch.no_grad()
def compress_vbr_v35_matrix(name, weight_tensor, max_energy_error, leniency_map):
    """
    V35 Compositional Variable Bitrate Quantizer (Pure Monte Carlo Edition)
    Features: Inverted Topology (m inside, c outside) & Continuous Linkage
    """
    device = weight_tensor.device
    rows, cols = weight_tensor.shape
    
    if leniency_map is None:
        leniency_map = LENIENCY_MASK_DEFAULT

    # 1. Row-wise Absolute Max Scaling
    row_max = torch.max(torch.abs(weight_tensor), dim=1)[0].clamp(min=1e-8)
    normalized_weights = torch.abs(weight_tensor) / row_max.unsqueeze(1)
    
    # 2. Output Tensors
    final_bitrates = torch.full((rows,), 8, dtype=torch.int8, device=device)
    final_a = torch.zeros(rows, dtype=torch.float32, device=device)
    final_c = torch.zeros(rows, dtype=torch.float32, device=device)
    final_m = torch.zeros(rows, dtype=torch.float32, device=device)
    
    active_indices = torch.arange(rows, device=device)
    
    for current_D in range(2, 9):
        if len(active_indices) == 0:
            break

        mag_bits = current_D - 1
        K_bins = 2 ** mag_bits
        M_MAX = float(K_bins)

        # ==========================================
        # BASE TENSORS & EXACT CDF PRECOMPUTATION
        # ==========================================
        active_norm = normalized_weights[active_indices]
        active_scale = row_max[active_indices]
        batch_len = len(active_indices)

        oracle_targets = active_norm * active_scale.unsqueeze(1)
        row_energy = torch.sum(oracle_targets, dim=1).clamp(min=1e-8)

        # 1. Sort the weights once
        norm_sorted, _ = torch.sort(active_norm, dim=1) 
        
        # 2. Build the Cumulative Sum (Prefix Sum) for the algebraic shortcut
        W_cdf = torch.zeros((batch_len, cols + 1), device=device, dtype=torch.float32)
        W_cdf[:, 1:] = torch.cumsum(norm_sorted.to(torch.float32), dim=1)
        
        # Math constants for the evaluator
        row_offsets = torch.arange(batch_len, device=device).view(-1, 1, 1) * (cols + 1)
        scale_f32 = active_scale.to(torch.float32).unsqueeze(1) 
        energy_f32 = row_energy.to(torch.float32).unsqueeze(1)  
        
        divisor = float(K_bins - 1)
        base_bins = torch.arange(K_bins, device=device).float() / divisor

        # ==========================================
        # INLINE GRID EVALUATOR (THE CDF ALGEBRAIC SHORTCUT)
        # ==========================================
        def evaluate_grid(guess_w_a, guess_w_c, guess_w_m, chunk_size=128):
            G = guess_w_a.shape[1]
            best_w_a = torch.empty(batch_len, device=device)
            best_w_c = torch.empty(batch_len, device=device)
            best_w_m = torch.empty(batch_len, device=device)
            best_error = torch.full((batch_len,), float('inf'), device=device)
            
            b = base_bins.view(1, 1, K_bins) 
            
            for i in range(0, G, chunk_size):
                gw_a_chunk = guess_w_a[:, i:i+chunk_size].unsqueeze(2) 
                gw_c_chunk = guess_w_c[:, i:i+chunk_size].unsqueeze(2) 
                gw_m_chunk = guess_w_m[:, i:i+chunk_size].unsqueeze(2) 
                
                current_chunk_len = gw_a_chunk.shape[1]
                
                a_g = torch.tanh(gw_a_chunk) 
                c_g = C_MAX * torch.sigmoid(gw_c_chunk) 
                wm_g = gw_m_chunk
                
                # Linkage & Topology
                safe_denom = torch.abs(a_g) + 0.001 
                link_factor = 1.0 + (1.0 / safe_denom)
                m_log = 0.5 * torch.log(wm_g**2 + 1.0)
                m_g = torch.clamp(((M_MAX /  link_factor) + 1.0) * m_log * link_factor, 0, M_MAX)
                
                inner = (1.0 - a_g) * b + a_g * (b ** m_g)
                warped = torch.clamp(inner, 1e-6, 1.0) ** c_g
                warped_sorted, _ = torch.sort(warped, dim=2) 
                
                # ==========================================
                # THE 0-MEMORY MAGIC: Threshold Search
                # ==========================================
                # Find the halfway points between bins
                thresholds = (warped_sorted[:, :, :-1] + warped_sorted[:, :, 1:]) / 2.0 
                
                # Combine bins and thresholds to search them all at once
                queries = torch.cat([warped_sorted, thresholds], dim=2) 
                queries_flat = queries.view(batch_len, -1) 
                
                # Search the 2D original weights using the flat queries natively!
                idx_flat = torch.searchsorted(norm_sorted, queries_flat) 
                idx = idx_flat.view(batch_len, current_chunk_len, 2 * K_bins - 1)
                
                # Split the results back out
                idx_mid = idx[:, :, :K_bins]
                idx_thresh = idx[:, :, K_bins:]
                
                # Define the absolute start and end of the row
                zeros = torch.zeros((batch_len, current_chunk_len, 1), dtype=torch.long, device=device)
                cols_tensor = torch.full((batch_len, current_chunk_len, 1), cols, dtype=torch.long, device=device)
                
                idx_start = torch.cat([zeros, idx_thresh], dim=2)
                idx_end = torch.cat([idx_thresh, cols_tensor], dim=2)
                
                # ==========================================
                # EXACT ERROR CALCULATION (No 4096-column loops!)
                # ==========================================
                flat_idx_start = (idx_start + row_offsets).view(-1)
                flat_idx_mid = (idx_mid + row_offsets).view(-1)
                flat_idx_end = (idx_end + row_offsets).view(-1)
                
                W_flat = W_cdf.view(-1)
                W_start = W_flat[flat_idx_start].view(batch_len, current_chunk_len, K_bins)
                W_mid = W_flat[flat_idx_mid].view(batch_len, current_chunk_len, K_bins)
                W_end = W_flat[flat_idx_end].view(batch_len, current_chunk_len, K_bins)
                
                # The exact algebraic formula for Absolute Delta Error!
                E_bins = warped_sorted * (2 * idx_mid - idx_start - idx_end) - 2 * W_mid + W_start + W_end
                
                norm_error_chunk = E_bins.sum(dim=2)
                
                # Convert back to physical energy error instantly
                energy_error = (norm_error_chunk * scale_f32) / energy_f32
                
                chunk_best_error, chunk_best_idx = torch.min(energy_error, dim=1)
                update_mask = chunk_best_error < best_error
                
                if update_mask.any():
                    best_error[update_mask] = chunk_best_error[update_mask]
                    row_indexer = torch.arange(batch_len, device=device)[update_mask]
                    idx_in_chunk = chunk_best_idx[update_mask]
                    
                    best_w_a[update_mask] = guess_w_a[row_indexer, i + idx_in_chunk]
                    best_w_c[update_mask] = guess_w_c[row_indexer, i + idx_in_chunk]
                    best_w_m[update_mask] = guess_w_m[row_indexer, i + idx_in_chunk]

                del gw_a_chunk, gw_c_chunk, gw_m_chunk, a_g, c_g, wm_g
                del inner, warped, warped_sorted, thresholds, queries, queries_flat
                del idx_flat, idx, idx_mid, idx_thresh, zeros, cols_tensor, idx_start, idx_end
                del flat_idx_start, flat_idx_mid, flat_idx_end, W_start, W_mid, W_end
                del E_bins, norm_error_chunk, energy_error

            return best_w_a, best_w_c, best_w_m, best_error

        # ==========================================
        # UNIFORM BOUNDARIES & RANGES
        # ==========================================
        W_A_MIN, W_A_MAX = -3.0, 3.0
        W_C_MIN, W_C_MAX = -3.0, 3.0
        W_M_MIN, W_M_MAX = 0.0, 1.0
        
        W_A_RANGE = W_A_MAX - W_A_MIN
        W_C_RANGE = W_C_MAX - W_C_MIN
        W_M_RANGE = W_M_MAX - W_M_MIN

        safe_chunk = 32 if (rows * cols) <= 20_000_000 else 16

        # ==========================================
        # STAGE 0: GLOBAL COMBINATORIAL SURVEY
        # ==========================================

        guess_w_a = torch.empty(batch_len, N_stage0, device=device).uniform_(W_A_MIN, W_A_MAX)
        guess_w_c = torch.empty(batch_len, N_stage0, device=device).uniform_(W_C_MIN, W_C_MAX)
        guess_w_m = torch.empty(batch_len, N_stage0, device=device).uniform_(W_M_MIN, W_M_MAX)
        best_w_a, best_w_c, best_w_m, best_error = evaluate_grid(guess_w_a, guess_w_c, guess_w_m, safe_chunk)

        # ==========================================
        # STAGE 1: UNIFORM NEIGHBORHOOD SEARCH
        # ==========================================

        jit_a = torch.empty(batch_len, N_stage1, device=device).uniform_(-REFINEMENT_AREA_1, REFINEMENT_AREA_1) * W_A_RANGE
        guess_w_a = torch.clamp(best_w_a.unsqueeze(1) + jit_a, W_A_MIN, W_A_MAX)
        
        jit_c = torch.empty(batch_len, N_stage1, device=device).uniform_(-REFINEMENT_AREA_1, REFINEMENT_AREA_1) * W_C_RANGE
        guess_w_c = torch.clamp(best_w_c.unsqueeze(1) + jit_c, W_C_MIN, W_C_MAX)
        
        jit_m = torch.empty(batch_len, N_stage1, device=device).uniform_(-REFINEMENT_AREA_1, REFINEMENT_AREA_1) * W_M_RANGE
        guess_w_m = torch.clamp(best_w_m.unsqueeze(1) + jit_m, W_M_MIN, W_M_MAX)
        
        best_w_a, best_w_c, best_w_m, best_error = evaluate_grid(guess_w_a, guess_w_c, guess_w_m, safe_chunk)

        # ==========================================
        # STAGE 2: UNIFORM MICRO-POLISH
        # ==========================================

        jit_a = torch.empty(batch_len, N_stage2, device=device).uniform_(-REFINEMENT_AREA_2, REFINEMENT_AREA_2) * W_A_RANGE
        guess_w_a = torch.clamp(best_w_a.unsqueeze(1) + jit_a, W_A_MIN, W_A_MAX)
        
        jit_c = torch.empty(batch_len, N_stage2, device=device).uniform_(-REFINEMENT_AREA_2, REFINEMENT_AREA_2) * W_C_RANGE
        guess_w_c = torch.clamp(best_w_c.unsqueeze(1) + jit_c, W_C_MIN, W_C_MAX)
        
        jit_m = torch.empty(batch_len, N_stage2, device=device).uniform_(-REFINEMENT_AREA_2, REFINEMENT_AREA_2) * W_M_RANGE
        guess_w_m = torch.clamp(best_w_m.unsqueeze(1) + jit_m, W_M_MIN, W_M_MAX)
        
        best_w_a, best_w_c, best_w_m, best_error = evaluate_grid(guess_w_a, guess_w_c, guess_w_m, safe_chunk)

        # ==========================================
        # PARETO THRESHOLD & PHYSICAL ASSIGNMENT
        # ==========================================
        threshold = max_energy_error * leniency_map.get(current_D, 1.0)
        pass_mask = best_error <= threshold

        if current_D == 5 and "gate" in name:
            print(f"\n[X-RAY] {name} | 5-Bit Mean L1 Error: {best_error.mean().item():.4f} | Target: {threshold:.4f}")

        if pass_mask.any():
            passed_global_indices = active_indices[pass_mask]
            final_bitrates[passed_global_indices] = current_D
            
            # Unpack the winning w-parameters into final physical parameters
            win_w_a = best_w_a[pass_mask]
            win_w_c = best_w_c[pass_mask]
            win_w_m = best_w_m[pass_mask]
            
            a_phys = torch.tanh(win_w_a)
            c_phys = C_MAX * torch.sigmoid(win_w_c)
            
            safe_denom = torch.abs(a_phys) + 0.001 
            link_factor = 1.0 + (1.0 / safe_denom)
            m_log = 0.5 * torch.log(win_w_m**2 + 1.0)
            m_raw = ((M_MAX /  link_factor) + 1.0) * m_log * link_factor
            m_phys = torch.clamp(m_raw, 0, M_MAX)
            
            final_a[passed_global_indices] = a_phys
            final_c[passed_global_indices] = c_phys
            final_m[passed_global_indices] = m_phys
        
        active_indices = active_indices[~pass_mask]

    # ==========================================
    # THE SUPERBLOCK PACKER & DYNAMIC ASSIGNMENT
    # ==========================================
    # We use int32 so PyTorch doesn't do weird sign-extensions during bitwise shifts
    quantized_indices = torch.zeros((rows, cols), dtype=torch.int32, device=device)

    for current_D in range(2, 9):
        mask = final_bitrates == current_D
        if not mask.any(): continue

        K_bins = 2 ** (current_D - 1)

        divisor = float(K_bins - 1)
        base_bins = torch.arange(K_bins, device=device).float() / divisor

        r_a = final_a[mask].unsqueeze(1)
        r_c = final_c[mask].unsqueeze(1)
        r_m = final_m[mask].unsqueeze(1)
        r_norm = normalized_weights[mask]

        inner = (1.0 - r_a) * base_bins.unsqueeze(0) + r_a * (base_bins.unsqueeze(0) ** r_m)
        safe_inner_pack = torch.clamp(inner, 1e-6, 1.0)
        warped_bins = torch.clamp(safe_inner_pack ** r_c, 0.0, 1.0)

        best_indices = torch.zeros_like(r_norm, dtype=torch.long)
        chunk_size_pack = 256
        
        for i in range(0, r_norm.shape[0], chunk_size_pack):
            r_norm_chunk = r_norm[i : i + chunk_size_pack]
            warped_chunk = warped_bins[i : i + chunk_size_pack]
            diffs = torch.abs(r_norm_chunk.unsqueeze(2) - warped_chunk.unsqueeze(1))
            best_indices[i : i + chunk_size_pack] = torch.argmin(diffs, dim=2)

        is_negative = weight_tensor[mask] < 0
        
        # The Sign Bit is dynamically fused into the MSB right here!
        final_idx = best_indices + (is_negative.to(torch.long) * K_bins)
        quantized_indices[mask] = final_idx.to(torch.int32)

    # ==========================================
    # FINAL BIT PACKING (SWAR)
    # ==========================================
    """
    Packs a quantized matrix into the V36 4-2-1-1 Superblock format.
    """
    rows, in_features = quantized_indices.shape
    
    packed_bytes = []
    offsets = []
    current_offset = 0
    
    for i in range(rows):
        D = int(final_bitrates[i].item())
        mag_bits = D - 1
        num_superblocks = in_features // 32
        
        # Extract row magnitudes and signs, reshaping into superblocks of 32
        row_q = quantized_indices[i]
        mag = row_q.abs().to(torch.int32).view(num_superblocks, 32)
        sgn = (row_q < 0).to(torch.int32).view(num_superblocks, 32)
        
        superblocks_data = []
        
        # 1. Pack 4-Bit Base (16 bytes per superblock)
        if mag_bits >= 4:
            m4 = mag & 0xF
            shifts = torch.arange(8, device=device) * 4
            words_4bit = (m4.view(num_superblocks, 4, 8) << shifts).sum(dim=-1, dtype=torch.int32)
            superblocks_data.append(words_4bit)
            mag >>= 4
            mag_bits -= 4
            
        # 2. Pack 2-Bit Residual (8 bytes per superblock)
        if mag_bits >= 2:
            m2 = mag & 0x3
            shifts = torch.arange(16, device=device) * 2
            words_2bit = (m2.view(num_superblocks, 2, 16) << shifts).sum(dim=-1, dtype=torch.int32)
            superblocks_data.append(words_2bit)
            mag >>= 2
            mag_bits -= 2
            
        # 3. Pack 1-Bit Residual (4 bytes per superblock)
        if mag_bits == 1:
            m1 = mag & 0x1
            shifts = torch.arange(32, device=device)
            words_1bit = (m1.view(num_superblocks, 1, 32) << shifts).sum(dim=-1, dtype=torch.int32)
            superblocks_data.append(words_1bit)
            
        # 4. Pack Signs (4 bytes per superblock)
        shifts = torch.arange(32, device=device)
        words_sign = (sgn.view(num_superblocks, 1, 32) << shifts).sum(dim=-1, dtype=torch.int32)
        superblocks_data.append(words_sign)
        
        # Interleave and flatten the superblock words directly into a raw bytearray
        row_data = torch.cat(superblocks_data, dim=1).flatten().view(torch.uint8)
        
        packed_bytes.append(row_data)
        offsets.append(current_offset)
        current_offset += len(row_data)

    vbr_data = torch.cat(packed_bytes)
    vbr_offsets = torch.tensor(offsets, dtype=torch.int32)

    return {
        "is_vbr_compressed": True,
        "original_shape": quantized_indices.shape,
        "vbr_data": vbr_data.cpu(),
        "vbr_offsets": vbr_offsets.cpu(),
        "bitrates": final_bitrates.cpu().to(torch.uint8),
        "param_a": final_a.half().cpu(),
        "param_c": final_c.half().cpu(),
        "param_m": final_m.half().cpu(),
        "row_max": row_max.half().cpu()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_idx", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--total_chunks", type=int, default=4)
    args = parser.parse_args()

    DEVICE = f"cuda:{args.gpu}"
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cpu")
    layer_names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "lm_head" not in n]
    
    chunk_size = math.ceil(len(layer_names) / args.total_chunks)
    start_idx = args.chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, len(layer_names))
    chunk_names = layer_names[start_idx:end_idx]
    
    if not chunk_names: return

    chunk_dict = {}
    pbar = tqdm(chunk_names, desc=f"GPU {args.gpu} Starting...", position=0, ncols=100)
    
    for name in pbar:
        short_name = name.split(".")[-1]
        pbar.set_description(f"GPU {args.gpu} | {short_name[:12]:<12}")
        
        weight = model.get_submodule(name).weight.to(DEVICE)
        
        # 1. Dynamic Error Routing
        if any(attn_type in name for attn_type in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            layer_error = ATTENTION_ERROR
            leniency_mask = LENIENCY_MASK_ATTN
        elif any(ffn_type in name for ffn_type in ["gate_proj", "up_proj", "down_proj"]):
            layer_error = EXPERT_ERROR
            leniency_mask = LENIENCY_MASK_EXPERT
        else:
            layer_error = DEFAULT_ERROR  # Fallback for any other linear layers
            leniency_mask = LENIENCY_MASK_DEFAULT
        
        # 2. Compress
        vbr_result = compress_vbr_v35_matrix(name, weight, layer_error, leniency_mask) 
        
        if vbr_result is not None:
            if hasattr(model.get_submodule(name), "bias") and model.get_submodule(name).bias is not None:
                vbr_result["bias"] = model.get_submodule(name).bias.half().cpu()
            chunk_dict[name] = vbr_result
        else:
            chunk_dict[name] = {
                "raw_data": weight.half().cpu(), 
                "is_vbr_compressed": False,
                "original_shape": weight.shape
            }
            
        del weight
        torch.cuda.empty_cache()
        gc.collect()
        
    out_file = os.path.join(OUTPUT_DIR, f"compressed_{str(args.chunk_idx + 1).zfill(2)}.pt")
    torch.save({"experts_cold": chunk_dict}, out_file)

if __name__ == "__main__":
    main()
