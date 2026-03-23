import torch
import os
import gc
import math
import argparse
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM

print("==========================================")
print(" VIRTUALBRAIN V30: THE NEURAL AUTOENCODER ")
print("==========================================")

# --- CONFIGURATION ---
MODEL_ID = os.path.expanduser("~/models/quant/qwen25_7b")
OUTPUT_DIR = os.path.expanduser("~/models/quant/vbr_qwen25_v30")

MAX_ENERGY_ERROR_EXPERT    = 0.01      
MAX_ENERGY_ERROR_ATTENTION = 0.0005    

LENIENCY_MAP_ATTENTION = {2: 2.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0}
LENIENCY_MAP_EXPERT = {2: 4.0, 3: 2.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0}

os.makedirs(OUTPUT_DIR, exist_ok=True)
  
# ==========================================
# THE V30 NEURAL COMPRESSOR
# ==========================================

def compress_vbr_v30_matrix(name, weight):
    device = weight.device
    out_features, in_features = weight.shape
    weight_flat = weight.detach().view(out_features, -1).to(device)

    # 1. Base Energy and Signs
    signs = weight_flat < 0
    M = torch.abs(weight_flat)
    raw_max = M.max(dim=1)[0].clamp(min=1e-5)
    row_energy = (M ** 2).sum(dim=1)

    # 2. Master Tracking Tensors
    best_D = torch.zeros(out_features, dtype=torch.uint8, device=device)
    best_ints = torch.zeros_like(M, dtype=torch.int32)
    alpha_a = torch.zeros(out_features, dtype=torch.float16, device=device)
    alpha_b = torch.zeros(out_features, dtype=torch.float16, device=device)
    power_m = torch.ones(out_features, dtype=torch.float16, device=device)
    power_n = torch.ones(out_features, dtype=torch.float16, device=device)
    dust_anchors = torch.zeros(out_features, dtype=torch.float16, device=device)
    vbr_scales = torch.zeros(out_features, dtype=torch.float16, device=device)

    active_indices = torch.arange(out_features, device=device)
    M_untested = M.clone()
    raw_max_chunk = raw_max.clone()
    chunk_energy = row_energy.clone()

    is_attention = any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj"])
    base_threshold = MAX_ENERGY_ERROR_ATTENTION if is_attention else MAX_ENERGY_ERROR_EXPERT
    threshold_chunk = torch.full((out_features,), base_threshold, device=device)
    
    leniency_map = LENIENCY_MAP_ATTENTION if is_attention else LENIENCY_MAP_EXPERT

    # ==========================================
    # THE VARIABLE BITRATE COMPRESSION LOOP
    # ==========================================
    for current_D in range(2, 9):
        if len(active_indices) == 0:
            break

        mag_bits = current_D - 1
        K_bins = 2 ** mag_bits
        divisor = float(K_bins - 1)
        norm_indices = torch.arange(K_bins, device=device).float() / divisor

        current_threshold = threshold_chunk * leniency_map[current_D]
        if current_D == 8:
            current_threshold = torch.full_like(current_threshold, float('inf'))

        # Chunk Sorting
        chunk_size_sort = 4096
        M_sorted_list = []
        for i in range(0, M_untested.shape[0], chunk_size_sort):
            vals, _ = torch.sort(M_untested[i:i+chunk_size_sort], dim=1)
            M_sorted_list.append(vals)
        M_sorted = torch.cat(M_sorted_list, dim=0)

        # Oracle Targets via Percentile CDF
        quantile_indices = torch.linspace(0, M_untested.shape[1] - 1, steps=K_bins, device=device).long()
        oracle_targets = M_sorted[:, quantile_indices]

        # Analytical Dust Engine
        if mag_bits > 3:
            active_dust = torch.zeros_like(raw_max_chunk)
        else:
            dynamic_dust_ratio = 1.0 / K_bins
            dust_budget = chunk_energy * (current_threshold * dynamic_dust_ratio)
            cum_energy = torch.cumsum(M_sorted.pow(2), dim=1) 
            counts = torch.searchsorted(cum_energy, dust_budget.unsqueeze(1)).squeeze(1)
            anchor_indices = (counts - 1).clamp(min=0)
            active_dust = torch.gather(M_sorted, 1, anchor_indices.unsqueeze(1)).squeeze(1)
            
        active_scale = raw_max_chunk - active_dust
        safe_norm = torch.clamp(norm_indices, min=1e-4).unsqueeze(0)
        #safe_norm = norm_indices.unsqueeze(0) + 1e-5
        batch_len = M_untested.shape[0]

        # ==========================================
        # V34: FULL PIPELINE ALTERNATING GRID SEARCH
        # ==========================================
        chunk_size_mc = 4096

        with torch.no_grad():
            # --- HELPER: Memory-Safe Grid Evaluator ---
            def evaluate_grid(a_guesses, b_guesses, m_guesses, n_guesses):
                """ Takes [batch_len, num_guesses] and returns the best parameters per row """
                num_g = a_guesses.shape[1]
                b_a, b_b, b_m, b_n = [], [], [], []
                
                a_c = a_guesses.unsqueeze(2)
                b_c = b_guesses.unsqueeze(2)
                m_c = m_guesses.unsqueeze(2)
                n_c = n_guesses.unsqueeze(2)
                norm_c = safe_norm.unsqueeze(0) # [1, 128]
                
                for i in range(0, batch_len, chunk_size_mc):
                    a_chunk = a_c[i:i+chunk_size_mc]
                    b_chunk = b_c[i:i+chunk_size_mc]
                    m_chunk = m_c[i:i+chunk_size_mc]
                    n_chunk = n_c[i:i+chunk_size_mc]
                    
                    linear = (1.0 - a_chunk - b_chunk) * norm_c
                    curve = linear + a_chunk * (norm_c ** m_chunk) + b_chunk * (norm_c ** n_chunk)
                    
                    # 🚨 The Clamp: If it dips, it flattens. No explosions.
                    curve = torch.clamp(curve, 0.0, 1.0)
                    
                    scale_chunk = active_scale[i:i+chunk_size_mc].view(-1, 1, 1)
                    dust_chunk = active_dust[i:i+chunk_size_mc].view(-1, 1, 1)
                    oracle_chunk = oracle_targets[i:i+chunk_size_mc].unsqueeze(1).expand(-1, num_g, -1)
                    
                    bin_chunk = (curve * scale_chunk) + dust_chunk
                    
                    # FULL SPECTRUM MSE
                    mse = F.mse_loss(bin_chunk.to(oracle_chunk.dtype), oracle_chunk, reduction='none').mean(dim=2)
                    
                    best_idx = torch.argmin(mse, dim=1)
                    row_idx = torch.arange(a_chunk.shape[0], device=device)
                    
                    b_a.append(a_chunk[row_idx, best_idx, 0])
                    b_b.append(b_chunk[row_idx, best_idx, 0])
                    b_m.append(m_chunk[row_idx, best_idx, 0])
                    b_n.append(n_chunk[row_idx, best_idx, 0])
                    
                return torch.cat(b_a), torch.cat(b_b), torch.cat(b_m), torch.cat(b_n)

            # --- STAGE 1: FULL SPECTRUM (b, n) ---
            N = 1024
            guess_b = torch.empty(1, N, device=device).uniform_(0.0, 1.0).expand(batch_len, -1)
            guess_n = torch.empty(1, N, device=device).uniform_(2.0, 25.0).expand(batch_len, -1)
            guess_a = torch.zeros(1, N, device=device).expand(batch_len, -1) # a=0
            guess_m = torch.ones(1, N, device=device).expand(batch_len, -1)  # m=1
            
            best_a, best_b, best_m, best_n = evaluate_grid(guess_a, guess_b, guess_m, guess_n)

            # --- STAGE 2: FULL SPECTRUM (k, m) ---
            N = 1024
            guess_k = torch.empty(1, N, device=device).uniform_(-2.0, 2.0).expand(batch_len, -1)
            
            m_low = torch.empty(1, N // 2, device=device).uniform_(0.15, 0.95)
            m_high = torch.empty(1, N - (N // 2), device=device).uniform_(1.05, 8.0)
            guess_m = torch.cat([m_low, m_high], dim=1).expand(batch_len, -1)
            
            guess_a = guess_k / (1.0 - guess_m)
            guess_b = best_b.unsqueeze(1).expand(-1, N) 
            guess_n = best_n.unsqueeze(1).expand(-1, N) 
            
            best_a, best_b, best_m, best_n = evaluate_grid(guess_a, guess_b, guess_m, guess_n)

            # --- STAGE 3: RE-EVALUATE (b, n) ---
            guess_b = torch.empty(1, N, device=device).uniform_(0.0, 1.0).expand(batch_len, -1)
            guess_n = torch.empty(1, N, device=device).uniform_(2.0, 25.0).expand(batch_len, -1)
            guess_a = best_a.unsqueeze(1).expand(-1, N) # Locked from Stage 2
            guess_m = best_m.unsqueeze(1).expand(-1, N) # Locked from Stage 2
            
            best_a, best_b, best_m, best_n = evaluate_grid(guess_a, guess_b, guess_m, guess_n)

            # --- STAGE 4: RE-EVALUATE (k, m) ---
            guess_k = torch.empty(1, N, device=device).uniform_(-2.0, 2.0).expand(batch_len, -1)
            
            m_low = torch.empty(1, N // 2, device=device).uniform_(0.15, 0.95)
            m_high = torch.empty(1, N - (N // 2), device=device).uniform_(1.05, 8.0)
            guess_m = torch.cat([m_low, m_high], dim=1).expand(batch_len, -1)
            
            guess_a = guess_k / (1.0 - guess_m)
            guess_b = best_b.unsqueeze(1).expand(-1, N) 
            guess_n = best_n.unsqueeze(1).expand(-1, N) 
            
            best_a, best_b, best_m, best_n = evaluate_grid(guess_a, guess_b, guess_m, guess_n)

            # --- STAGE 5: MICRO-ADJUSTMENTS (Joint Jitter) ---
            N_micro = 512
            
            # 1. Generate the Jitters using N_micro (512)
            jit_k = torch.empty(1, N_micro, device=device).uniform_(-0.2, 0.2)
            jit_b = torch.empty(1, N_micro, device=device).uniform_(-0.05, 0.05)
            jit_n = torch.empty(1, N_micro, device=device).uniform_(-0.5, 0.5)

            # 2. Dynamic m jitter using N_micro (512)
            base_jit_m = torch.empty(1, N_micro, device=device).uniform_(-1.0, 1.0)
            m_scale = torch.where(best_m > 1.0, 0.5, 0.05).unsqueeze(1)
            jit_m = base_jit_m * m_scale

            # 3. Apply jitters and clamp
            guess_m = torch.clamp(best_m.unsqueeze(1) + jit_m, 0.15, 8.0)
            
            # SAFEGUARD: Prevent Division by Zero if m lands exactly on 1.0
            guess_m = torch.where(guess_m == 1.0, 1.001, guess_m)

            # 4. Reconstruct current best_k to apply the jitter
            current_best_k = best_a * (1.0 - best_m)
            guess_k = current_best_k.unsqueeze(1) + jit_k
            
            # 5. Final parameter calculations
            guess_a = guess_k / (1.0 - guess_m)
            guess_b = torch.clamp(best_b.unsqueeze(1) + jit_b, 0.0, 1.0)
            guess_n = torch.clamp(best_n.unsqueeze(1) + jit_n, 1.0, 25.0)

            best_a, best_b, best_m, best_n = evaluate_grid(guess_a, guess_b, guess_m, guess_n)

        # --- END OF OPTIMIZATION ---
        # The Evaluation Stage uses best_a, best_b, best_m, best_n from here!

        # --- 3. THE EVALUATION STAGE ---
        with torch.no_grad():
            # Use the HARVESTED WINNERS from the Top-16 Beam Search!
            a_f = best_a.detach().clone().flatten()
            b_f = torch.clamp(best_b.detach(), 0.0, 1.0).clone().flatten()
            m_f = torch.clamp(best_m.detach(), 0.1, 4.0).clone().flatten()
            n_f = torch.clamp(best_n.detach(), 1.0, 25.0).clone().flatten() # Matched to 25.0 bound

            a_c = a_f.unsqueeze(1)
            b_c = b_f.unsqueeze(1)
            m_c = m_f.unsqueeze(1)
            n_c = n_f.unsqueeze(1)

            linear = (1.0 - a_c - b_c) * safe_norm
            curve = linear + a_c * (safe_norm ** m_c) + b_c * (safe_norm ** n_c)
            curve = torch.clamp(curve, 0.0, 1.0)

            lut = (curve * active_scale.unsqueeze(1)) + active_dust.unsqueeze(1)
            
            closest_ints_global = torch.zeros_like(M_untested, dtype=torch.int16) 
            error_energy = torch.zeros(M_untested.shape[0], device=device, dtype=torch.float32)
            
            chunk_size_eval = 4096
            for i in range(0, M_untested.shape[0], chunk_size_eval):
                M_chunk = M_untested[i:i+chunk_size_eval]
                lut_chunk = lut[i:i+chunk_size_eval]
                
                min_dist_chunk = torch.full_like(M_chunk, float('inf'))
                closest_ints_chunk = torch.zeros_like(M_chunk, dtype=torch.int64) 
                
                for k in range(lut_chunk.shape[1]):
                    bin_dist_chunk = torch.abs(M_chunk - lut_chunk[:, k:k+1])
                    is_closer_chunk = bin_dist_chunk < min_dist_chunk
                    closest_ints_chunk.masked_fill_(is_closer_chunk, k)
                    torch.minimum(min_dist_chunk, bin_dist_chunk, out=min_dist_chunk)
                    
                closest_ints_global[i:i+chunk_size_eval] = closest_ints_chunk.to(torch.int16)
                
                M_recon_chunk = torch.gather(lut_chunk, 1, closest_ints_chunk)
                error_energy[i:i+chunk_size_eval] = (M_chunk - M_recon_chunk).pow_(2).sum(dim=1)
                
                del M_chunk, lut_chunk, min_dist_chunk, closest_ints_chunk, M_recon_chunk

            closest_ints = closest_ints_global

        # --- 4. THE SIEVE UPDATE ---
        passed_curves = error_energy < (chunk_energy * current_threshold)
        if passed_curves.any():
            global_passed_indices = active_indices[passed_curves]
            
            best_D[global_passed_indices] = current_D
            best_ints[global_passed_indices] = closest_ints[passed_curves].to(torch.int32)
            
            alpha_a[global_passed_indices] = a_f[passed_curves].half()
            alpha_b[global_passed_indices] = b_f[passed_curves].half()
            power_m[global_passed_indices] = m_f[passed_curves].half()
            power_n[global_passed_indices] = n_f[passed_curves].half()
            dust_anchors[global_passed_indices] = active_dust[passed_curves].half()
            vbr_scales[global_passed_indices] = active_scale[passed_curves].half()
            
            unsolved_mask = ~passed_curves
            active_indices = active_indices[unsolved_mask]
            M_untested = M_untested[unsolved_mask]
            threshold_chunk = threshold_chunk[unsolved_mask]
            chunk_energy = chunk_energy[unsolved_mask]
            raw_max_chunk = raw_max_chunk[unsolved_mask]

    # ==========================================
    # FINAL BIT PACKING (COMPOUND SUPERBLOCKS)
    # ==========================================
    packed_bytes = []
    offsets = []
    current_offset = 0
    
    # Assert sanity check for the 32-weight Superblock
    assert in_features % 32 == 0, "in_features must be cleanly divisible by 32 for Compound Superblocks."
    
    for i in range(out_features):
        final_d = int(best_D[i].item())
        if final_d == 0: 
            final_d = 8  
            best_D[i] = 8
            
        mag_bits = final_d - 1
        
        # Squeeze the entire row into Superblocks of 32 weights
        r_ints = best_ints[i].to(torch.uint8).view(-1, 32)
        r_signs = signs[i].to(torch.uint8).view(-1, 32)
        
        blocks = []
        current_shift = 0
        
        # 1. Extract 4-bit Base Chunk (if magnitude has at least 4 bits)
        if mag_bits >= 4:
            b4 = (r_ints >> current_shift) & 0x0F
            pack4 = b4[:, 0::2] | (b4[:, 1::2] << 4)
            blocks.append(pack4)
            current_shift += 4
            mag_bits -= 4
            
        # 2. Extract 2-bit Residual Chunk (if magnitude has at least 2 bits left)
        if mag_bits >= 2:
            b2 = (r_ints >> current_shift) & 0x03
            pack2 = b2[:, 0::4] | (b2[:, 1::4] << 2) | (b2[:, 2::4] << 4) | (b2[:, 3::4] << 6)
            blocks.append(pack2)
            current_shift += 2
            mag_bits -= 2
            
        # 3. Extract 1-bit Residual Chunk (if magnitude has exactly 1 bit left)
        if mag_bits == 1:
            b1 = (r_ints >> current_shift) & 0x01
            pack1 = (b1[:, 0::8] | (b1[:, 1::8] << 1) | (b1[:, 2::8] << 2) | (b1[:, 3::8] << 3) |
                     (b1[:, 4::8] << 4) | (b1[:, 5::8] << 5) | (b1[:, 6::8] << 6) | (b1[:, 7::8] << 7))
            blocks.append(pack1)
            
        # 4. Extract 1-bit Sign Chunk (ALWAYS PRESENT)
        pack_sign = (r_signs[:, 0::8] | (r_signs[:, 1::8] << 1) | (r_signs[:, 2::8] << 2) | (r_signs[:, 3::8] << 3) |
                     (r_signs[:, 4::8] << 4) | (r_signs[:, 5::8] << 5) | (r_signs[:, 6::8] << 6) | (r_signs[:, 7::8] << 7))
        blocks.append(pack_sign)
        
        # 5. Compound Fusion (Creates the perfect Cache-Local Blocks)
        packed_row = torch.cat(blocks, dim=1).flatten()
        
        packed_bytes.append(packed_row)
        offsets.append(current_offset)
        current_offset += len(packed_row)

    vbr_data = torch.cat(packed_bytes)
    vbr_offsets = torch.tensor(offsets, dtype=torch.int32)
    row_divisors = (2 ** (best_D.float() - 1)) - 1.0
    
    return {
        "is_vbr_compressed": True,
        "original_shape": weight.shape,
        "vbr_data": vbr_data.cpu(),
        "vbr_offsets": vbr_offsets.cpu(),
        "vbr_headers": best_D.cpu().to(torch.uint8),
        "row_divisors": row_divisors.cpu().to(torch.float16), 
        "vbr_scales": vbr_scales.cpu(),
        "alpha_a": alpha_a.cpu(),
        "alpha_b": alpha_b.cpu(),
        "power_m": power_m.cpu(),
        "power_n": power_n.cpu(),
        "dust_anchors": dust_anchors.cpu()
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
    #layer_names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    
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
        vbr_result = compress_vbr_v30_matrix(name, weight) 
        
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
