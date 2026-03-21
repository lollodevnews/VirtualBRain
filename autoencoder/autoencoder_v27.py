import torch
import os
import gc
import math
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM

print("==========================================")
print(" VIRTUALBRAIN V27: THE NEURAL AUTOENCODER ")
print("==========================================")

# --- CONFIGURATION ---
MODEL_ID = os.path.expanduser("~/models/quant/qwen25_7b")
OUTPUT_DIR = os.path.expanduser("~/models/quant/vbr_qwen25_v27")

MAX_ENERGY_ERROR_EXPERT    = 0.01      
MAX_ENERGY_ERROR_ATTENTION = 0.0005    

LENIENCY_MAP_ATTENTION = {2: 4.0, 3: 2.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0}
LENIENCY_MAP_EXPERT = {2: 5.0, 3: 2.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0}

os.makedirs(OUTPUT_DIR, exist_ok=True)
  
# ==========================================
# THE V25 NEURAL COMPRESSOR
# ==========================================

def compress_vbr_v25_matrix(name, weight):
    device = weight.device  # Safely inherit the specific GPU!
    out_features, in_features = weight.shape
    # .detach() prevents PyTorch from building a 14GB autograd graph back to the HF model
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

    # Dynamic Thresholds & V28 Leniency Multipliers
    is_attention = any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj"])
    base_threshold = MAX_ENERGY_ERROR_ATTENTION if is_attention else MAX_ENERGY_ERROR_EXPERT
    threshold_chunk = torch.full((out_features,), base_threshold, device=device)
    
    # THE V28 UPGRADE: Pareto-Optimal SNR Multipliers (D: Multiplier)
    if is_attention:
        leniency_map = LENIENCY_MAP_ATTENTION
    else:
        # MLPs can tolerate 5x more error at Q2, 2x more at Q3
        leniency_map = LENIENCY_MAP_EXPERT

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

        # Apply the current depth's leniency multiplier
        current_threshold = threshold_chunk * leniency_map[current_D]

        # If we hit 8-bit (max size), force everything left to pass the Sieve!
        if current_D == 8:
            current_threshold = torch.full_like(current_threshold, float('inf'))

        # THE FIX: 4096 Chunk Size for Sorting
        chunk_size_sort = 4096
        M_sorted_list = []
        for i in range(0, M_untested.shape[0], chunk_size_sort):
            vals, _ = torch.sort(M_untested[i:i+chunk_size_sort], dim=1)
            M_sorted_list.append(vals)
        M_sorted = torch.cat(M_sorted_list, dim=0)

        # Generate Oracle Targets (Shape: [batch_len, K_bins]) via Percentile CDF
        quantile_indices = torch.linspace(0, M_untested.shape[1] - 1, steps=K_bins, device=device).long()
        oracle_targets = M_sorted[:, quantile_indices]

        # Analytical Dust Engine
        if mag_bits > 3:
            active_dust = torch.zeros_like(raw_max_chunk)
        else:
            dust_budget = chunk_energy * (current_threshold * 0.20)
            # Use in-place square to save another 1GB of VRAM!
            cum_energy = torch.cumsum(M_sorted.pow(2), dim=1) 
            # THE FIX: Binary search bypasses the 4GB sum() reduction completely!
            counts = torch.searchsorted(cum_energy, dust_budget.unsqueeze(1)).squeeze(1)
            anchor_indices = (counts - 1).clamp(min=0)
            active_dust = torch.gather(M_sorted, 1, anchor_indices.unsqueeze(1)).squeeze(1)

        active_scale = raw_max_chunk - active_dust

        # ==========================================
        # V27: ROCM-SAFE MICRO-GRID + 4-STAGE DESCENT
        # ==========================================
        batch_len = M_untested.shape[0]
        a = torch.nn.Parameter(torch.zeros(batch_len, device=device, dtype=torch.float32))
        b = torch.nn.Parameter(torch.full((batch_len,), 0.1, device=device, dtype=torch.float32))
        m = torch.nn.Parameter(torch.full((batch_len,), 1.0, device=device, dtype=torch.float32)) 
        n = torch.nn.Parameter(torch.full((batch_len,), 3.0, device=device, dtype=torch.float32)) 
        
        safe_norm = norm_indices.unsqueeze(0) + 1e-5

        # --- 1. THE ROCM-SAFE 1D MICRO-GRID ---
        with torch.no_grad():
            best_loss = torch.full((batch_len,), float('inf'), device=device, dtype=torch.float32)
            best_a = torch.zeros_like(a)
            best_m = torch.ones_like(m)

            # This is your scatter map for the middleweights!
            # The V28 Tuned Scatter Map for the Middleweights (a, m)
            test_points = [
                (-2.5, 1.25), 
                ( 2.5, 0.80), 
                ( 0.4, 1.85), 
                (-1.3, 1.50), 
                ( 0.0, 1.00)
            ]
            for test_a, test_m in test_points:
                linear_1d = (1.0 - test_a - 0.1) * norm_indices
                curve_1d = linear_1d + test_a * ((norm_indices + 1e-5) ** test_m) + 0.1 * ((norm_indices + 1e-5) ** 3.0)
                curve_1d = torch.clamp(curve_1d, 0.0, 1.0)
                
                bin_chunk = (curve_1d.unsqueeze(0) * active_scale.unsqueeze(1)) + active_dust.unsqueeze(1)
                loss_fit = torch.nn.functional.mse_loss(bin_chunk.to(oracle_targets.dtype), oracle_targets, reduction='none').mean(dim=1).to(torch.float32)
                
                improved = loss_fit < best_loss
                best_loss[improved] = loss_fit[improved]
                best_a[improved] = test_a
                best_m[improved] = test_m

            a.data.copy_(best_a)
            m.data.copy_(best_m)

        # --- 2. THE OPTIMIZERS ---
        opt_tips = torch.optim.AdamW([b, n], lr=0.1, weight_decay=0.01)
        opt_plateau = torch.optim.AdamW([a, m], lr=0.1, weight_decay=0.01)
        opt_joint = torch.optim.AdamW([a, b, m, n], lr=0.05, weight_decay=0.01)

        def compute_loss():
            a_c = a.unsqueeze(1)
            b_c = torch.clamp(b, 0.0, 1.0).unsqueeze(1)
            m_c = torch.clamp(m, 0.1, 4.0).unsqueeze(1)
            n_c = torch.clamp(n, 1.0, 8.0).unsqueeze(1)

            linear = (1.0 - a_c - b_c) * safe_norm
            curve = linear + a_c * (safe_norm ** m_c) + b_c * (safe_norm ** n_c)
            curve = torch.clamp(curve, 0.0, 1.0)

            bin_chunk = (curve * active_scale.unsqueeze(1)) + active_dust.unsqueeze(1)
            loss_fit = torch.nn.functional.mse_loss(bin_chunk.to(oracle_targets.dtype), oracle_targets, reduction='none').mean(dim=1)
            diffs = curve[:, 1:] - curve[:, :-1]
            loss_mono = torch.relu(-diffs).sum(dim=1) * 100.0

            return (loss_fit + loss_mono).mean()

        # Stages 1-4
        for _ in range(10): opt_tips.zero_grad(); compute_loss().backward(); opt_tips.step()
        for _ in range(10): opt_plateau.zero_grad(); compute_loss().backward(); opt_plateau.step()
        for _ in range(10): opt_tips.zero_grad(); compute_loss().backward(); opt_tips.step()
        for _ in range(10): opt_joint.zero_grad(); compute_loss().backward(); opt_joint.step()

        # --- 3. THE EVALUATION STAGE ---
        with torch.no_grad():
            # 1. Generate the continuous polynomial curve
            a_f = a.detach().clone().flatten()
            b_f = torch.clamp(b.detach(), 0.0, 1.0).clone().flatten()
            m_f = torch.clamp(m.detach(), 0.1, 4.0).clone().flatten()
            n_f = torch.clamp(n.detach(), 1.0, 8.0).clone().flatten()

            a_c = a_f.unsqueeze(1)
            b_c = b_f.unsqueeze(1)
            m_c = m_f.unsqueeze(1)
            n_c = n_f.unsqueeze(1)

            linear = (1.0 - a_c - b_c) * safe_norm
            curve = linear + a_c * (safe_norm ** m_c) + b_c * (safe_norm ** n_c)
            curve = torch.clamp(curve, 0.0, 1.0)

            # 2. Build the Look-Up Table
            lut = (curve * active_scale.unsqueeze(1)) + active_dust.unsqueeze(1)
            
            # 3. Micro-Batched 2D Sweep (Zero Global Allocation)
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

            # 4. Map back for the Sieve
            closest_ints = closest_ints_global

        # --- 4. THE SIEVE UPDATE ---
        # The Sieve now actively utilizes the leniency multiplier!
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
    # FINAL BIT PACKING (SWAR)
    # ==========================================
    packed_bytes = []
    offsets = []
    current_offset = 0
    powers_of_2 = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=device, dtype=torch.uint8)
    
    for i in range(out_features):
        final_d = int(best_D[i].item())
        if final_d == 0: 
            final_d = 8  
            best_D[i] = 8
            
        mag_bits = final_d - 1
        r_ints = best_ints[i].view(-1, 8)
        r_signs = signs[i].view(-1, 8) 
        r_bytes = []
        
        for bit_idx in range(mag_bits):
            bit_slice = (r_ints >> bit_idx) & 1
            r_bytes.append((bit_slice * powers_of_2).sum(dim=1).to(torch.uint8))
        
        sign_byte = (r_signs * powers_of_2).sum(dim=1).to(torch.uint8)
        r_bytes.append(sign_byte)
        packed_row = torch.stack(r_bytes, dim=1).flatten()
        
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
    layer_names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    
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
        vbr_result = compress_vbr_v25_matrix(name, weight) 
        
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
