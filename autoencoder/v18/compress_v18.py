import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import gc
import math
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM

print("==========================================")
print(" VIRTUALBRAIN V18 SWARM COMPILER          ")
print("==========================================")

# --- CONFIGURATION ---
MODEL_ID = os.path.expanduser("~/models/quant/qwen25_7b")
OUTPUT_DIR = os.path.expanduser("~/models/quant/vbr_qwen25_v18")
COUNCIL_WEIGHTS = os.path.expanduser("~/models/quant/v18_swarm_council.pt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_ENERGY_POINTS = 512
MAX_ENERGY_ERROR_EXPERT = 0.005     
MAX_ENERGY_ERROR_ATTENTION = 0.001  
SWARM_SIZE = 8

class Swarm_VBR_Oracle_V18_Compiler(nn.Module):
    def __init__(self, num_points=512, k=8):
        super().__init__()
        self.k = k
        self.base_extractor = nn.Sequential(
            nn.Linear(num_points, 1024), nn.GELU(),
            nn.Linear(1024, 256), nn.GELU()
        )
        ctx = (k * 5) + k 
        self.stage2_extractor = nn.Sequential(nn.Linear(256 + ctx, 256), nn.GELU())
        self.stage2_head = nn.Linear(256, k * 5)
        self.stage3_extractor = nn.Sequential(nn.Linear(256 + (ctx * 2), 256), nn.GELU())
        self.stage3_head = nn.Linear(256, k * 5)
        
        self.register_buffer("grid_anchors", torch.tensor([
            [0.0, 0.0, 0.0, -2.0, -2.0],  [0.0, 3.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 4.0],    [0.0, 3.0, 3.0, 2.0, 6.0],
            [0.0, -3.0, 3.0, -1.0, 4.0],  [0.0, 3.0, -3.0, 0.0, 2.0],
            [0.0, 0.0, 3.0, 0.0, 8.0],    [0.0, -3.0, 0.0, 0.0, -2.0]
        ], dtype=torch.float32))

    def decode_params(self, raw_params):
        reshaped = raw_params.view(-1, 5)
        scale = F.softplus(reshaped[:, 0:1]) + 1e-8 
        a = torch.tanh(reshaped[:, 1:2]) 
        b = torch.tanh(reshaped[:, 2:3])
        m = F.softplus(reshaped[:, 3:4]) + 1.0 
        n = F.softplus(reshaped[:, 4:5]) + 2.0 
        return scale, a, b, m, n

    def simulate_swarm(self, scale, a, b, m, n, norm_mags, norm_dust, num_bins, return_ints=False):
        batch_size = norm_mags.shape[0]
        mag_k = norm_mags.repeat_interleave(self.k, dim=0)
        dust_k = norm_dust.repeat_interleave(self.k, dim=0)
        scale_safe = scale.clamp(min=1e-8) 
        
        norm_w = (mag_k - dust_k).clamp(min=0.0) / scale_safe
        norm_w = torch.clamp(norm_w, 0.0, 1.0)
        
        linear_term = (1.0 - a - b) * norm_w
        curve = linear_term + (a * (norm_w ** m)) + (b * (norm_w ** n))
        curve = torch.clamp(curve, 0.0, 1.0)
        
        scaled_curve = curve * num_bins
        quantized_ints = torch.round(scaled_curve)
        quantized_ste = scaled_curve + (quantized_ints - scaled_curve).detach()
        
        reconstructed_curve = quantized_ste / num_bins
        reconstructed_mag = (reconstructed_curve * scale_safe) + dust_k
        
        error_energy = torch.sum((reconstructed_mag - mag_k) ** 2, dim=1, keepdim=True)
        signal_energy = torch.sum(mag_k ** 2, dim=1, keepdim=True) + 1e-8
        
        err = (error_energy / signal_energy).view(batch_size, self.k)
        if return_ints:
            return err, quantized_ints.view(batch_size, self.k, -1)
        return err

    def forward(self, energy_signature, norm_mags, norm_dust, num_bins):
        features = self.base_extractor(energy_signature)
        batch_size = features.shape[0]
        def get_ctx(raw_p, err): return torch.cat([raw_p.view(batch_size, -1), err], dim=1)
        
        raw_p1 = self.grid_anchors.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        s1, a1, b1, m1, n1 = self.decode_params(raw_p1)
        err1 = self.simulate_swarm(s1, a1, b1, m1, n1, norm_mags, norm_dust, num_bins)
        
        s2_in = torch.cat([features, get_ctx(raw_p1, err1)], dim=1)
        raw_p2 = raw_p1.view(batch_size, -1) + self.stage2_head(self.stage2_extractor(s2_in))
        s2, a2, b2, m2, n2 = self.decode_params(raw_p2)
        err2 = self.simulate_swarm(s2, a2, b2, m2, n2, norm_mags, norm_dust, num_bins)
        
        s3_in = torch.cat([features, get_ctx(raw_p1, err1), get_ctx(raw_p2, err2)], dim=1)
        raw_p3 = raw_p2 + self.stage3_head(self.stage3_extractor(s3_in))
        s3, a3, b3, m3, n3 = self.decode_params(raw_p3)
        
        # We ask the final simulation to hand us the raw integers!
        err3, final_ints = self.simulate_swarm(s3, a3, b3, m3, n3, norm_mags, norm_dust, num_bins, return_ints=True)
        return raw_p3, err3, final_ints

def load_oracle_council(device):
    print(f"[*] Awakening the V18 Swarm Council on {device}...")
    council = {}
    state_dicts = torch.load(COUNCIL_WEIGHTS, map_location=device)
    for target_D in range(2, 9):
        oracle = Swarm_VBR_Oracle_V18_Compiler(num_points=NUM_ENERGY_POINTS, k=SWARM_SIZE).to(device)
        oracle.load_state_dict(state_dicts[f"Q{target_D}"])
        oracle.eval() 
        council[target_D] = oracle
    return council

def compile_vbr_v18_matrix(name, weight_tensor, council, position=0):
    if "mlp" in name: threshold = MAX_ENERGY_ERROR_EXPERT
    elif "self_attn" in name: threshold = MAX_ENERGY_ERROR_ATTENTION
    else: return None 

    out_features, in_features = weight_tensor.shape
    device = weight_tensor.device
    
    signs = (weight_tensor < 0).to(torch.uint8) 
    raw_magnitudes = torch.abs(weight_tensor).float().detach()
    
    max_vals, _ = torch.max(raw_magnitudes, dim=1, keepdim=True)
    max_vals = max_vals.clamp(min=1e-8)
    norm_mags = raw_magnitudes / max_vals 
    
    k_dust = max(1, int(in_features * 0.03))
    bottom_k_values, _ = torch.topk(norm_mags, k_dust, dim=1, largest=False)
    norm_dust = torch.mean(bottom_k_values, dim=1, keepdim=True)
    
    sorted_mags, _ = torch.sort(norm_mags, dim=1)
    energy = sorted_mags ** 2
    cumulative_energy = torch.cumsum(energy, dim=1)
    energy_cdf = cumulative_energy / (cumulative_energy[:, -1:] + 1e-8)
    
    x_query = torch.linspace(0.0, 1.0, NUM_ENERGY_POINTS, device=device).unsqueeze(0).expand(out_features, -1)
    x_search = (sorted_mags / (sorted_mags[:, -1:] + 1e-8)).contiguous()
    indices = torch.clamp(torch.searchsorted(x_search, x_query), 0, in_features - 1)
    full_energy_signatures = torch.gather(energy_cdf, 1, indices)
    
    best_D = torch.full((out_features,), 8, dtype=torch.uint8, device=device)
    best_ints = torch.zeros((out_features, in_features), dtype=torch.uint8, device=device)
    best_scale = torch.zeros(out_features, device=device)
    best_a = torch.zeros(out_features, device=device)
    best_b = torch.zeros(out_features, device=device)
    best_m = torch.ones(out_features, device=device)
    best_n = torch.ones(out_features, device=device)
    
    row_resolved = torch.zeros(out_features, dtype=torch.bool, device=device)
    chunk_size = 512 
    
    short_name = name.split(".")[-1]
    pbar = tqdm(range(0, out_features, chunk_size), desc=f"[{device}] {short_name:<10}", position=position, leave=False, ncols=100)
    
    with torch.no_grad():
        for c in pbar:
            end = min(c + chunk_size, out_features)
            M_norm_chunk = norm_mags[c:end]
            dust_norm_chunk = norm_dust[c:end]
            max_vals_chunk = max_vals[c:end]
            energy_sig_chunk = full_energy_signatures[c:end]
            chunk_resolved = torch.zeros(end - c, dtype=torch.bool, device=device)
            
            for target_D in range(2, 9):
                if chunk_resolved.all(): break 
                num_bins = (2 ** (target_D - 1)) - 1
                
                raw_p3, err3, final_ints = council[target_D](energy_sig_chunk, M_norm_chunk, dust_norm_chunk, num_bins)
                best_err3, best_agent_idx = torch.min(err3, dim=1)
                
                passed = (best_err3 < threshold) & (~chunk_resolved)
                if passed.any():
                    reshaped_p3 = raw_p3.view(end - c, SWARM_SIZE, 5)
                    winning_idx_expanded = best_agent_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, 5)
                    winning_params = torch.gather(reshaped_p3, 1, winning_idx_expanded).squeeze(1)
                    
                    # DECODE AND RE-MULTIPLY BY THE ACTUAL ROW MAX TO GET TRUE PHYSICAL SCALE
                    w_scale = (F.softplus(winning_params[:, 0:1]) + 1e-8) * max_vals_chunk
                    w_a = torch.tanh(winning_params[:, 1])
                    w_b = torch.tanh(winning_params[:, 2])
                    w_m = F.softplus(winning_params[:, 3]) + 1.0
                    w_n = F.softplus(winning_params[:, 4]) + 2.0
                    
                    winning_ints_idx = best_agent_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, in_features)
                    winning_ints = torch.gather(final_ints, 1, winning_ints_idx).squeeze(1)
                    
                    best_D[c:end][passed] = target_D
                    best_ints[c:end][passed] = winning_ints[passed].to(torch.uint8)
                    best_scale[c:end][passed] = w_scale.squeeze(-1)[passed] # Store absolute scale!
                    best_a[c:end][passed] = w_a[passed]
                    best_b[c:end][passed] = w_b[passed]
                    best_m[c:end][passed] = w_m[passed]
                    best_n[c:end][passed] = w_n[passed]
                    
                    chunk_resolved[passed] = True

            if not chunk_resolved.all():
                return None 

    pbar.close()

    # --- BIT PACKING ENGINE (Unchanged V11/V15 Format) ---
    packed_bytes = []
    offsets = []
    current_offset = 0
    powers_of_2 = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=device, dtype=torch.uint8)
    best_divisors = (2 ** (best_D - 1)) - 1
    shift_zeros = torch.zeros(out_features, dtype=torch.float16, device=device) 
    
    # Re-multiply normalized dust to get physical dust back
    true_dust_anchors = (norm_dust * max_vals).squeeze(-1)
    
    for i in range(out_features):
        D = int(best_D[i].item())
        mag_bits = D - 1
        
        r_ints = best_ints[i].view(-1, 8)
        r_signs = signs[i].view(-1, 8)
        r_bytes = []
        
        for bit_idx in range(mag_bits):
            bit_slice = (r_ints >> bit_idx) & 1
            r_bytes.append((bit_slice * powers_of_2).sum(dim=1).to(torch.uint8))
        
        r_bytes.append((r_signs * powers_of_2).sum(dim=1).to(torch.uint8))
        packed_row = torch.stack(r_bytes, dim=1).flatten()
        
        packed_bytes.append(packed_row)
        offsets.append(current_offset)
        current_offset += len(packed_row)
        
    vbr_data = torch.cat(packed_bytes)
    offsets_tensor = torch.tensor(offsets, dtype=torch.int32, device=device)
    
    return {
        "vbr_data": vbr_data.cpu(),
        "vbr_offsets": offsets_tensor.cpu(),
        "vbr_headers": best_D.cpu(), 
        "vbr_scales": best_scale.half().cpu(), 
        "dust_anchors": true_dust_anchors.half().cpu(), 
        "alpha_a": best_a.half().cpu(),
        "alpha_b": best_b.half().cpu(),
        "power_m": best_m.half().cpu(),
        "power_n": best_n.half().cpu(),
        "shift_e": shift_zeros.cpu(), 
        "shift_f": shift_zeros.cpu(), 
        "shift_g": shift_zeros.cpu(), 
        "row_divisors": best_divisors.half().cpu(),
        "original_shape": weight_tensor.shape,
        "is_vbr_compressed": True
    }

def main():
    parser = argparse.ArgumentParser(description="V18 Swarm Matrix Compiler")
    parser.add_argument("--chunk_idx", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--total_chunks", type=int, default=4)
    args = parser.parse_args()

    DEVICE = f"cuda:{args.gpu}"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cpu")
    layer_names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    
    chunk_size = math.ceil(len(layer_names) / args.total_chunks)
    chunk_names = layer_names[args.chunk_idx * chunk_size : (args.chunk_idx + 1) * chunk_size]
    
    council = load_oracle_council(DEVICE)
    chunk_dict = {}
    
    for name in chunk_names:
        weight = model.get_submodule(name).weight.detach().to(DEVICE)
        vbr_result = compile_vbr_v18_matrix(name, weight, council, position=args.gpu)
        
        if vbr_result is not None:
            if hasattr(model.get_submodule(name), "bias") and model.get_submodule(name).bias is not None:
                vbr_result["bias"] = model.get_submodule(name).bias.half().cpu()
            chunk_dict[name] = vbr_result
        else:
            chunk_dict[name] = {"raw_data": weight.half().cpu(), "is_vbr_compressed": False, "original_shape": weight.shape}
            
        del weight
        torch.cuda.empty_cache()
        gc.collect()
        
    out_file = os.path.join(OUTPUT_DIR, f"compressed_{str(args.chunk_idx + 1).zfill(2)}.pt")
    torch.save({"experts_cold": chunk_dict}, out_file)
    print(f"\n[+] GPU {args.gpu} saved to {out_file}")

if __name__ == "__main__":
    main()
