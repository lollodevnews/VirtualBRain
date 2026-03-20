import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM

print("==========================================")
print(" VIRTUALBRAIN V18 DYNAMIC SCALE SWARM     ")
print("==========================================")

MODEL_ID = os.path.expanduser("~/models/quant/qwen25_7b")
OUTPUT_WEIGHTS = os.path.expanduser("~/models/quant/v18_swarm_council.pt")
DEVICE = "cuda:0"
NUM_ENERGY_POINTS = 512  
TRAIN_STEPS_PER_ORACLE = 15000
BATCH_SIZE = 128
SWARM_SIZE = 8 

class Swarm_VBR_Oracle_V18(nn.Module):
    def __init__(self, num_points=512, k=8):
        super().__init__()
        self.k = k
        self.base_extractor = nn.Sequential(
            nn.Linear(num_points, 1024), nn.GELU(),
            nn.Linear(1024, 256), nn.GELU()
        )
        
        ctx = (k * 5) + k # K shapes (5 params each) + K errors
        
        # Stage 2 ingests the Base + Stage 1 Grid context
        self.stage2_extractor = nn.Sequential(nn.Linear(256 + ctx, 256), nn.GELU())
        self.stage2_head = nn.Linear(256, k * 5)
        
        # Stage 3 ingests Base + Stage 1 Grid + Stage 2 Guesses
        self.stage3_extractor = nn.Sequential(nn.Linear(256 + (ctx * 2), 256), nn.GELU())
        self.stage3_head = nn.Linear(256, k * 5)
        
        # THE 8 EQUIDISTANT GRID ANCHORS: [scale_raw, a_raw, b_raw, m_raw, n_raw]
        # scale_raw = 0.0 translates to a physical scale of ~0.69, preventing initial overshoots
        self.register_buffer("grid_anchors", torch.tensor([
            [0.0, 0.0, 0.0, -2.0, -2.0],  # 1. Pure Linear
            [0.0, 3.0, 0.0, 1.0, 0.0],    # 2. Pure Poly M
            [0.0, 0.0, 3.0, 0.0, 4.0],    # 3. Pure Poly N
            [0.0, 3.0, 3.0, 2.0, 6.0],    # 4. Dual Exponential
            [0.0, -3.0, 3.0, -1.0, 4.0],  # 5. Dip and Spike
            [0.0, 3.0, -3.0, 0.0, 2.0],   # 6. Convex Core
            [0.0, 0.0, 3.0, 0.0, 8.0],    # 7. Extreme Outlier
            [0.0, -3.0, 0.0, 0.0, -2.0]   # 8. Core Preserver
        ], dtype=torch.float32))

    def decode_params(self, raw_params):
        reshaped = raw_params.view(-1, 5)
        scale = F.softplus(reshaped[:, 0:1]) + 1e-8 # THE NETWORK RECLAIMS SCALE
        a = torch.tanh(reshaped[:, 1:2]) 
        b = torch.tanh(reshaped[:, 2:3])
        m = F.softplus(reshaped[:, 3:4]) + 1.0 
        n = F.softplus(reshaped[:, 4:5]) + 2.0 
        return scale, a, b, m, n

    def simulate_swarm(self, scale, a, b, m, n, norm_mags, norm_dust, num_bins):
        batch_size = norm_mags.shape[0]
        mag_k = norm_mags.repeat_interleave(self.k, dim=0)
        dust_k = norm_dust.repeat_interleave(self.k, dim=0)
        #scale_k = scale.repeat_interleave(self.k, dim=0)
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
        
        return (error_energy / signal_energy).view(batch_size, self.k)

    def forward(self, energy_signature, norm_mags, norm_dust, num_bins):
        features = self.base_extractor(energy_signature)
        batch_size = features.shape[0]
        
        def get_ctx(raw_p, err): return torch.cat([raw_p.view(batch_size, -1), err], dim=1)
        
        # === STAGE 1: THE PHYSICAL GRID ===
        raw_p1 = self.grid_anchors.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        s1, a1, b1, m1, n1 = self.decode_params(raw_p1)
        err1 = self.simulate_swarm(s1, a1, b1, m1, n1, norm_mags, norm_dust, num_bins)
        
        # === STAGE 2: THE RESIDUAL NUDGE ===
        s2_in = torch.cat([features, get_ctx(raw_p1, err1)], dim=1)
        raw_p2 = raw_p1.view(batch_size, -1) + self.stage2_head(self.stage2_extractor(s2_in))
        s2, a2, b2, m2, n2 = self.decode_params(raw_p2)
        err2 = self.simulate_swarm(s2, a2, b2, m2, n2, norm_mags, norm_dust, num_bins)
        
        # === STAGE 3: FINAL POLISH ===
        s3_in = torch.cat([features, get_ctx(raw_p1, err1), get_ctx(raw_p2, err2)], dim=1)
        raw_p3 = raw_p2 + self.stage3_head(self.stage3_extractor(s3_in))
        s3, a3, b3, m3, n3 = self.decode_params(raw_p3)
        err3 = self.simulate_swarm(s3, a3, b3, m3, n3, norm_mags, norm_dust, num_bins)
        
        return err1, err2, err3

def main():
    print("[*] Loading Qwen 2.5 7B into System RAM...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map="cpu")
    all_rows = [m.weight.data for n, m in model.named_modules() if isinstance(m, nn.Linear)]

    x_query = torch.linspace(0.0, 1.0, NUM_ENERGY_POINTS, device=DEVICE).unsqueeze(0).expand(BATCH_SIZE, -1)
    council_state_dicts = {}
    
    for target_D_int in range(2, 9):
        print(f"\n==========================================")
        print(f"[*] SPAWNING V18 DYNAMIC SWARM FOR Q{target_D_int}")
        print(f"==========================================")
        
        oracle = Swarm_VBR_Oracle_V18(num_points=NUM_ENERGY_POINTS, k=SWARM_SIZE).to(DEVICE)
        optimizer = torch.optim.AdamW(oracle.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_STEPS_PER_ORACLE, eta_min=1e-5)
        
        num_bins = (2 ** (target_D_int - 1)) - 1
        
        pbar = tqdm(range(TRAIN_STEPS_PER_ORACLE), desc=f"Training Swarm Q{target_D_int}")
        for step in pbar:
            optimizer.zero_grad()
            
            matrix = random.choice(all_rows)
            row_indices = torch.randint(0, matrix.shape[0], (BATCH_SIZE,))
            raw_magnitudes = torch.abs(matrix[row_indices].to(DEVICE))
            
            # PERFECT STABLE NORMALIZATION
            max_vals, _ = torch.max(raw_magnitudes, dim=1, keepdim=True)
            max_vals = max_vals.clamp(min=1e-8)
            norm_mags = raw_magnitudes / max_vals 
            
            k_dust = max(1, int(norm_mags.size(1) * 0.03))
            bottom_k_values, _ = torch.topk(norm_mags, k_dust, dim=1, largest=False)
            norm_dust = torch.mean(bottom_k_values, dim=1, keepdim=True)
            
            # CDF Signature based on normalized physical layout
            sorted_mags, _ = torch.sort(norm_mags, dim=1)
            energy = sorted_mags ** 2
            cumulative_energy = torch.cumsum(energy, dim=1)
            energy_cdf = cumulative_energy / (cumulative_energy[:, -1:] + 1e-8)
            
            x_search = sorted_mags / (sorted_mags[:, -1:] + 1e-8)
            indices = torch.clamp(torch.searchsorted(x_search, x_query), 0, norm_mags.size(1) - 1)
            energy_signature = torch.gather(energy_cdf, 1, indices)
            
            err1, err2, err3 = oracle(energy_signature, norm_mags, norm_dust, num_bins)
            
            best_err3, _ = torch.min(err3, dim=1)
            best_err2, _ = torch.min(err2, dim=1)
            
            loss = best_err3.mean() + (best_err2.mean() * 0.1)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(oracle.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            if step % 50 == 0:
                best_err1, _ = torch.min(err1, dim=1)
                pbar.set_postfix({
                    "Grid_L1": f"{best_err1.mean().item():.4f}", 
                    "Final_L3": f"{best_err3.mean().item():.4f}"
                })

        council_state_dicts[f"Q{target_D_int}"] = oracle.state_dict().copy()

    torch.save(council_state_dicts, OUTPUT_WEIGHTS)
    print(f"\n[+] V18 Dynamic Swarm Council saved to {OUTPUT_WEIGHTS}")

if __name__ == "__main__":
    main()
