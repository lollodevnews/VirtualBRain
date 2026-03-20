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
import torch.nn as nn
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==============================================================================
# 🎛️ SYSTEM CONTROL DECK
# ==============================================================================
MODEL_ID = "Qwen/Qwen1.5-0.5B"
COMPILED_DIR = "./vbr_compiled_qwen_0.5B"
# ==============================================================================

class PoorMansFPLinear(nn.Module):
    """Zero-math Bit-Shift Reconstruction for Attention Geometry."""
    def __init__(self, packed_dict, in_features, out_features, device, compute_dtype):
        super().__init__()
        self.W_packed = packed_dict["W_packed_fp_hack"].to(device)
        self.target_bits = packed_dict.get("target_bits", 8)
        self.drop_bits = 16 - self.target_bits
        self.compute_dtype = compute_dtype
        self.bias = None 

    def forward(self, x):
        W_int16 = self.W_packed.to(torch.int16)
        W_recon_int16 = W_int16 << self.drop_bits
        W_recon_fp16 = W_recon_int16.view(torch.float16)
        
        out = torch.matmul(x.to(self.compute_dtype), W_recon_fp16.to(self.compute_dtype).t())
        if self.bias is not None:
            out += self.bias
        return out.to(torch.float16)


class TrueVBRLinear(nn.Module):
    def __init__(self, packed_dict, in_features, out_features, device, compute_dtype):
        super().__init__()
        self.W_mag_int = packed_dict["W_mag_int"].to(device)
        self.W_sign = packed_dict["W_sign"].to(device)
        
        self.scales = packed_dict["vbr_scales"].to(device)
        self.alphas = packed_dict["alphas"].to(device).to(compute_dtype)
        self.row_powers = packed_dict["row_powers"].to(device).to(compute_dtype)
        
        # NEW: The per-row dynamic divisor (1.0, 3.0, or 15.0)
        self.row_divisors = packed_dict["row_divisors"].to(device).to(compute_dtype)
        
        self.compute_dtype = compute_dtype
        self.bias = None

    def forward(self, x):
        # 1. Normalize dynamically based on what bit-depth that row won!
        X_norm = self.W_mag_int.to(self.compute_dtype) / self.row_divisors
        
        # 2. Hybrid Gear-Shift Curve
        M_curved = (self.alphas * X_norm) + ((1.0 - self.alphas) * torch.pow(X_norm, self.row_powers))
        
        # 3. Scale up and Apply Sign
        M_recon = M_curved * self.scales
        W_recon = self.W_sign.to(self.compute_dtype) * M_recon
        
        out = torch.matmul(x.to(self.compute_dtype), W_recon.t())
        if self.bias is not None:
            out += self.bias
            
        return out.to(torch.float16)


def inject_vbr_modules(target_module, packed_dict, prefix, device, compute_dtype):
    for name, child in target_module.named_children():
        full_key = f"{prefix}.{name}.weight"
        bias_key = f"{prefix}.{name}.bias"
        
        if isinstance(child, nn.Linear) and full_key in packed_dict and isinstance(packed_dict[full_key], dict):
            # 1. Spawn the correct logic layer
            if "W_packed_fp_hack" in packed_dict[full_key]:
                vbr_layer = PoorMansFPLinear(packed_dict[full_key], child.in_features, child.out_features, device, compute_dtype)
            else:
                vbr_layer = TrueVBRLinear(packed_dict[full_key], child.in_features, child.out_features, device, compute_dtype)
            
            # 2. Safely extract and mount the Bias
            if bias_key in packed_dict and packed_dict[bias_key] is not None:
                vbr_layer.bias = nn.Parameter(packed_dict[bias_key].to(device).to(compute_dtype), requires_grad=False)
            elif hasattr(child, 'bias') and child.bias is not None:
                vbr_layer.bias = nn.Parameter(child.bias.data.to(device).to(compute_dtype), requires_grad=False)
                
            setattr(target_module, name, vbr_layer)
            
        elif isinstance(child, nn.Linear) and full_key in packed_dict:
            child.weight.data = packed_dict[full_key].to(device)
            if bias_key in packed_dict and child.bias is not None:
                child.bias.data = packed_dict[bias_key].to(device)
        else:
            inject_vbr_modules(child, packed_dict, f"{prefix}.{name}", device, compute_dtype)


def load_virtualBrain_graph(model_id, compiled_dir, target_device, compute_dtype):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cpu")
    
    with open(os.path.join(compiled_dir, "vbr_config.json"), "r") as f:
        config = json.load(f)
        
    attn_name, mlp_name = config["attention_module"], config["mlp_module"]
    ln1_name, ln2_name = config["input_layernorm"], config["post_attention_layernorm"]
    
    sys_path = os.path.join(compiled_dir, "system_analog.pt")
    system_dict = torch.load(sys_path, map_location=target_device, weights_only=True)
    
    model.model.embed_tokens.weight.data = system_dict["model.embed_tokens.weight"].to(target_device)
    model.model.norm.weight.data = system_dict["model.norm.weight"].to(target_device)
    model.lm_head.weight.data = system_dict["lm_head.weight"].to(target_device)
    
    for i in range(len(model.model.layers)):
        layer_file = os.path.join(compiled_dir, f"layer_{i:02d}.pt")
        payload = torch.load(layer_file, map_location=target_device, weights_only=True)
        hf_layer = model.model.layers[i]
        
        try:
            attn_block = getattr(hf_layer, attn_name)
        except AttributeError:
            layer_modules = dict(hf_layer.named_children())
            live_attn_name = next((k for k in layer_modules.keys() if 'attn' in k or 'attention' in k), None)
            if live_attn_name: attn_block = getattr(hf_layer, live_attn_name)
            else: raise RuntimeError(f"Architecture failure Layer {i}.")

        inject_vbr_modules(attn_block, payload["attention_hot"], attn_name, target_device, compute_dtype)
        
        mlp_block = getattr(hf_layer, mlp_name)
        inject_vbr_modules(mlp_block, payload["experts_cold"], mlp_name, target_device, compute_dtype)
        
        getattr(hf_layer, ln1_name).weight.data = payload["experts_cold"][f"{ln1_name}.weight"].to(target_device)
        getattr(hf_layer, ln2_name).weight.data = payload["experts_cold"][f"{ln2_name}.weight"].to(target_device)

    return model.to(target_device)

def detect_hardware():
    if torch.cuda.is_available(): return "cuda", torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps", torch.float16
    else: return "cpu", torch.float32

if __name__ == "__main__":
    device, compute_dtype = detect_hardware()
    print(f"\nBooting VirtualBRain Engine on [{device.upper()}] with core precision [{compute_dtype}]...")
    
    model = load_virtualBrain_graph(MODEL_ID, COMPILED_DIR, device, compute_dtype)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    prompt = "The architecture of a neural network can be described as"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print("\nVirtualBRain Engine generating...")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    print(f"\nResponse:\n{tokenizer.decode(outputs[0], skip_special_tokens=True)}")
