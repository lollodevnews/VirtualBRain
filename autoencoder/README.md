# VirtualBrain VBR: The Non-Linear Grid Search Engine (V34)
**A Variable BitRate (VBR) bare-metal quantization framework.**

> **THE EVOLUTION OF THIS PROJECT:**
The journey started by wanting to move away from brute-force compression algorithms, which initially led us to a simple 1-layer neural network that yielded minimal compression. We then implemented a 4-layer neural network using predetermined anchor points. With this latest iteration, we move away from fixed points entirely into a Monte Carlo simulation, where the random scatter of points combined with gradient descent and localized grid-search optimizations offers the absolute best compression performance so far.

---

## 1. The Zero-Crutch Philosophy (Row-Wise vs. Group-Wise)

The open-source quantization community relies on a shared deception: **Group-Wise Scaling**. To make standard 4-bit models (like AWQ or GGUF) retain their intelligence, they chop rows into tiny 64-weight blocks and inject gigabytes of hidden FP16 metadata (scales and zero-points) to prop up the math. 

**VirtualBrain VBR abandons group-wise scaling entirely.**
Instead of relying on hidden FP16 grids, VBR uses a custom Autoencoder to mathematically model the weight distribution of an *entire row* using a continuous polynomial S-Curve. One scale, one dust anchor, and one curve per row.

---

## 2. The Mathematical Formulation & The $k$ Substitution

We fit a 4-parameter polynomial curve ($a, b, m, n$) to hyper-focus the quantization bins around the matrix's dense core, while expanding the tails to naturally absorb violent outliers.

The physical curve is defined as:
y = (1 - a - b)x + a(x^m) + b(x^n)

**The Taylor Series Limit ($k$ Substitution):**
In previous versions, the parameter $a$ caused massive instability as $m$ approached 1.0. We discovered that the physical "bend" of the curve collapses as $m$ nears 1. To prevent the autoencoder from wasting compute on extreme, invalid geometries, V34 normalizes the search space using a constant bend factor $k$:
a = k / (1 - m)
By optimizing for $k \in [-2.0, 2.0]$ instead of $a$, the solver evaluates geometrically perfect curves across the entire spectrum.

---

## 3. Alternating Grid Search & Domain Splitting

To optimize these curves across complex non-linear bounds, V34 utilizes a **Multi-Stage Block Coordinate Descent** (a Monte Carlo grid search pipeline).

* **Alternating Evaluation:** The solver strictly isolates the structural outliers ($b, n$) from the dense core ($k, m$), ping-ponging between them to find the perfect macro-shape, followed by a dynamically scaled $\pm$ micro-jitter.
* **The Root Curve Unlock:** If the $m$ exponent is sampled uniformly, the solver is blinded to crucial "Root" geometries ($x^{0.5}$, etc.). V34 physically splits the domain generation: 50% of the guesses are forced into the 0.15 to 0.95 Root Zone to hug hyper-dense, zero-centered weights, while the other 50% map the 1.05 to 8.0 Polynomial Zone.

---

## 4. Modulating Noise, Not Bits (Layer-Aware Routing)

The true breakthrough of VirtualBrain VBR is that **we modulate noise, not bits.** Instead of forcing a global, hardcoded bit-depth (like strict INT4), the VBR Sieve actively evaluates the Mean Squared Error (MSE) of every single row during compilation and dynamically assigns it a bit-depth purely based on its noise tolerance.

To achieve true Pareto efficiency, the energy thresholds are highly modular and strictly enforced based on the neural architecture:
* **Attention Tensors (`q_proj`, `k_proj`):** Hyper-sensitive to angular rotation. The Sieve enforces a strict **0.0005 (0.05%)** maximum energy loss, naturally promoting them to higher bitrates to protect the model's context recall.
* **Expert / MLP Tensors:** Highly robust to noise. The Sieve applies a relaxed **0.01 (1%)** error allowance, crushing them down to 3-bit or 4-bit arrays to save massive amounts of VRAM without sacrificing intelligence.

---

## 5. The Hard Numbers (Qwen 2.5 7B)

Unlike standard repositories, we publish the exact mathematical degradation to prove the structural coherence of our flat file sizes. Benchmarked on an AMD Instinct MI50.

| Architecture | Total File Size | Bits Per Weight | WikiText-2 Perplexity | Degradation | MI50 Inference Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Base (FP16)** | 14.0 GB | 16.0 bpw | ~6.1400 | - | - |
| **V28 (AdamW)** | 4.80 GB | ~5.48 bpw | 6.4656 | +0.3256 | - |
| **V34 (Grid Search)** | **5.06 GB** | **~5.80 bpw** | **6.2285** | **+0.0885** | **22.83 T/s** |

*Note: The 5.06 GB footprint is the strict, effective flat file size. It encompasses compressed matrices (`compressed_01.pt` - `compressed_04.pt`), all polynomial headers, scale vectors, and VBR byte maps. Zero group-wise bloat.*

---

## 6. How to Run the Compressor

### Step 1: Compile the Model (The Autoencoder)
Because VBR evaluates massive mathematical grids, the compression is split into chunks to protect GPU memory. Run the Autoencoder iteratively across your available GPUs:

```bash
python3 autoencoder.py --chunk_idx 0 --gpu 0 --total_chunks 4
python3 autoencoder.py --chunk_idx 1 --gpu 1 --total_chunks 4
python3 autoencoder.py --chunk_idx 2 --gpu 2 --total_chunks 4
python3 autoencoder.py --chunk_idx 3 --gpu 3 --total_chunks 4
```

### Step 2: Inference & Verification
Run the bare-metal HIP engine to verify continuous generation, or launch the sliding-window Perplexity benchmark to measure the exact mathematical degradation:

```bash
# Test generation and Tokens/Sec speed
python3 inference.py

# Evaluate WikiText-2 Perplexity
python3 perplexity.py
```
