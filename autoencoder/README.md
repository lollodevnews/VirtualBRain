# VirtualBrain VBR: The Polynomial Quantization Engine (V28)
**A Variable Bitrate (VBR) quantization framework utilizing Continuous S-Curve Reparameterization.**

> **THE V28 REDEMPTION ARCHIVE:** Yesterday, in V18, we declared complex Autoencoders dead for this size of problem optimizations. The massive 14GB matrix geometry, the parameters range inherited by the brute force setup and the PyTorch ROCm memory bombs forced us to fall back to dumb brute-forcing or yielding a bloated 11GB model. 
> 
> **Today, we proved that conclusion wrong.** By expanding the number of available combinations for the parameters well beyond what was reasonably computable via brute force we managed to leverage the strengths of the autoencoder, using gradient descent instead of brute force to map a wider area of potential values, avoiding local minima thanks to the strategical placement of the new grid of starting values. Brute force is out. True continuous geometry is in.

---

## 1. The Zero-Crutch Philosophy (Row-Wise vs. Group-Wise)

The open-source quantization community relies on a shared deception: **Group-Wise Scaling**. To make "4-bit" models (like standard AWQ or GGUF) retain their intelligence, they chop rows into tiny 64-weight blocks and inject gigabytes of hidden FP16 metadata (scales and zero-points) to prop up the math. Their "4-bit" models actually consume ~5.0 bits per weight in total file size.

**VirtualBrain VBR abandons group-wise scaling entirely.**
Instead of relying on hidden FP16 grids, VBR uses a custom Autoencoder to mathematically model the weight distribution of an *entire row* using a continuous polynomial S-Curve. One scale, one dust anchor, and one curve per row.

---

## 2. The Mathematical Formulation

We fit a 4-parameter polynomial curve (a, b, m, n) to hyper-focus the quantization bins around the matrix's dense core, while expanding the tails to naturally absorb violent outliers.

The physical curve is defined as:
f(x) = (1 - a - b)x + a(x^m) + b(x^n)

Where:
* x is the normalized weight magnitude.
* a is unbound (-∞ to +∞). It controls the amplitude and inversion of the middleweight plateau, allowing negative coefficients to flip the curve's concavity.
* b in [0.0, 1.0] strictly controls the extreme tip outlier expansion.
* m in [0.1, 4.0] controls the concavity of the middleweights.
* n in [1.0, 8.0] dictates the extreme outlier exponential wall.

By dynamically compiling this curve, the compressor creates a highly efficient binary layout ready for dynamic decompression, entirely bypassing iterative lookup tables.

---

## 3. The Pareto-Optimal VBR Sieve

The true breakthrough of V28 is the **Dynamic Signal-to-Noise (SNR) Sieve**. 
Instead of forcing a global bit-depth, the Sieve actively evaluates the Energy Loss (Relative Squared Error) of every single row during compilation and dynamically assigns it a bit-depth from 2-bit (Q2) to 8-bit (Q8).

To achieve true Pareto efficiency, the thresholds are dynamically warped based on the neural architecture:
* **Attention Tensors:** Hyper-sensitive to angular rotation. The Sieve enforces a strict 0.05% maximum energy loss, mostly forcing them to 6-bit or 8-bit.
* **MLP / Expert Tensors:** Highly robust to noise. The Sieve applies an aggressive leniency multiplier, allowing up to 5.0% error for Q2 and 2.0% for Q3. 

If a row can survive at 2 bits without causing structural brain damage, it stays at 2 bits. If it contains complex outlier geometry, the Sieve promotes it.

---

## 4. Strict VRAM Engineering

Compiling a 14GB LLM row-by-row on a consumer GPU normally results in catastrophic Out-Of-Memory (OOM) failures due to hidden 64-bit PyTorch allocations. 

The V28 Autoencoder is written with hyper-strict VRAM bulkhead engineering:
* **Micro-Batched 2D Sweeps:** Bypassing 2GB arithmetic allocation spikes by slicing row evaluation into 4096-width chunks.
* **Binary Search Targeting:** Replacing global boolean reduction matrices with `torch.searchsorted` to eliminate 4GB int64 accumulator bombs.
* **Strict In-Place Updating:** Eliminating `torch.where` allocations in favor of direct `torch.minimum(out=)` memory overwrites.

---

## 5. The Hard Numbers (Qwen 2.5 7B)

Unlike standard repositories, we publish the exact mathematical degradation to prove the structural coherence of our flat file sizes.

| Architecture | Model Size | WikiText-2 Perplexity | Degradation | Note |
| :--- | :--- | :--- | :--- | :--- |
| **Base (FP16)** | 14.0 GB | 6.1050 | - | Baseline intelligence |
| **V18 (simple VBR)** | 11.0 GB | 6.1316 | +0.0266 | The bloated failure |
| **V28 (Pareto VBR)** | **4.8 GB** | **6.4656** | **+0.3606** | **The Mathematical Floor** |

*Note: The 4.8 GB footprint is the strict, effective flat file size (~5.48 bpw) including all polynomial headers, scale vectors, and VBR byte maps. Zero group-wise bloat.*

---

## 6. How to Run the Compressor

### Step 1: Compile the Model (The Autoencoder)
Because VBR uses heavy continuous geometry, the compression is split into chunks to protect GPU memory. Run the Autoencoder script iteratively across your available GPUs:

```bash
python3 scripts/autoencoder_v25.py --chunk_idx 0 --gpu 0 --total_chunks 4
python3 scripts/autoencoder_v25.py --chunk_idx 1 --gpu 1 --total_chunks 4
python3 scripts/autoencoder_v25.py --chunk_idx 2 --gpu 2 --total_chunks 4
python3 scripts/autoencoder_v25.py --chunk_idx 3 --gpu 3 --total_chunks 4
```
*This will output `compressed_01.pt` through `compressed_04.pt` into your target directory.*

### Step 2: Verify the Intelligence (Perplexity)
Run the sliding-window WikiText-2 benchmark to dynamically unpack the VBR files back into the Hugging Face skeleton and measure the exact mathematical degradation:

```bash
python3 scripts/eval_perplexity_v27.py
```
