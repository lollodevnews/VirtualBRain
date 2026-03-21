# VirtualBrain (VBR)
**The Polynomial Quantization Engine: A Variable Bitrate (VBR) framework utilizing Continuous S-Curve Reparameterization.**

![VirtualBRain Architecture Diagram](diagram.png)

> ⚠️ **IMPORTANT: THE BUILDER'S SHIELD**
> There are many potentially good intuitions here that are being actively explored, but don't treat this as "the truth". This is a highly experimental work in progress and, like every construction site, there are plenty of exposed sharp edges that will hurt you if you are not careful. Enjoy with awareness.

---

## 🗺️ Repository Navigation

VirtualBrain is structured as a monorepo. Please navigate to the specific module you wish to explore:

* **[`📁 autoencoder/`](./autoencoder/)** — **[CURRENT STATE OF THE ART]** Contains the V27 Neural Compressor and the Python Inference Emulator. This is where the continuous $(a, b, m, n)$ polynomial math and the 7B compression pipeline live.
* **[`📁 theory/`](./Theory/)** — Contains the core physics philosophy. Explores how the Transformer maps to Quantum Superposition, Wave-Collapse (Decoherence), and zero-point energy, complete with a QPU Emulator script.
* **[`📁 qwen1.5_0.5b/`](./qwen1.5_0.5b/)** — **[ARCHIVE]** The historical "Phase 4" proof of concept. A rigid 5-bit grid implementation that first proved the viability of Signed-Magnitude VBR logic. 
* **`📁 engine_hip/`** — **[COMING SOON]** The bare-metal C++ AMD/ROCm Soft-FPGA inference kernel.

---

## 1. The Zero-Crutch Philosophy (Row-Wise vs. Group-Wise)

The open-source quantization community relies on a shared deception: **Group-Wise Scaling**. To make "4-bit" models (like standard AWQ or GGUF) retain their intelligence, they chop rows into tiny 64-weight blocks and inject gigabytes of hidden FP16 metadata (scales and zero-points) to prop up the math. Their "4-bit" models actually consume ~5.0 bits per weight in total file size.

**VirtualBrain VBR abandons group-wise scaling entirely.**
Instead of relying on hidden FP16 grids, VBR uses a custom Autoencoder to mathematically model the weight distribution of an *entire row* using a continuous polynomial S-Curve. One scale, one dust anchor, and one curve per row.

---

## 2. The Mathematical Formulation (V27)

We fit a 4-parameter polynomial curve ($a, b, m, n$) to hyper-focus the quantization bins around the matrix's dense core, while expanding the tails to naturally absorb violent outliers.

The physical curve is defined as:
$$f(x) = (1 - a - b)x + a(x^m) + b(x^n)$$

Where:
* **$x$** is the normalized weight magnitude.
* **$a$** is unbound ($-\infty$ to $+\infty$). It controls the amplitude and inversion of the middleweight plateau, allowing negative coefficients to flip the curve's concavity.
* **$b \in [0.0, 1.0]$** strictly controls the extreme tip outlier expansion.
* **$m \in [0.1, 4.0]$** controls the shape of the middleweights.
* **$n \in [1.0, 8.0]$** dictates the extreme outlier exponential wall.

By dynamically compiling this curve, the compressor creates a highly efficient binary layout ready for dynamic decompression, entirely bypassing iterative lookup tables.

---

## 3. The Pareto-Optimal VBR Sieve

The true breakthrough of V27 is the **Dynamic Signal-to-Noise (SNR) Sieve**. 
Instead of forcing a global bit-depth, the Sieve actively evaluates the Energy Loss (Relative Squared Error) of every single row during compilation and dynamically assigns it a bit-depth from 2-bit (Q2) to 8-bit (Q8).

To achieve true Pareto efficiency, the thresholds are dynamically warped based on the neural architecture:
* **Attention Tensors:** Hyper-sensitive to angular rotation. The Sieve enforces a strict $0.05\%$ maximum energy loss, mostly forcing them to 6-bit or 8-bit.
* **MLP / Expert Tensors:** Highly robust to noise. The Sieve applies an aggressive leniency multiplier, allowing up to $5.0\%$ error for Q2 and $2.0\%$ for Q3. 

If a row can survive at 2 bits without causing structural brain damage, it stays at 2 bits. If it contains complex outlier geometry, the Sieve promotes it.

---

## 4. The Hard Numbers (Qwen 2.5 7B)

Unlike standard repositories, we publish the exact mathematical degradation to prove the structural coherence of our flat file sizes.

| Architecture | Model Size | WikiText-2 Perplexity | Degradation | Note |
| :--- | :--- | :--- | :--- | :--- |
| **Base (FP16)** | 14.0 GB | 6.1050 | - | Baseline intelligence |
| **V27 (Pareto VBR)** | **4.8 GB** | **6.4656** | **+0.3606** |  **Compressed** |

*Note: The 4.8 GB footprint is the strict, effective flat file size (~5.48 bpw) including the dictionary, all polynomial headers, scale vectors, and VBR byte maps. Zero group-wise bloat.*

---

## Quick Start (V27 Pipeline)

Navigate to the `autoencoder/` directory to run the state-of-the-art framework.

**1. Compress the Model (VRAM Protected)**
```bash
cd autoencoder
python3 autoencoder_v27.py --chunk_idx 0 --gpu 0 --total_chunks 4
```

**2. Evaluate the Intelligence**
```bash
python3 eval_perplexity_v27.py
```

**3. Run Python Inference Emulator**
```bash
python3 v27_inference.py
```

---

## 🚀 The Roadmap & Future Scope

VirtualBrain is not just a quantizer; it is the foundation for a non-sequential, Turing-complete Neural CPU. Our active research pipeline includes:

* **Bare-Metal GPU Fusion (HIP/Triton):** Fusing the continuous polynomial S-Curve evaluation directly into the Matrix Multiplication SRAM steps to achieve native FP16 token throughput with a 5-bit memory footprint.
* **Mixed-Precision MoE Tournaments:** Dynamically assigning 1-bit to 4-bit divisors per-row for Mixture of Experts (like Mixtral), physically collapsing cold expert blocks while preserving high precision for chaotic logic hubs.
* **Neural Turing Execution (LISP Routing):** Transitioning from a sequential layer executor to a dynamic `while` loop, allowing the matrix to output a 32-bit integer pointer to physically address the next required expert matrix in VRAM.
* **Quantum Emulation:** Leveraging the VBR architecture's mapping to high-dimensional Hilbert spaces to execute logic gates that mimic quantum search algorithms on classical deterministic silicon (see **[`Theory/qpu_emulator`](./Theory/qpu_emulator.py)** ).
