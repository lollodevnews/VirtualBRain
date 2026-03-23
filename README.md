# VirtualBrain (VBR)
**The Non-Linear Quantization Engine: Modulating Noise, Not Bits.**

![VirtualBRain Architecture Diagram](diagram.png)

> ⚠️ **IMPORTANT: THE BUILDER'S SHIELD**
> There are many potentially good intuitions here that are being actively explored, but don't treat this as "the truth". This is a highly experimental work in progress and, like every construction site, there are plenty of exposed sharp edges that will hurt you if you are not careful. Enjoy with awareness.

---

## The Zero-Crutch Philosophy

The open-source quantization community relies on a shared deception: **Group-Wise Scaling**. To make standard "4-bit" models (like AWQ or GGUF) retain their intelligence, they chop rows into tiny 64-weight blocks and inject gigabytes of hidden FP16 metadata to prop up the math. 

**VirtualBrain VBR abandons group-wise scaling entirely.** Instead of forcing a global bit-depth and patching the damage with metadata, VBR utilizes a custom Autoencoder powered by a **Monte Carlo Alternating Grid Search**. It evaluates the physical weight distribution of an entire row and compresses it using a continuous, non-linear polynomial S-Curve.

### 1. Modulating Noise, Not Bits
VBR actively evaluates the Mean Squared Error (MSE) of every single row and dynamically assigns it a bit-depth purely based on its noise tolerance. 
* **Attention Tensors** are hyper-sensitive. The engine enforces a strict **0.05%** maximum energy loss threshold to protect context recall.
* **Expert / MLP Tensors** are robust. The engine applies a relaxed **1.0%** error allowance, crushing them down to save massive amounts of VRAM without sacrificing intelligence.

### 2. The Superblock Archive
Instead of scattered bit-planes, VBR packs its continuous variable-bitrate streams into perfectly aligned, contiguous memory **Superblocks**. This allows the bare-metal C++ kernel to execute wide, 16-byte vectorized loads (`float4`), instantly saturating the GPU's L1 cache and hitting ~23 Tokens/Sec on older AMD MI50 hardware without a single warp divergence.

---

## 🏆 The Hard Numbers (Qwen 2.5 7B)

We publish the exact mathematical degradation to prove the structural coherence of our flat file sizes. By using continuous polynomial curves instead of fixed group-wise grids, V34 achieves near-lossless intelligence compression. 

| Architecture | Total File Size (`ls -lh`) | Bits Per Weight | WikiText-2 Perplexity | Degradation | MI50 Inference Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Base (FP16)** | 14.0 GB | 16.0 bpw | ~6.1400 | - | - |
| **V34 (Grid Search)** | **4.9 GB** | **~5.60 bpw** | **6.2285** | **+0.0885** | **22.83 T/s** |

*Note: The 4.9 GB footprint is the strict, effective flat file size reported by the OS. It encompasses all compressed matrices, polynomial headers, scale vectors, and VBR byte maps. Zero group-wise bloat.*

---

## 🗺️ Repository Navigation

VirtualBrain is structured as a monorepo. Please navigate to the specific module you wish to explore:

* **[`📁 autoencoder/`](./autoencoder/)** — **[CURRENT STATE OF THE ART]** Contains the V34 Non-Linear Autoencoder and the Python Inference Emulator. **Read the sub-README here for the deep-dive into the $(a, b, m, n)$ polynomial math, the $k$ substitution, and the grid-search mechanics.**
* **[`📁 theory/`](./Theory/)** — Contains the core physics philosophy. Explores how the Transformer maps to Quantum Superposition, Wave-Collapse (Decoherence), and zero-point energy, complete with a QPU Emulator script.
* **[`📁 qwen1.5_0.5b/`](./qwen1.5_0.5b/)** — **[ARCHIVE]** The historical "Phase 4" proof of concept. A rigid 5-bit grid implementation that first proved the viability of Signed-Magnitude VBR logic. 
* **`📁 engine_hip/`** — **[WIP]** The bare-metal C++ AMD/ROCm Soft-FPGA inference kernel designed to natively ingest Superblocks.

---

## 🚀 The Roadmap & Future Scope

VirtualBrain is not just a quantizer; it is the foundation for a non-sequential, Turing-complete Neural CPU. Our active research pipeline includes:

* **Bare-Metal GPU Fusion (HIP/Triton):** Fusing the continuous polynomial S-Curve evaluation directly into the Matrix Multiplication SRAM steps to achieve native FP16 token throughput with an n-bit memory footprint.
* **Mixed-Precision MoE Tournaments:** Dynamically assigning 1-bit to 4-bit divisors per-row for Mixture of Experts (like Mixtral), physically collapsing cold expert blocks while preserving high precision for chaotic logic hubs.
* **Neural Turing Execution (LISP Routing):** Transitioning from a sequential layer executor to a dynamic `while` loop, allowing the matrix to output a 32-bit integer pointer to physically address the next required expert matrix in VRAM.
* **Quantum Emulation:** Leveraging the VBR architecture's mapping to high-dimensional Hilbert spaces to execute logic gates that mimic quantum search algorithms on classical deterministic silicon (see **[`Theory/qpu_emulator`](./Theory/qpu_emulator.py)**).
