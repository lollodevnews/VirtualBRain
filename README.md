# VirtualBrain (VBR)
**The Non-Linear Quantization Engine: Modulating Noise, Not Bits (V35).**

![VirtualBRain Architecture Diagram](diagram.png)

> ⚠️ **IMPORTANT: THE BUILDER'S SHIELD**
> There are many potentially good intuitions here that are being actively explored, but don't treat this as "the truth". This is a highly experimental work in progress and, like every construction site, there are plenty of exposed sharp edges that will hurt you if you are not careful. Enjoy with awareness.

---

## The Zero-Crutch Philosophy

The open-source quantization community relies on a shared deception: **Group-Wise Scaling**. To make standard "4-bit" models (like AWQ or GGUF) retain their intelligence, they chop rows into tiny 64-weight blocks and inject gigabytes of hidden FP16 metadata to prop up the math. 

**VirtualBrain VBR abandons group-wise scaling entirely.** Instead of forcing a global bit-depth and patching the damage with metadata, VBR utilizes a custom Autoencoder powered by a **Zero-Memory Algebraic CDF Shortcut**. It evaluates the physical weight distribution of an entire row and compresses it using a continuous, non-linear Desmos Topology (a, c, m).

### 1. Modulating Noise, Not Bits (L1 Energy Routing)
VBR actively evaluates the **Normalized L1 Energy** (the pure physical mass) of every single row and dynamically assigns it a bit-depth purely based on its noise tolerance, completely eliminating Mean Squared Error (MSE) illusions. 
* **Attention Tensors** are hyper-sensitive. The engine enforces a strict **~4%** maximum L1 energy shift to protect context recall, naturally retaining higher bitrates.
* **Expert / MLP Tensors** are massive but robust. The engine applies a relaxed **~8% to 12.5%** L1 allowance, crushing them down to 4-bit and 5-bit arrays to save massive amounts of VRAM without sacrificing intelligence.

### 2. Fused SWAR & The Superblock Archive
Instead of scattered bit-planes, VBR mathematically fuses the Sign Bit directly into the Most Significant Bit (MSB) during compilation. It then packs its continuous variable-bitrate streams into perfectly aligned, contiguous memory **Superblocks**. This allows bare-metal C++ and vectorized Python kernels to execute wide, zero-waste loads, instantly saturating the GPU's memory bandwidth without a single warp divergence.

---

## 🏆 The Hard Numbers (Qwen 2.5 7B)

We publish the exact mathematical degradation to prove the structural coherence of our flat file sizes. By using continuous polynomial curves and exact Voronoi thresholds instead of fixed group-wise grids, V35 achieves near-lossless intelligence compression. 

| Architecture | Total File Size (`ls -lh`) | Bits Per Weight | WikiText-2 Perplexity | Degradation |
| :--- | :--- | :--- | :--- | :--- |
| **Base (FP16)** | 14.0 GB | 16.00 bpw | 6.1050 | - |
| **V35 (High Fidelity)**| **4.00 GB** | **~4.57 bpw** | **6.2631** | **+0.1581** |
| **V35 (Extreme VBR)** | **3.56 GB** | **~4.06 bpw** | **6.4080** | **+0.3030** |

*Note: The footprints reported above represent the strict, effective flat file size on disk. They encompass all compressed matrices, polynomial headers, scale vectors, and VBR byte maps. Zero group-wise bloat.*

---

## 🗺️ Repository Navigation

VirtualBrain is structured as a monorepo. Please navigate to the specific module you wish to explore:

* **[`📁 autoencoder/`](./autoencoder/)** — **[CURRENT STATE OF THE ART]** Contains the V35 Algebraic CDF Autoencoder and the Python Inference Emulator. **Read the sub-README here for the deep-dive into the Desmos Topology (a, c, m), the Prefix-Sum shortcut, and the L1 Energy routing mechanics.**
* **[`📁 theory/`](./Theory/)** — Contains the core physics philosophy. Explores how the Transformer maps to Quantum Superposition, Wave-Collapse (Decoherence), and zero-point energy, complete with a QPU Emulator script.
* **[`📁 qwen1.5_0.5b/`](./qwen1.5_0.5b/)** — **[ARCHIVE]** The historical "Phase 4" proof of concept. A rigid 5-bit grid implementation that first proved the viability of Signed-Magnitude VBR logic. 
* **`📁 engine_hip/`** — **[WIP]** The bare-metal C++ AMD/ROCm Soft-FPGA inference kernel designed to natively ingest Superblocks.

---

## 🚀 The Roadmap & Future Scope

VirtualBrain is not just a quantizer; it is the foundation for a non-sequential, Turing-complete Neural CPU. Our active research pipeline includes:

* **Bare-Metal GPU Fusion (HIP/Triton):** Fusing the continuous polynomial curve evaluation directly into the Matrix Multiplication SRAM steps to achieve native FP16 token throughput with an n-bit memory footprint.
* **Mixed-Precision MoE Tournaments:** Dynamically assigning 1-bit to 4-bit divisors per-row for Mixture of Experts (like Mixtral), physically collapsing cold expert blocks while preserving high precision for chaotic logic hubs.
* **Neural Turing Execution (LISP Routing):** Transitioning from a sequential layer executor to a dynamic `while` loop, allowing the matrix to output a 32-bit integer pointer to physically address the next required expert matrix in VRAM.
* **Quantum Emulation:** Leveraging the VBR architecture's mapping to high-dimensional Hilbert spaces to execute logic gates that mimic quantum search algorithms on classical deterministic silicon (see **[`Theory/qpu_emulator`](./Theory/qpu_emulator.py)**).
