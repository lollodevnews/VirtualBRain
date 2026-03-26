# VirtualBrain VBR: The Algebraic CDF Quantizer (V35)
**A Variable BitRate (VBR) bare-metal quantization framework.**

**THE EVOLUTION OF THIS PROJECT:** The journey started by wanting to move away from brute-force algorithms, leading us to simple neural networks and fixed anchor points. In V34, we moved to a Monte Carlo simulation but were bottlenecked by the extreme VRAM overhead of 3D mathematical grids and Mean Squared Error (MSE) evaluations. 

With **V35**, we have completely shattered the Pareto frontier. By abandoning MSE in favor of a **Normalized L1 Energy Metric**, and replacing 3D tensor grids with a **Zero-Memory Algebraic CDF Shortcut**, V35 evaluates millions of non-linear Voronoi thresholds in a fraction of a millisecond. The result is an ultra-dense, mathematically pure compression engine that operates at the physical speed limit of the GPU.

---

## 1. The Zero-Crutch Philosophy (Row-Wise vs. Group-Wise)

The open-source quantization community relies on a shared deception: **Group-Wise Scaling**. To make standard 4-bit models (like AWQ or GGUF) retain their intelligence, they chop rows into tiny 64-weight blocks and inject gigabytes of hidden FP16 metadata (scales and zero-points) to prop up the math. 

**VirtualBrain VBR abandons group-wise scaling entirely.**
Instead of relying on hidden FP16 grids, VBR uses a custom Autoencoder to mathematically model the weight distribution of an *entire row* using a Continuous Linkage Topology. One scale and one continuous curve per row. Zero metadata bloat.

---

## 2. The Mathematical Formulation: V35 Topology

In V34, we fought against complex non-linear bounds and k-substitutions to keep the polynomial curves stable. V35 introduces a perfectly stable, 3-parameter Desmos topology (a, c, m) that dynamically warps quantization bins without catastrophic algorithmic collapse.

**Interactive Desmos Topology Graph:** [Play with the V35 Curve Here](https://www.desmos.com/calculator/jwadm38ufo)

The physical continuous curve is defined as:
y = ((1 - a)x + a * x^m)^c

**Surviving the Roller Coaster Loop:**
Because the 'a' parameter is allowed to swing negative, the mathematical derivative of this curve can violently invert, causing the physical bins to loop backward on themselves. Traditional algorithms (like binary search trees) panic when given non-monotonic arrays. V35 utilizes a brute-force hardware sweep that calculates the absolute physical distance to every bin independently. Even if the curve loops backward, V35 mathematically guarantees the absolute closest Voronoi assignment for all 14,336 weights in a row simultaneously.

---

## 3. The Prefix-Sum Breakthrough (Zero-Memory CDF)

Evaluating 4,000 geometric realities across 58 million weights traditionally requires hundreds of gigabytes of VRAM. 

V35 bypasses the memory wall entirely by using a **Cumulative Distribution Function (CDF)**.
Instead of measuring the distance for every single weight individually, the engine:
1. Sorts the original weights and calculates their continuous Prefix Sum once.
2. Identifies the halfway threshold boundaries between the warped quantization bins.
3. Uses a single algebraic equation to extract the exact sub-pixel error directly from the CDF.

This drops the mathematical complexity from O(N * K) to essentially O(1) per chunk, executing entirely within the GPU's L1/L2 cache and eliminating the 8GB OOM crashes of previous versions.

---

## 4. Modulating Noise, Not Bits (L1 Energy Routing)

Earlier versions of VBR relied on Mean Squared Error (MSE). However, MSE inherently squares fractions, creating an optical illusion that heavily penalizes outliers while ignoring the physical "mass" of the matrix.

**V35 upgrades to a Normalized L1 Energy Metric.**
The VBR Sieve now measures the exact sum of the Y-axis divergences and divides it by the total absolute mass of the row. This transforms the threshold into a pure **Signal-to-Noise Ratio (SNR)**. 
* If the Sieve is given a `0.125` target, it searches the Monte Carlo grid for the lowest possible bit-depth that perfectly preserves **87.5% of the row's physical mass**.
* Massive, noise-resilient Expert/FFN layers are intelligently crushed down to 4-bit and 5-bit arrays, while highly sensitive Attention layers naturally fail the strict L1 checks and retain higher bitrates.

---

## 5. Fused SWAR Bit-Packing

V34 relied on extracting the sign bit into a separate bit-plane, which added unnecessary indexing overhead.

**V35 mathematically fuses the Sign Bit directly into the Most Significant Bit (MSB).**
By offsetting negative assignments by K_bins during compilation, a 4-bit array naturally contains positive values in bins 0-7 and negative values in bins 8-15. This allows the SWAR (SIMD Within A Register) bit-packer to slice continuous data streams flawlessly. The GPU extracts D bits exactly D times without any manual sign-reconstruction logic, maximizing memory bandwidth saturation.

---

## 6. The Hard Numbers (Qwen 2.5 7B)

Unlike standard repositories, we publish the exact mathematical degradation to prove the structural coherence of our flat file sizes. Benchmarked on an AMD Instinct MI50.

| Architecture | Total File Size | Bits Per Weight | WikiText-2 Perplexity | Degradation |
| :--- | :--- | :--- | :--- | :--- |
| **Base (FP16)** | ~14.0 GB | 16.0 bpw | 6.1050 | - |
| **V28 (AdamW)** | 4.80 GB | ~5.48 bpw | 6.4656 | +0.3606 |
| **V34 (Grid Search)** | 4.90 GB | ~5.60 bpw | 6.2285 | +0.1235 |
| **V35 (High Fidelity)** | **4.10 GB** | **~4.60 bpw** | **6.1752** | **+0.0702** |
| **V35 (Extreme VBR)** | **3.3 GB** | **~3.90 bpw** | **6.4151** | **+0.3101** |

*Note: The footprints reported above encompass all compressed matrices, polynomial headers, scale vectors, and VBR byte maps. Zero group-wise bloat.*

---

## 7. How to Run the Framework

### Step 1: Compile the Model (The Autoencoder)
Because VBR evaluates massive mathematical grids, the compression is split into chunks to protect GPU memory. Run the Autoencoder iteratively across your available GPUs:

```bash
HIP_VISIBLE_DEVICES=0 python3 autoencoder.py --chunk_idx 0 --total_chunks 4
HIP_VISIBLE_DEVICES=1 python3 autoencoder.py --chunk_idx 1 --total_chunks 4
# ... repeat for all chunks
```

### Step 2: Inference & Verification
Run the inference.py script (or your custom bare-metal HIP engine) to verify continuous generation, or launch the sliding-window Perplexity benchmark to measure the exact mathematical degradation:

```bash
# Test generation and Tokens/Sec speed
python3 inference.py

# Evaluate WikiText-2 Perplexity
python3 perplexity.py
```

## Execution Diagram

```mermaid
graph TD
    subgraph Phase1 [Phase 1: FP16 Ingestion]
        A[Raw FP16 Weights] --> B[Row-Wise Extraction]
        B --> C[Calculate Row Scale and Normalize to 0.0 - 1.0]
    end

    subgraph Phase2 [Phase 2: The V35 Autoencoder]
        C --> D[Sort Weights and Build Zero-Memory CDF]
        D --> E[Monte Carlo Grid Search: Desmos a, c, m]
        E --> F[Algebraic L1 Energy Evaluation]
        F --> G{The Pareto Sieve}
        G -- Robust FFN/MLP --> H[Assign 3-bit or 4-bit]
        G -- Sensitive Attention --> I[Assign 5-bit or 6-bit]
    end

    subgraph Phase3 [Phase 3: Fused SWAR Packing]
        H --> J[Map to Closest Voronoi Bins]
        I --> J
        J --> K[Fuse Sign Bit into MSB]
        K --> L[Pack into Contiguous VBR Superblocks]
        L --> M[(V35 Compressed .pt Archive)]
    end

    subgraph Phase4 [Phase 4: Native Inference Reconstruction]
        M --> N[Vectorized SWAR Unpack]
        N --> O[Evaluate Desmos Topology Curve]
        O --> P[Apply Row Scale and Fused Sign]
        P --> Q[Reconstructed FP16 Matrix]
        Q --> R[Hardware MatMul and Text Generation]
    end

    style A fill:#1e1e1e,stroke:#fff,stroke-width:2px,color:#fff
    style M fill:#8b0000,stroke:#fff,stroke-width:2px,color:#fff
    style Q fill:#004d00,stroke:#fff,stroke-width:2px,color:#fff
    style G fill:#b8860b,stroke:#fff,stroke-width:2px,color:#fff
```
