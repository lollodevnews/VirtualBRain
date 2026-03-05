# VirtualBRain (VBR)
**A LISP-style virtual machine for LLM brains.**

Standard Large Language Model (LLM) inference treats neural networks as legacy Complex Instruction Set Computers (CISC). They use massive, monolithic FP16 matrices to simultaneously calculate boolean routing logic, nuanced perplexity, and analog amplitude in a single, bandwidth-heavy pass. 

**VirtualBRain (VBR)** is a fundamentally different approach to quantization and model execution. It is a **Universal Inference Emulator** that decompiles dense neural networks into a Reduced Instruction Set (RISC) architecture. It physically decouples the network's boolean logic gates from its analog signal amplitude. 

By treating the neural network as a Digital Signal Processor (DSP), VirtualBRain rebuilds the 1970s LISP Machine for modern GPUs: executing ultra-fast, variable-bitrate integer routing masks anchored to stable floating-point baselines.



## Core Architectural Concepts

### 1. The Dust Anchor (Negative Perplexity)
Standard quantization relies on absolute minimums and maximums, making the math highly vulnerable to freak outliers. VirtualBRain introduces the **Dust Anchor**. 

During the packing phase, the engine isolates the lowest **5%** of weights by magnitude in a given column and averages them. This calculates the true gravitational center of the network's "No"—the dense accumulation of near-zero weights that form the structural boundaries of a concept. This anchor is stored in full FP16 precision and acts as a baseline resistor.

### 2. Decoupled Logic and Amplitude
VirtualBRain destroys the FP16 monolith by breaking the forward pass into two discrete physical components:

$$W_{approx} = (W_{logic\_gate} \times R_{computation}) + M_{baseline}$$

* **The Logic Gate (Variable-Bitrate Integer Planes):** Pure structural routing. Determines if a signal passes and at what relative intensity (1-bit to 4-bit).
* **The Resistors ($R$ and $M$ in FP16):** The analog amplifiers. They apply the maximum scaling range ($R$) and the Dust Anchor baseline ($M$) to the signals that survive the logic gates.



### 3. The Entropy Sieve (Bit-Plane Thresholding)
Instead of relying on computationally heavy cosine similarity to determine bit-depth, VirtualBRain uses Information Theory. The model is first quantized to a pristine 4-bit baseline. The packer then analyzes the individual bit-planes (from the Most Significant Bit down to the Least Significant Bit). 

If a lower bit-plane across a column possesses an activity ratio (entropy) below a defined threshold, it is deemed structural noise and physically dropped. The micro-energy of the dropped bits is absorbed into the FP16 Resistor, resulting in a mathematically clean Variable-Bitrate matrix without perplexity collapse.

### 4. Signed / Unsigned Column Flags
To eliminate dynamic range dead-space, VirtualBRain calculates a 1-bit global flag per column.
* **Unsigned:** The feature detector only flows in one direction from the dust. The integer bins are packed tightly from the anchor to the extreme.
* **Signed:** The feature detector crosses zero. The integer bins are anchored at the center dust and scale symmetrically outward.

## Current State & Usage

The current v1.0 release establishes the physical **4-bit VBR baseline** and the LISP machine execution graph. It physically packs two 4-bit weights per byte, cutting legacy FP16 model sizes in half while proving the structural integrity of the Dust Anchor theory.

* `importer.py`: The Decompiler. Ingests standard Hugging Face models and outputs the decoupled RISC components.
* `VirtualBRainEngine.py`: The Virtual Machine. Surgically injects the emulator into the model's architecture to execute the packed logic on the fly.

## Future Scope

* **Deep 1-Bit/2-Bit Physical Packing:** Upgrading the byte-packer to physically cram 4 to 8 weights into a single byte based on the Entropy Sieve's map, pushing the model toward ~2.0 BPW.
* **Dynamic Perplexity:** Exploring biological "cognitive dimming"—allowing the emulator to dynamically drop from 4-bit deep thinking to 1-bit instinctual routing depending on prompt complexity.

### Neural Turing Execution (Dynamic LISP Routing)
VirtualBRain v2.0 will transition the engine from a sequential layer executor into a non-sequential, Turing-complete Neural CPU. The architecture will physically partition the matrix output vector:
* **The Payload:** The upper dimensions carrying the VBR-processed semantic features.
* **The Address Register:** The bottom columns dynamically calculating a 32-bit integer pointer.

Instead of a static forward pass, the VirtualBRain Engine will execute a dynamic `while` loop. If the matrix outputs a positive address (`EVAL`), the engine routes the payload to the specific VRAM location of the next required expert matrix. If the matrix outputs a `0` address (`APPLY`), the cognitive loop halts, and the token is generated immediately. This will enable infinite-depth, dynamic-routing logic directly at the matrix level.



### Universal Turing Completeness & Variable-Dimension Function Blocks
VirtualBRain inherently achieves Turing Completeness without requiring hidden memory dimensions:
* **The KV Tape & Recursion:** The LLM's standard Context Window (KV Cache) acts as the infinite Turing Tape. Furthermore, the architecture supports Church's Lambda Recursion: a matrix can output its *own* address, spinning up a recursive `while` loop to output intermediate "scratchpad" tokens to the tape. The matrix reads its own output, processes the next step, and only fires the `APPLY` (0) pointer when the mathematical entropy collapses.
* **Escaping the Residual Highway:** To prevent the 4,096-dimensional bottleneck of standard Transformer architecture, VirtualBRain will merge the `down_proj` into the function block. This allows the LISP machine to chain logic gates dynamically in high-dimensional spectral space for as long as needed, only compressing back to the residual highway to write to the tape once the concept is fully resolved.
