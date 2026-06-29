# THEORY — 8.16 Neural Signal Compression & Wireless BCI Transmission

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. Diagrams in Mermaid/ASCII
> are welcome. See [README.md](README.md) for the quick tour and build steps.
>
> _Educational only — not for clinical use._

<!-- =======================================================================
     The block below is the verbatim catalog deep-dive for this project,
     stamped in by scaffold.py as raw material. Use it to write the sections
     that follow, then DELETE it (or fold it into "The science"). Every
     TODO(theory) below must be completed before the project is "done".
     ======================================================================= -->

<details>
<summary>Catalog deep-dive (raw source material — fold into the sections below, then remove)</summary>

### 8.16 Neural Signal Compression & Wireless BCI Transmission 🟡 · Active R&D
- **Deep dive:** Fully implanted high-channel-count BCIs (1 024–65 000 electrodes in emerging platforms) cannot transmit raw 30 kHz × N-channel data wirelessly due to power/bandwidth limits. GPU-accelerated on-device compression (threshold crossing, wavelet compression, PCA projection, spike detection) must reduce data 100–1 000× before wireless transmission. Implantable ASICs perform this in hardware, but GPU simulation of compression algorithms enables algorithm design and fidelity evaluation before silicon tape-out.
- **Key algorithms:** Threshold-based spike detection, wavelet packet decomposition (WPD), compressed sensing (L1 minimization / OMP), PCA projection for dimensionality reduction, delta-encoding, Huffman/arithmetic coding, matched filter spike detection, signal reconstruction via iterative thresholding (ISTA/FISTA).
- **Datasets:** DANDI Neuropixels recordings (https://dandiarchive.org); BrainGate implanted array datasets (https://www.braingate.org); SpikeInterface benchmark recordings (https://spikeinterface.readthedocs.io); PhysioNet neural datasets (https://physionet.org).
- **Starter repos/tools:** BrainFlow (https://github.com/brainflow-dev/brainflow) — real-time neural signal SDK; SpikeInterface (https://github.com/SpikeInterface/spikeinterface) — spike detection and feature extraction pipeline; PyWavelets (https://github.com/PyWavelets/pywt) — wavelet decomposition with CuPy GPU backend; FISTA implementations in PyTorch (verify URL — numerous public repos).
- **CUDA libraries & GPU pattern:** cuFFT for wavelet and frequency-domain feature extraction; cuBLAS for PCA projection (matrix-vector multiply); CUDA Thrust for threshold scan across all channels; pattern: streaming pipeline—raw samples in via DMA, CUDA kernels for detection and projection, compressed output via pinned host memory ring buffer.

</details>

---

## 1. The science

TODO(theory): The biology / medicine / physics being modeled — enough for a
reader to understand the *problem* before any math. What real-world question
does computing this answer?

## 2. The math

TODO(theory): The governing equations / formal problem statement, with **every
symbol defined** (units, ranges). State inputs, outputs, and the objective.

## 3. The algorithm

TODO(theory): Step-by-step. Include **complexity analysis**: serial cost vs. the
parallel work/depth. Where is the arithmetic intensity? What is the data-access
pattern?

## 4. The GPU mapping

TODO(theory): How the algorithm becomes **threads / blocks / grids**.
- Thread-to-data mapping (which thread owns which element).
- Launch configuration and the reasoning (block size, grid size).
- Memory hierarchy used and **why**: global / shared / registers / constant /
  texture. Where is the bandwidth bottleneck? What is the occupancy story?
- Which CUDA library (cuBLAS / cuFFT / cuRAND / cuSOLVER / Thrust) does what,
  and what it would take to write that step by hand (no black boxes — §6.1.6).

```
TODO(theory): an ASCII or Mermaid diagram of the grid/block decomposition.
```

## 5. Numerical considerations

TODO(theory): Precision (FP32 vs FP64) and why. Stability. Race conditions and
whether atomics are used. **Determinism**: does the parallel reduction reorder
floating-point sums? If so, say so and quantify the caveat.

## 6. How we verify correctness

TODO(theory): The CPU reference (`src/reference_cpu.cpp`), the **tolerance** and
why that value, and the edge cases checked. Explain why agreement between an
independent serial implementation and the GPU implementation is convincing
evidence of correctness.

## 7. Where this sits in the real world

TODO(theory): How production tools (named in the catalog "Prior art") do this
differently — what they add (scale, accuracy, features) that this teaching
version omits. If this is a 🔴 frontier project shipped as a reduced-scope
teaching version, describe the full approach here.

---

## References

TODO(theory): Papers, docs, and the starter repos from the catalog, with one
line each on what to learn from them.
