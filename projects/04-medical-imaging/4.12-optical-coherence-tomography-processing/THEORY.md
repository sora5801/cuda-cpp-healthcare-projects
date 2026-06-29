# THEORY — 4.12 Optical Coherence Tomography Processing

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

### 4.12 Optical Coherence Tomography Processing 🟡 · Active R&D
- **Deep dive:** Spectral-domain OCT acquires spectra per A-scan (axial line); reconstruction requires dispersion compensation, interpolation from wavelength to wavenumber space, and 1D FFT per A-scan. A single B-scan of 2,048 A-scans × 2,048 spectral pixels requires 2,048 FFTs of length 2,048, easily parallelizable in GPU batches. Real-time 3D OCT volumes for surgical guidance require processing ~100 B-scans/second (~4 × 10⁸ FFT points/s), achievable only with GPU. Downstream retinal layer segmentation (8 boundaries, 3D graph search) and fluid detection (intra/subretinal, FLIO) add CNN inference workload; TensorRT-optimized U-Net achieves 3.5 ms/B-scan inference.
- **Key algorithms:** Spectral-domain FFT reconstruction, dispersion compensation, k-space resampling (NUFFT), GPU-batched FFT (cuFFT), 3D graph-cut layer segmentation, deep learning retinal layer segmentation (U-Net, U-NetRT), fluid detection CNN, Doppler OCT velocity mapping.
- **Datasets:** OCTDL (https://www.nature.com/articles/s41597-024-03182-7) — 2,064 labeled OCT B-scans; Duke DME OCT dataset (https://people.duke.edu/~sf59/Chiu_BOE_2012_dataset.htm) — 110 annotated volumes; OCTA-500 (https://arxiv.org/abs/2012.07261) — OCT angiography volumes with labels.
- **Starter repos/tools:** OCT-Marker (https://github.com/neurodial/OCT-Marker) — annotation tool for OCT B-scans; Iowa Reference Algorithms (https://www.iibi.uiowa.edu/content/shared-software-Iowa-reference-algorithms) — graph-based segmentation (verify URL); k-Wave CUDA (https://github.com/klepo/k-Wave-Fluid-CUDA) — relevant for photoacoustic OCT extensions; real-time OCT reconstruction demos available in NVIDIA cuFFT samples.
- **CUDA libraries & GPU pattern:** cuFFT batched 1D FFT (one FFT per A-scan, entire B-scan in one cuFFT call); custom CUDA kernel for dispersion phase correction; cuDNN for U-Net inference; CUDA streams for pipelining A-scan acquisition and reconstruction.

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
