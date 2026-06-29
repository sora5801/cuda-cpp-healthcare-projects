# THEORY — 4.3 MRI Reconstruction with Compressed Sensing

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

### 4.3 MRI Reconstruction with Compressed Sensing 🟡 · Active R&D
- **Deep dive:** MRI acquires k-space (Fourier-domain) samples; compressed sensing (CS) reconstructs images from highly under-sampled k-space using sparsity priors (wavelet, total variation), enabling 4–8× scan acceleration. The core computation is a sequence of non-uniform FFTs (NUFFT/NFFT) for arbitrary k-space trajectories, followed by iterative soft-thresholding or proximal operators. NUFFT on a 3D grid at clinical resolution (~256³) involves ~10⁹ operations per iteration; GPU parallelism reduces each NUFFT to milliseconds vs. seconds on CPU, enabling real-time feedback. Multi-channel parallel imaging (SENSE, GRAPPA, PICS) adds per-coil FFTs (~32 channels), multiplying the compute by the coil count and making GPU essential.
- **Key algorithms:** SENSE, GRAPPA, non-uniform FFT (NUFFT/NFFT3), PICS (Parallel Imaging + CS), Split-Bregman / ADMM, FISTA, total variation, wavelet sparsity, k-t SENSE for dynamic MRI.
- **Datasets:** fastMRI (NYU/Facebook, https://fastmri.med.nyu.edu/ and https://github.com/facebookresearch/fastMRI) — 1,500+ knee and 6,970+ brain raw k-space MRI scans; Calgary-Campinas-359 — multi-channel brain MRI k-space (https://sites.google.com/view/calgary-campinas-dataset/); SKM-TEA (Stanford knee MRI, https://github.com/StanfordMIMI/skm-tea).
- **Starter repos/tools:** BART (Berkeley Advanced Reconstruction Toolbox, https://github.com/mrirecon/bart) — production CS-MRI tool, GPU-accelerated PICS, SENSE, NUFFT; SigPy (https://github.com/mikgroup/sigpy) — Python GPU (CuPy) MRI signal-processing and NUFFT; MIRT (Michigan Image Reconstruction Toolbox, https://github.com/JeffFessler/MIRT.jl) — Julia/MATLAB iterative reconstruction with NUFFT; PyNUFFT (https://github.com/jyhmiinlin/pynufft) — Python NUFFT with CUDA/OpenCL backends.
- **CUDA libraries & GPU pattern:** cuFFT for gridded FFT; custom CUDA NUFFT gridding kernels; cuBLAS for coil combination; per-coil FFT parallelized across CUDA streams; shared memory for gridding accumulation.

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
