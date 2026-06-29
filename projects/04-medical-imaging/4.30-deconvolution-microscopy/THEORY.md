# THEORY — 4.30 Deconvolution Microscopy

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

### 4.30 Deconvolution Microscopy 🟢 · Established
- **Deep dive:** Wide-field and confocal fluorescence microscopes suffer from out-of-focus blur described by the point spread function (PSF); iterative deconvolution (Richardson-Lucy, Landweber) sharpens images by deblurring via the known or estimated PSF. Each R-L iteration requires two 3D FFT-based convolutions (forward: image×PSF; backward: ratio×PSF_flipped) on a volume as large as 2,048³; GPU cuFFT reduces each convolution from minutes to seconds. Blind deconvolution jointly estimates the PSF, requiring a second optimization variable and more iterations. Commercial instruments (Zeiss, Leica) offer GPU-accelerated deconvolution; open-source tools (CSBDeep, DeconvolutionLab2) provide GPU implementations for research.
- **Key algorithms:** Richardson-Lucy (RL) deconvolution, accelerated RL with TV regularization, Wiener deconvolution, total-variation deconvolution, blind PSF estimation (EM-algorithm), 3D FFT-based convolution via cuFFT, PSF measurement (bead-based calibration), CARE (content-aware image restoration).
- **Datasets:** BioImage Archive fluorescence microscopy datasets (https://www.ebi.ac.uk/biostudies/bioimages); EPFL Biomedical Imaging Group benchmark datasets (https://bigwww.epfl.ch/deconvolution/); ImageJ/Fiji sample datasets (https://imagej.net/); COBA microscopy benchmark.
- **Starter repos/tools:** CSBDeep/CARE (https://github.com/CSBDeep/CSBDeep) — GPU-accelerated content-aware restoration network; DeconvolutionLab2 (https://github.com/Biomedical-Imaging-Group/DeconvolutionLab2) — multi-algorithm with GPU; FlowDec (https://github.com/hammerlab/flowdec) — TF-based GPU deconvolution; N2V (https://github.com/juglab/n2v) — self-supervised denoising prior to deconvolution.
- **CUDA libraries & GPU pattern:** cuFFT 3D in-place FFT for PSF convolution; custom CUDA kernel for R-L multiplicative update (element-wise ratio); texture memory for PSF; batched cuFFT for simultaneous channel deconvolution; pinned memory for streaming large microscopy volumes.

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
