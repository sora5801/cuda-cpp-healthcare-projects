# THEORY — 4.29 Light-Sheet Microscopy Reconstruction

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

### 4.29 Light-Sheet Microscopy Reconstruction 🟡 · Active R&D
- **Deep dive:** Light-sheet fluorescence microscopy (LSFM / selective plane illumination, SPIM) acquires terabyte-scale datasets of developing embryos or cleared organs by illuminating a thin optical plane; the resulting multi-view 3D stacks must be: (1) registered across views/illuminations, (2) fused via multi-view deconvolution, and (3) stitched from tiled acquisitions. Multi-view deconvolution (Richardson-Lucy per view, Gaussian PSF model) on a 10³ × 10³ × 10³ sub-volume requires ~10¹² multiply-accumulates per outer iteration — GPU essential. BigStitcher (Fiji/ImageJ) uses GPU-accelerated image correlation for tile alignment and multi-GPU deconvolution for simultaneous multi-view fusion.
- **Key algorithms:** Multi-view Richardson-Lucy deconvolution (GPU), entropy-based content-weighted fusion, phase correlation tile stitching, BigStitcher alignment, iterative PSF estimation (blind deconvolution), SPIM dual-illumination fusion, 4D cell tracking (convolutional tracker).
- **Datasets:** OpenOrganelle (https://openorganelle.janelia.org/) — FIB-SEM and light-sheet neuroscience; EMBL LSFM public datasets (https://www.embl.org/); Zebrafish SPIM atlas data from Nature Methods papers; BioImage Archive LSFM collections (https://www.ebi.ac.uk/biostudies/bioimages).
- **Starter repos/tools:** BigStitcher (https://github.com/PreibischLab/BigStitcher) — GPU-accelerated LSFM stitching/fusion; CSBDeep/CARE (https://github.com/CSBDeep/CSBDeep) — deep learning LSFM denoising/restoration; N2V (https://github.com/juglab/n2v) — self-supervised GPU denoising for LSFM; DeconvolutionLab2 (https://github.com/Biomedical-Imaging-Group/DeconvolutionLab2) — multi-algorithm deconvolution with GPU.
- **CUDA libraries & GPU pattern:** cuFFT for Fourier-domain deconvolution (Richardson-Lucy in k-space); cuBLAS for view-weight matrix products; custom CUDA for phase-correlation peak detection; multi-GPU domain decomposition across z-planes for large volumes; pinned host memory for streaming TB-scale data.

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
