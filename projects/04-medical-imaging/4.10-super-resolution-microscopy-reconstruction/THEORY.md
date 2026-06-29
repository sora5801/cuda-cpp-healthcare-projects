# THEORY — 4.10 Super-Resolution Microscopy Reconstruction

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

### 4.10 Super-Resolution Microscopy Reconstruction 🟡 · Active R&D
- **Deep dive:** STORM/PALM single-molecule localization microscopy (SMLM) acquires thousands of diffraction-limited frames; each fluorophore's sub-pixel position is estimated by fitting a 2D Gaussian PSF to sparse activated emitters. Processing 10⁴–10⁵ raw frames at 512×512 per acquisition demands massively parallel PSF fitting — each detected spot is independent, creating an embarrassingly parallel workload ideal for GPU. SRRF (Super-Resolution Radial Fluctuations) and SOFI (Second-Order Fluctuation Imaging) compute cross-correlations or cumulants over time stacks, with O(N·T) operations per pixel. Structured Illumination Microscopy (SIM) reconstruction requires per-orientation/phase FFT and OTF inversion, naturally parallelizable across k-space.
- **Key algorithms:** Gaussian/PSF maximum-likelihood fitting (SMLM), SRRF (radial fluctuation analysis), SOFI (cumulant imaging), SIM reconstruction (OTF inversion, Wiener filter), deconvolution (Richardson-Lucy + GPU), DECODE (deep probabilistic SMLM localization).
- **Datasets:** EPFL SMLM Challenge dataset (https://srm.epfl.ch/srm/dataset/challenge-2016/) — synthetic and real STORM/PALM frames; BioImage Archive SMLM collections (https://www.ebi.ac.uk/biostudies/bioimages); OpenMicroscopy Environment (OME-TIFF standard).
- **Starter repos/tools:** DECODE (https://github.com/TuragaLab/DECODE) — deep learning GPU SMLM localizer, orders of magnitude faster than MLE; ThunderSTORM (FIJI plugin, GPU-optional); NanoJ-SRRF (https://github.com/HenriquesLab/NanoJ-SRRF) — GPU-accelerated SRRF in ImageJ; fairSIM (https://github.com/fairSIM/fairSIM) — GPU-enabled SIM reconstruction.
- **CUDA libraries & GPU pattern:** Custom CUDA kernels for per-emitter Gaussian fitting (one warp per candidate emitter); cuFFT for SIM phase/OTF; shared memory for 7×7 PSF patch fitting; atomic additions for localization histogram accumulation.

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
