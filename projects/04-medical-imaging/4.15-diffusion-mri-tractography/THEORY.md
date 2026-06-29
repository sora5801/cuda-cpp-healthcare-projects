# THEORY — 4.15 Diffusion MRI & Tractography

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

### 4.15 Diffusion MRI & Tractography 🟡 · Active R&D
- **Deep dive:** Diffusion MRI models water diffusion anisotropy in tissue to map white-matter fiber orientations. Fitting diffusion models (DTI, DKI, NODDI) per voxel is trivially parallel — each voxel is independent — and for a 2 mm isotropic brain (~10⁵ voxels × 100 diffusion directions), batch GPU fitting is 50–100× faster than serial CPU. Constrained spherical deconvolution (CSD) solves a per-voxel fiber orientation distribution function (fODF), requiring spherical harmonic decomposition (cuBLAS) at each voxel. Probabilistic tractography (particle filtering, iFOD2) samples millions of streamlines simultaneously, with each streamline step requiring trilinear interpolation of the fODF field — massively parallel across streamlines on GPU. BEDPOSTX GPU accelerates Markov chain Monte Carlo fiber model fitting by 200× vs. CPU.
- **Key algorithms:** DTI (diffusion tensor imaging), NODDI (neurite orientation dispersion), constrained spherical deconvolution (CSD), iFOD2 probabilistic tractography, SIFT/SIFT2 streamline filtering, multi-tissue CSD, particle filtering tractography, deep learning tractography (TractSeg).
- **Datasets:** Human Connectome Project (HCP) — 1,200 subjects, 3T/7T multi-shell dMRI (https://db.humanconnectome.org/); ABCD Study dMRI (https://abcdstudy.org/); UK Biobank dMRI (https://www.ukbiobank.ac.uk/); TMS-EEG Tractography Contest (verify URL).
- **Starter repos/tools:** MRtrix3 (https://github.com/MRtrix3/mrtrix3) — gold-standard CSD, iFOD2, SIFT2, GPU-accelerated deconvolution; FSL BEDPOSTX GPU (https://fsl.fmrib.ox.ac.uk/) — GPU Bayesian fiber orientation estimation (200× speedup); TractSeg (https://github.com/MIC-DKFZ/TractSeg) — direct CNN white-matter tract segmentation; DIPY (https://github.com/dipy/dipy) — Python dMRI analysis with GPU-compatible operations.
- **CUDA libraries & GPU pattern:** cuBLAS for spherical harmonic matrix products (CSD); custom CUDA kernel for per-voxel DTI tensor fitting (SVD); CUDA random number generation (cuRAND) for probabilistic streamline sampling; texture memory for fODF field interpolation during tractography.

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
