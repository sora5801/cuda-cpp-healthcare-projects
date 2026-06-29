# THEORY — 4.16 Functional MRI Analysis

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

### 4.16 Functional MRI Analysis 🟡 · Active R&D
- **Deep dive:** fMRI BOLD signal analysis involves preprocessing pipelines (motion correction, slice-timing, smoothing, registration) and statistical modeling (general linear model, GLM) across hundreds of thousands of voxels and thousands of time points. ICA (independent component analysis) via MELODIC decomposes a T × V spatiotemporal matrix; for 1,200 TRs and 150,000 gray-matter voxels, the matrix-SVD and subsequent unmixing are natural cuBLAS workloads. Resting-state functional connectivity computes a V × V correlation matrix — for 100,000 voxels this is a 10¹⁰-element matrix — computed efficiently on GPU via batched inner products. Dynamic functional connectivity via sliding-window or HMM approaches further multiply this cost, requiring GPU for tractable runtimes.
- **Key algorithms:** GLM (HRF convolution and OLS/WLS per voxel), ICA (MELODIC), seed-based connectivity, graph-theoretic brain network analysis, HMM dynamic connectivity, diffusion embedding, CNN/transformer resting-state biomarker extraction, k-means parcellation on GPU.
- **Datasets:** HCP fMRI (https://db.humanconnectome.org/) — resting-state and task fMRI, 7T/3T; OpenFMRI / OpenNeuro (https://openneuro.org/) — thousands of fMRI datasets in BIDS; ABIDE autism fMRI (http://fcon_1000.projects.nitrc.org/indi/abide/); UK Biobank fMRI (https://www.ukbiobank.ac.uk/).
- **Starter repos/tools:** FSL (https://fsl.fmrib.ox.ac.uk/) — MELODIC GPU ICA, FEAT GLM, BEDPOSTX; Nilearn (https://nilearn.github.io/) — Python fMRI statistical learning with scikit-learn; BrainSpace (https://github.com/MICA-MNI/BrainSpace) — gradient analysis on GPU; fMRIPrep (https://github.com/nipreps/fmriprep) — standardized preprocessing pipeline (CUDA-accelerated ANTs registration within).
- **CUDA libraries & GPU pattern:** cuBLAS for GLM design-matrix product (V × T × T × T^-1 × T × V batched); cuSOLVER for ICA SVD; cuRAND for permutation testing; GPU histogram for parcellation; multi-GPU via PyTorch for DL resting-state classifiers.

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
