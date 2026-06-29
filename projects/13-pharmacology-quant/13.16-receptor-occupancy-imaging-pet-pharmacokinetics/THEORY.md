# THEORY — 13.16 Receptor Occupancy Imaging & PET Pharmacokinetics

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

### 13.16 Receptor Occupancy Imaging & PET Pharmacokinetics 🟡 · Active R&D

- **Deep dive:** Analyses PET neuroimaging data to quantify receptor occupancy by drug candidates across thousands of brain voxels simultaneously. The Logan reference tissue method and two-tissue compartmental models must be fitted to the time-activity curve (TAC) at each voxel — a problem with 100k+ independent nonlinear regressions that map directly to GPU parallelism. GPU-parallel voxel-wise model fitting achieves near-real-time analysis of 3D PET volumes (128×128×63 voxels). Virtual receptor occupancy simulations (coupling PBPK with brain RO submodel) for dose selection require batched ODE integration on GPU across candidate dose levels.
- **Key algorithms:** Logan reference tissue method, two-tissue compartmental model, simplified reference tissue model (SRTM), voxel-wise ODE fitting with Levenberg-Marquardt on GPU, Patlak graphical analysis, partial volume correction, kinetic parameter estimation (K1, k2, BP_ND).
- **Datasets:**
  - OpenNeuro PET datasets — open-access brain PET with kinetic data (https://openneuro.org/)
  - NeuroVault PET studies — aggregated neuroimaging PET data (https://neurovault.org/)
  - BrainPET benchmark datasets (verify URL — NIMH)
  - ADNI PET-amyloid data — longitudinal PET for Alzheimer imaging (https://adni.loni.usc.edu/)
- **Starter repos/tools:**
  - NiftyPAD (verify URL) — GPU-parallelised PET kinetic modelling toolkit
  - TPCCLIB (verify URL) — C library for PET kinetic analysis (CPU; GPU extension possible)
  - Pumas (https://pumas.ai/) — GPU-accelerated brain RO-PBPK coupling in Julia
  - SimplePET (https://github.com/UCL/simplicity) — Python PET simulation and analysis (verify URL)
- **CUDA libraries & GPU pattern:** Custom CUDA Levenberg-Marquardt kernels for per-voxel TAC fitting, cuBLAS for covariance matrix inversion, cuFFT for PET sinogram reconstruction; pattern: one CUDA thread per voxel, embarrassingly parallel kinetic fitting.

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
