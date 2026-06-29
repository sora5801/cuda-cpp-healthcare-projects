# THEORY — 5.5 Deformable Dose Accumulation & Adaptive Radiotherapy

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

### 5.5 Deformable Dose Accumulation & Adaptive Radiotherapy 🟡 · Active R&D
- **Deep dive:** Adaptive radiotherapy (ART) adjusts the treatment plan during a course of fractions based on daily imaging (CBCT), requiring: (1) daily GPU CBCT reconstruction, (2) deformable image registration (DIR) between planning CT and daily image, (3) deformable warping of the dose distribution via the DVF to accumulate physically meaningful total dose. DIR and dose warping on a 512³ volume require iterative GPU Demons/B-spline followed by trilinear interpolation of the 3D DVF — each voxel's dose is mapped to its deformed position. Online ART workflows (MR-Linac) must complete all steps in <5 min, achievable only with GPU. Uncertainty in DIR propagates to dose uncertainty, motivating ensemble DIR and probabilistic dose accumulation on GPU.
- **Key algorithms:** Diffeomorphic Demons DIR, B-spline FFD, VoxelMorph for daily DIR, trilinear DVF warp for dose accumulation, summation-of-deformed-doses vs. energy-mass-transfer method, DIR uncertainty quantification, plan re-optimization on adapted anatomy.
- **Datasets:** TCIA CT-on-rails / CBCT datasets; DIR-Lab 4D-CT lung dataset (https://www.dir-lab.com/); AAPM TG-132 DIR test cases; CREATIS deformable lung phantom (https://www.creatis.insa-lyon.fr/).
- **Starter repos/tools:** Plastimatch (https://plastimatch.org/) — GPU B-spline DIR + dose warping, DICOM-RT; VoxelMorph (https://github.com/voxelmorph/voxelmorph) — DL DIR for daily CBCT to CT; CERR (https://github.com/cerr/CERR) — deformable dose accumulation pipeline; pyRadPlan (https://github.com/e0404/pyRadPlan) — adaptive plan re-optimization.
- **CUDA libraries & GPU pattern:** GPU Demons iterative kernel (per-voxel force computation + Gaussian smoothing via cuFFT); custom CUDA trilinear warp for dose mapping; cuBLAS for B-spline coefficient computation; CUDA atomic adds for accumulated dose histogram.

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
