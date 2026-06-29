# THEORY — 5.2 Radiotherapy Treatment-Plan Optimization

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

### 5.2 Radiotherapy Treatment-Plan Optimization 🟡 · Active R&D
- **Deep dive:** IMRT/VMAT plan optimization solves a large-scale constrained optimization: minimize dose to OARs subject to PTV coverage constraints, with variables being beam aperture shapes or fluence maps. The dose-influence matrix D (N_voxels × N_beamlets, typically 10⁶ × 10⁴) must be computed and stored on GPU; the iterative optimizer (gradient descent, IPOPT, L-BFGS) performs repeated sparse matrix-vector products (D·x) per iteration. GPU SpMV reduces each DMAT-vector product from seconds to milliseconds, enabling real-time adaptive re-optimization. Biological-effect optimization (TCP/NTCP) and robust optimization over uncertainty scenarios further multiply the compute by the number of scenarios (~50–100 for robust plans).
- **Key algorithms:** Fluence-map optimization (quadratic programming, L-BFGS), direct aperture optimization (DAO), volumetric modulated arc therapy (VMAT) optimization, robust optimization (minimax), biological TCP/NTCP optimization, multi-criteria optimization (Pareto front navigation), deep learning dose prediction (U-Net).
- **Datasets:** OpenKBP (knowledge-based planning) dataset (https://github.com/ababier/open-kbp) — 340 head-and-neck IMRT plans; TCIA RT datasets; PlanIQ (verify URL); AAPM TG-263 structure naming dataset; OpenTPS test datasets.
- **Starter repos/tools:** matRad (https://github.com/e0404/matRad) — open-source MATLAB treatment planning, photon/proton/carbon; pyRadPlan (https://github.com/e0404/pyRadPlan) — Python interoperable extension of matRad; CERR (https://github.com/cerr/CERR) — MATLAB comprehensive RT research platform with DICOM-RT; OpenTPS (https://opentps.org/) — open-source Python/GPU treatment planning system (verify URL).
- **CUDA libraries & GPU pattern:** cuSPARSE (SpMV for D·fluence products); cuBLAS (OAR/PTV dose-volume histogram computation); CUDA warp-level reductions for DVH statistics; GPU-resident D-matrix in CSR format; multi-GPU for scenario-parallel robust optimization.

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
