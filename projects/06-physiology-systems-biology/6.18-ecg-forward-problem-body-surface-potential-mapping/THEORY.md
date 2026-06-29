# THEORY — 6.18 ECG Forward Problem & Body-Surface Potential Mapping

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

### 6.18 ECG Forward Problem & Body-Surface Potential Mapping 🟢 · Established
- **Deep dive:** The ECG forward problem maps cardiac electrical sources (transmembrane currents from EP simulation) to body-surface potentials via the quasi-static Poisson equation on a torso volume conductor model. The transfer matrix (lead-field matrix) is computed once by solving many FEM boundary value problems (one per electrode), then applied repeatedly as a dense matrix-vector product at each time step of the EP simulation. GPU acceleration is ideal for both the batched FEM assembly and the dense matrix-vector multiply.
- **Key algorithms:** Quasi-static Poisson equation (torso conductivity model), finite element method on torso mesh, lead-field/transfer matrix computation, multipole source representation, method of fundamental solutions, ECG inverse problem (regularized Tikhonov, total variation), boundary element method (BEM).
- **Datasets:** PhysioNet ECG databases (https://physionet.org); EDGAR body-surface potential database (https://edgar.sci.utah.edu — verify URL); Cardioid ECG module examples (https://github.com/llnl/cardioid); Visible Human torso geometry (https://www.nlm.nih.gov/research/visible/visible_human.html).
- **Starter repos/tools:** Cardioid/LLNL (https://github.com/llnl/cardioid) — includes ECG forward solver module; openCARP (https://git.opencarp.org/openCARP/openCARP) — ECG lead calculation post-processing; SCIRun (https://github.com/SCIInstitute/SCIRun) — Utah scientific computing platform for ECG forward/inverse; APBS (https://github.com/Electrostatics/apbs) — electrostatics PDE solver adaptable to torso geometry.
- **CUDA libraries & GPU pattern:** cuBLAS DGEMV for transfer-matrix application at each time step; cuSOLVER for FEM system solve during transfer-matrix construction; batched cuSOLVER for simultaneous electrode-source BVPs; pattern: parallel BVP solves (one per electrode) with shared torso mesh.

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
