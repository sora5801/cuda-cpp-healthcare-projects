# THEORY — 6.16 Cardiac Mechanics & Electromechanical Coupling

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

### 6.16 Cardiac Mechanics & Electromechanical Coupling 🟡 · Active R&D
- **Deep dive:** Extends electrophysiology simulation by coupling electrical activation to active mechanical contraction through calcium-troponin cross-bridge kinetics (e.g., Rice-Wang-Bers model). The resulting system couples a stiff ODE (ionic + cross-bridge) at each integration point to a nonlinear FEM problem (hyperelastic myocardium with active stress/strain). GPU accelerates both the per-Gauss-point ODE batch and the global Newton-Raphson iterations for the mechanical equilibrium solve. Ventricular pressure-volume loops, ejection fraction, and wall stress distributions are clinical outputs.
- **Key algorithms:** Active-stress / active-strain formulations, Holzapfel-Ogden hyperelastic constitutive law, Rice-Wang-Bers cross-bridge kinetics, monodomain EP coupling, Newton-Raphson nonlinear FEM, Guccione passive strain energy, incompressibility via penalty/mixed formulation, Windkessel boundary conditions.
- **Datasets:** UK Biobank CMR + strain imaging (https://www.ukbiobank.ac.uk); Zenodo cardiac mechanics emulation dataset (https://zenodo.org/records/7075055); ACDC segmentation challenge (https://www.creatis.insa-lyon.fr/Challenge/acdc/); MICCAI STACOM cardiac mechanics challenge data (verify URL on grand-challenge.org).
- **Starter repos/tools:** FEBio (https://github.com/febiosoftware/FEBio) — nonlinear FEM cardiac/soft-tissue mechanics solver; simcardems (https://github.com/ComputationalPhysiology/simcardems) — FEniCS-based EP+mechanics coupling; OpenCMISS/cm (https://github.com/OpenCMISS/cm) — multi-physics FEM framework; Chaste (https://github.com/Chaste/Chaste) — cardiac electromechanics tutorial.
- **CUDA libraries & GPU pattern:** Batch CVODE GPU for per-Gauss-point ODE; cuSOLVER for Newton linear solve; cuSPARSE SpMV for stiffness matrix assembly; pattern: two-level CUDA grid—elements outer, Gauss points inner—with shared memory for per-element stiffness matrix accumulation.

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
