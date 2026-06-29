# THEORY — 6.14 Multi-Scale Physiological Modeling

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

### 6.14 Multi-Scale Physiological Modeling 🟡 · Active R&D
- **Deep dive:** Couples models operating at different spatial/temporal scales: molecular (ion channel kinetics, μs–ms), cellular (action potential, ms), tissue (wave propagation, ms–s), organ (cardiac output, heartbeat), and system (circulation, minutes). The computational challenge is that fine-scale models (cell ODE) must be solved at each quadrature point of a coarse FEM mesh simultaneously—yielding millions of ODE instances per time step. GPU batch-ODE solving (CVODE GPU) fills this role. The Virtual Physiological Human (VPH) framework coordinates inter-scale coupling.
- **Key algorithms:** Heterogeneous multiscale method (HMM), operator splitting for scale coupling, homogenization, batch CVODE for cell-level ODEs at FEM quadrature points, Windkessel/1D vessel network for circulation, FEM for organ-level mechanics/EP, co-simulation coupling (FMI standard).
- **Datasets:** Physiome Model Repository — VPH-standard CellML models (https://models.physiomeproject.org); BioModels Database (https://www.ebi.ac.uk/biomodels); UK Biobank multi-modal phenotyping (https://www.ukbiobank.ac.uk); OpenCMISS examples (https://github.com/OpenCMISS/examples).
- **Starter repos/tools:** OpenCMISS/cm (https://github.com/OpenCMISS/cm) — multi-physics multi-scale FEM framework; SUNDIALS batch CVODE GPU (https://github.com/LLNL/sundials) — batch ODE for sub-grid cell models; simcardems (https://github.com/ComputationalPhysiology/simcardems) — cardiac electromechanics multi-scale coupling; Chaste (https://github.com/Chaste/Chaste) — multi-scale cardiac + lung + tumor modeling.
- **CUDA libraries & GPU pattern:** SUNDIALS CUDA NVector + batch CVODE (cell ODE at quadrature points); cuSPARSE for coarse-mesh FEM assembly; CUDA streams for asynchronous scale coupling; pattern: two-level parallelism—CUDA grid over FEM elements, threads over per-element ODE RHS evaluation.

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
