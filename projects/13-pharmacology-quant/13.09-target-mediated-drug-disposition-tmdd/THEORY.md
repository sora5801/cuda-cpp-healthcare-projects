# THEORY — 13.9 Target-Mediated Drug Disposition (TMDD)

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

### 13.9 Target-Mediated Drug Disposition (TMDD) 🟡 · Active R&D

- **Deep dive:** TMDD models describe biologics (monoclonal antibodies, bispecifics) whose elimination is dominated by saturable binding to their pharmacological target, producing nonlinear, dose-dependent PK. The full TMDD ODE system (Mager-Jusko, 2001) is stiff due to fast receptor association/dissociation kinetics, requiring implicit stiff solvers. GPU parallelism is critical for virtual patient population simulations: fitting 1000 virtual patients × 100 dose schedules × stiff ODE = 10⁵ independent stiff integrations run simultaneously on GPU. Approximations (quasi-steady-state, Michaelis-Menten) reduce stiffness but must be validated against full TMDD for each compound — GPU enables this validation across large parameter grids cheaply.
- **Key algorithms:** Full TMDD ODE system (4 equations: free drug, free receptor, drug-receptor complex, total drug), Quasi-Equilibrium (QE) approximation, Quasi-Steady-State (QSS) / Michaelis-Menten approximation, stiff LSODA/CVODE integration, bivalent TMDD extensions (2025 Straube model), population NLME fitting of TMDD, slow-binding approximation.
- **Datasets:**
  - Published mAb PK datasets from Phase I trials (verify via PharmPK or ClinicalPharmacology.nih.gov)
  - Open Systems Pharmacology TMDD model examples (https://github.com/Open-Systems-Pharmacology/)
  - NONMEM TMDD example scripts (verify URL)
  - BioModels Database TMDD models (https://www.ebi.ac.uk/biomodels/)
- **Starter repos/tools:**
  - Pumas (https://pumas.ai/) — GPU population TMDD fitting in Julia
  - NONMEM (https://www.iconplc.com/solutions/technologies/nonmem/) — industry standard NLME for TMDD (verify GPU support status)
  - Monolix TMDD library (https://lixoft.com/model-libraries/pkpd-library/) — pre-built TMDD models (verify URL)
  - nvQSP (https://github.com/NVIDIA-Digital-Bio/nvQSP) — GPU stiff ODE solver applicable to TMDD virtual patient simulations
- **CUDA libraries & GPU pattern:** Custom CUDA CVODE/RODAS4 stiff solver, cuBLAS for Jacobian LU factorisation in implicit integration; pattern: batch-parallel stiff ODE integration — one virtual patient per CUDA thread block, receptor binding equations in shared memory.

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
