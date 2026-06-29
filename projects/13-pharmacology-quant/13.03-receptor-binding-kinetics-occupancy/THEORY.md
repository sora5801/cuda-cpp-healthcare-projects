# THEORY — 13.3 Receptor Binding Kinetics & Occupancy

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

### 13.3 Receptor Binding Kinetics & Occupancy 🟡 · Active R&D

- **Deep dive:** Simulates drug-receptor association, dissociation, and signalling downstream of receptor occupancy using differential equation models (two-state, ternary complex, operational models of agonism). In receptor occupancy (RO) imaging data analysis, GPU parallelism enables simultaneous fitting of PET tracer binding across thousands of brain voxels. For in silico virtual screening, GPU batch evaluation of binding kinetics models for thousands of drug candidates (each with different kon/koff) is the bottleneck — solved with CUDA-batched ODE integration. Extended kinetic models (induced-fit docking, conformational selection) couple binding kinetics to structural biology force fields for GPU-accelerated MD-enhanced occupancy predictions.
- **Key algorithms:** Two-state receptor model ODE, Ternary Complex Model (TCM), Operational Model of Agonism, kinetic rate equation fitting (kon, koff, Kd), PET Logan reference method, Receptor Occupancy ED50 estimation, cAMP/calcium signalling cascade ODEs, mean-field receptor population models.
- **Datasets:**
  - ChEMBL binding kinetics data — kon/koff/Kd for thousands of drug-receptor pairs (https://www.ebi.ac.uk/chembl/)
  - BindingDB kinetics subset (https://www.bindingdb.org/)
  - OpenNeuro PET datasets — receptor occupancy imaging data (https://openneuro.org/)
  - Guide to Pharmacology (GtoPdb) — curated receptor/ligand database (https://www.guidetopharmacology.org/)
- **Starter repos/tools:**
  - PyDyNo (verify URL) — dynamic receptor simulation in Python
  - RTKI (Receptor-Target Kinetics Interface) (verify URL) — kinetics fitting framework
  - nvQSP (https://github.com/NVIDIA-Digital-Bio/nvQSP) — GPU ODE batching applicable to receptor kinetics models
  - PySB (https://github.com/pysb/pysb) — Python rule-based biochemical network modelling
- **CUDA libraries & GPU pattern:** Custom CUDA RK4 batched ODE kernels for receptor kinetics, cuRAND for parameter uncertainty propagation, cuBLAS for Jacobian computation; pattern: one CUDA thread per drug candidate, each solving receptor binding ODEs in parallel.

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
