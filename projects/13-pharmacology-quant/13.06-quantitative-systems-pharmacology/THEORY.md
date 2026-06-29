# THEORY — 13.6 Quantitative Systems Pharmacology

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

### 13.6 Quantitative Systems Pharmacology 🟡 · Active R&D

- **Deep dive:** QSP models integrate pharmacokinetics with mechanistic biology (immune signalling, tumour growth, disease pathway models) through large ODE systems (100–10,000 equations). Stiff ODE integration dominates compute: a QSP model with 1,000 equations × 1,000 virtual patients requires solving 10⁶ coupled ODEs simultaneously. NVIDIA's nvQSP implements GPU-accelerated RODAS4 (an L-stable solver for stiff systems) specifically for this purpose, achieving orders-of-magnitude speedup. Virtual twin patient simulations for oncology trials run thousands of patient ODEs simultaneously, with GPU thread blocks each solving one patient's equation system. Physics-Informed Neural Networks (PINNs) are emerging as GPU-native surrogates that learn QSP system dynamics from data.
- **Key algorithms:** RODAS4/LSODA stiff ODE integration, sensitivity analysis (forward/adjoint), global parameter search (population Monte Carlo, ABC), PBPK-QSP coupling, immune checkpoint model ODEs (anti-PD1, CAR-T dynamics), tumour growth inhibition models, Physics-Informed Neural Networks (PINNs), QSP model reduction (MBAM).
- **Datasets:**
  - QSP model repository (DDMoRe consortium) — interoperable QSP models (https://www.ddmore.eu/)
  - BioModels Database — 2000+ curated mathematical models of biological processes (https://www.ebi.ac.uk/biomodels/)
  - NIH Systems Biology Data (verify URL) — mechanistic pathway data
  - Open Systems Pharmacology QSP library (https://github.com/Open-Systems-Pharmacology/QSP-PK-Model-Library)
- **Starter repos/tools:**
  - nvQSP (https://github.com/NVIDIA-Digital-Bio/nvQSP) — NVIDIA GPU-accelerated QSP ODE solver (CUDA RODAS4)
  - SBML/Tellurium (https://github.com/sys-bio/tellurium) — systems biology model simulator; GPU backend emerging
  - SBMLtoODEjl (verify URL) — Julia ODE generator from SBML for GPU integration via CUDA.jl
  - Copasi (https://copasi.org/) — biochemical network simulator; parallel via COPASI MPI interface
- **CUDA libraries & GPU pattern:** Custom CUDA RODAS4 stiff ODE kernels, cuBLAS for Jacobian LU factorisation, cuSPARSE for sparse ODE right-hand-side; pattern: one CUDA thread block per virtual patient, each thread within block updates one ODE compartment per step.

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
