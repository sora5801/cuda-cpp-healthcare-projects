# THEORY — 13.2 PBPK at Scale

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

### 13.2 PBPK at Scale 🟡 · Active R&D

- **Deep dive:** Physiologically based pharmacokinetic (PBPK) models describe drug disposition through ~15 interconnected physiological compartments (blood, liver, kidney, lung, fat, muscle, etc.), each defined by ODEs parameterised by tissue volumes, blood flows, and metabolic rate constants. High-throughput virtual screening of thousands of compounds requires solving the full PBPK ODE system (30–60 ODEs) for each compound simultaneously — a batch of 10,000 compounds is 600,000 simultaneous ODEs, well-suited to GPU-parallel Runge-Kutta integration. NVIDIA's nvQSP implements a GPU-accelerated RODAS4 stiff ODE solver specifically for QSP/PBPK population studies. Monte Carlo virtual population simulations (500–5000 virtual subjects per compound) further multiply the parallelism requirement.
- **Key algorithms:** RODAS4 stiff ODE solver (GPU implementation), Runge-Kutta 4/5, adaptive stepsize control, PBPK parameter estimation via Bayesian MCMC, machine-learning-predicted ADME inputs (logP, Vd, CLint), tissue-plasma partition coefficient estimation (Rodgers-Rowland, Berezhkovskiy).
- **Datasets:**
  - Open Systems Pharmacology PBPK model repository (https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library) — 100+ validated human PBPK models
  - DrugBank ADME data — 14k+ drugs with physicochemical and metabolic parameters (https://www.drugbank.com/)
  - FDA/EMA drug approval submission PK data — publicly available pharmacokinetic data from drug labels (verify URL)
  - ChEMBL ADMET data — assay-based ADME measurements (https://www.ebi.ac.uk/chembl/)
- **Starter repos/tools:**
  - PK-Sim (https://github.com/Open-Systems-Pharmacology/PK-Sim) — open-source whole-body PBPK software (C#; GPU via OSP Suite)
  - nvQSP (https://github.com/NVIDIA-Digital-Bio/nvQSP) — NVIDIA GPU-accelerated QSP/PBPK ODE solvers (CUDA)
  - SimBiology (MATLAB) — PBPK modelling with parallel computing toolbox for GPU (verify URL)
  - PBPKsim (verify URL) — Python PBPK simulation framework
- **CUDA libraries & GPU pattern:** Custom CUDA RODAS4/RK45 stiff ODE solver kernels, cuBLAS for Jacobian evaluation, Thrust for adaptive stepsize selection; pattern: one CUDA thread block per virtual subject, with ODE compartments mapped to shared memory.

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
