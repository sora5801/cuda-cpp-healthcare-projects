# THEORY — 11.12 Downstream Processing & Chromatography Simulation

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

### 11.12 Downstream Processing & Chromatography Simulation 🟡 · Active R&D

- **Deep dive:** Protein A affinity, ion-exchange, and size-exclusion chromatography columns for antibody purification are governed by advection-dispersion-reaction (ADR) PDEs coupled with adsorption isotherm equations (steric mass action, SMA). GPU-accelerated PDE solvers (finite-volume or spectral methods) simulate full column dynamics in seconds per run, enabling in silico process characterization (DoE) across 100s of loading, wash, and elution conditions in parallel. Inverse problem fitting of SMA parameters from batch isotherm experiments uses GPU-accelerated Bayesian optimization. The bottleneck is the large stiff ODE system for multi-component competitive adsorption.
- **Key algorithms:** Advection-dispersion-reaction PDE (Godunov scheme / WENO), steric mass action (SMA) isotherm model, general rate model (GRM), shrinking core diffusion model, Bayesian optimization for process development, GPU-parallel Latin hypercube DoE.
- **Datasets:** CADET Benchmark Cases — chromatography simulation validation (https://github.com/modsim/CADET); USP Bioprocess Data Repository — chromatography process development records (verify URL via NIST/USP); PDB-based antibody charge maps for adsorption prediction; OpenChrom mass-spectrometry chromatography datasets (https://www.openchrom.net/).
- **Starter repos/tools:** CADET (https://github.com/modsim/CADET) — Chromatography Analysis and Design Toolkit, CPU reference; CADET-Process (https://github.com/modsim/CADET-Process) — Python optimization wrapper for CADET; GPU-ADR solvers via CUDA finite-volume (custom implementation, verify via GitHub search "GPU chromatography simulation"); PyTorch surrogate for chromatography (verify URL via Biotechnology Journal 2024).
- **CUDA libraries & GPU pattern:** CUDA finite-volume kernels for 1D PDE time-stepping (one thread per spatial grid point), cuSPARSE for implicit diffusion system, Thrust for parallel DoE condition enumeration; pattern: 200 chromatography conditions enumerated → GPU PDE solve per condition in parallel → elution profile extraction → Bayesian optimizer selects next DoE → iterate until convergence.

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
