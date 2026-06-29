# THEORY — 6.11 Stochastic (Gillespie) Biochemical Simulation

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

### 6.11 Stochastic (Gillespie) Biochemical Simulation 🟢 · Established
- **Deep dive:** The Gillespie Stochastic Simulation Algorithm (SSA) exactly samples the master equation for discrete molecular counts in a well-mixed chemical reaction network—critical when molecule numbers are small (transcription factors, signaling molecules). Each stochastic trajectory is independent, so GPU parallelism maps one trajectory per thread. With 1 000–10 000 trajectories needed for statistics, GPU batch SSA achieves orders-of-magnitude speedup. Tau-leaping approximations (binomial/Poisson) trade exactness for speed at higher copy numbers.
- **Key algorithms:** Gillespie SSA (direct method), Gibson-Bruck next-reaction method, tau-leaping (explicit/implicit/binomial), R-leaping, chemical Langevin equation (CLE), reaction-diffusion master equation (RDME) for spatial stochastic simulation.
- **Datasets:** BioModels Database — curated stochastic SBML models (https://www.ebi.ac.uk/biomodels); NIST Chemical Kinetics Database (https://kinetics.nist.gov); single-molecule tracking datasets on DANDI (https://dandiarchive.org); smFISH gene expression data (various GEO deposits at https://www.ncbi.nlm.nih.gov/geo/).
- **Starter repos/tools:** GillesPy2 (https://github.com/GillesPy2/GillesPy2) — Python SSA + tau-leaping + CLE, GPU backend in progress; StochPy (https://github.com/SystemsBioinformatics/stochpy) — Python stochastic simulation with SSA and tau-leaping; cuTauLeaping (verify URL — CUDA tau-leaping reference implementations in CUDA samples literature); MOOSE (https://github.com/BhallaLab/moose-core) — compartmental stochastic kinetic simulations.
- **CUDA libraries & GPU pattern:** cuRAND for per-trajectory random exponential/uniform variates (one cuRAND stream per thread); CUDA Thrust for propensity prefix-sum (direct-method reaction selection); pattern: one CUDA thread per trajectory, independent RNG state in registers; atomic operations avoided by design (each thread is fully independent).

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
