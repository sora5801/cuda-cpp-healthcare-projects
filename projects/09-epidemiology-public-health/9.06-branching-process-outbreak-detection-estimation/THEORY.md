# THEORY — 9.6 Branching-Process Outbreak Detection & Estimation

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

### 9.6 Branching-Process Outbreak Detection & Estimation 🔴 · Frontier/Theoretical

- **Deep dive:** Models early epidemic growth as a Galton-Watson branching process or Hawkes point process to estimate the effective reproduction number Rt in near-real-time from case count time series. GPU parallelism enables simultaneous estimation of Rt across thousands of geographic units (counties, countries) simultaneously using batched Bayesian updates. The Hawkes process likelihood requires summing exponential kernels over all past events — a GPU-parallelised prefix sum operation. Branching process simulation (for outbreak probability calculations) is embarrassingly parallel: simulate 10⁵ independent outbreak realisations simultaneously on GPU to estimate extinction probabilities.
- **Key algorithms:** Galton-Watson branching process simulation, Hawkes self-exciting point process MLE, EpiEstim sliding-window Rt estimation, renewal equation Rt inference (Cori method), sequential Monte Carlo (particle filters) for real-time estimation, negative-binomial offspring distribution fitting, overdispersion estimation.
- **Datasets:**
  - CDC FluView — weekly US influenza-like illness surveillance (https://www.cdc.gov/flu/weekly/)
  - WHO Disease Outbreak News — global outbreak event data (https://www.who.int/emergencies/disease-outbreak-news)
  - COVID-19 Data Repository (CSSE Johns Hopkins) — archived case/death time series (https://github.com/CSSEGISandData/COVID-19)
  - ECDC Surveillance Atlas — European communicable disease surveillance (https://atlas.ecdc.europa.eu/)
- **Starter repos/tools:**
  - EpiEstim (https://github.com/mrc-ide/EpiEstim) — R package for Rt estimation (CPU; GPU via batched extension)
  - EpiNow2 (https://github.com/epiforecasts/EpiNow2) — Bayesian nowcasting and Rt estimation with Stan GPU backend
  - tick (https://github.com/X-DataInitiative/tick) — GPU-accelerated Hawkes process learning library
  - PyEpidemics (verify URL) — Python branching process simulation framework
- **CUDA libraries & GPU pattern:** cuRAND for Monte Carlo branching process simulation, Thrust parallel prefix sum for Hawkes likelihood, JAX/BlackJAX for GPU-based posterior inference; pattern: embarrassingly parallel ensemble simulation — each CUDA thread simulates one outbreak trajectory.

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
