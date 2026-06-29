# THEORY — 9.8 Wastewater-Based Epidemiology & Signal Detection

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

### 9.8 Wastewater-Based Epidemiology & Signal Detection 🟡 · Active R&D

- **Deep dive:** Infers community-level pathogen prevalence from viral RNA concentrations in wastewater, combining RT-qPCR signal time series with meteorological, demographic, and mobility covariates to nowcast and forecast disease incidence. GPU-accelerated deep learning (LSTM, Temporal Fusion Transformers) processes multivariate time series from thousands of sampling sites simultaneously; the data dimensionality is high (dozens of wastewater markers × weather variables × mobility indices per site). Bayesian hierarchical models fitted on GPU (via Stan with GPU backend or JAX) account for spatial correlation across sewage catchments. Deconvolution of wastewater signal to estimate case counts involves non-negative least-squares problems solved in parallel across sites.
- **Key algorithms:** Non-negative least-squares deconvolution, LSTM/GRU time series prediction, Temporal Fusion Transformers (TFT), Bayesian hierarchical regression, anomaly detection (isolation forests, CUSUM control charts), Poisson regression for count outcomes, spatial kriging for site interpolation.
- **Datasets:**
  - NWSS (National Wastewater Surveillance System) — US wastewater SARS-CoV-2 and flu data (https://www.cdc.gov/nwss/)
  - EU Sewage Sentinel System for SARS-CoV-2 (verify URL) — European wastewater surveillance
  - WastewaterSCAN — Stanford-led multi-pathogen wastewater monitoring (https://www.wastewaterscan.org/)
  - OpenWastewaterData (verify URL) — aggregated global wastewater surveillance
- **Starter repos/tools:**
  - PyTorch-Forecasting (https://github.com/jdb78/pytorch-forecasting) — TFT and LSTM for multivariate time series on GPU
  - Pyro (https://github.com/pyro-ppl/pyro) — GPU probabilistic programming for Bayesian wastewater signal deconvolution
  - Darts (https://github.com/unit8co/darts) — time series forecasting library with GPU support
  - NWSS Data Dashboard tools (https://www.cdc.gov/nwss/wastewater-surveillance-data-reporting.html) — CDC reference implementation
- **CUDA libraries & GPU pattern:** cuDNN for temporal model training, Pyro ELBO optimisation on GPU, cuBLAS for deconvolution least-squares; pattern: data-parallel forecasting across thousands of wastewater sites on GPU.

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
