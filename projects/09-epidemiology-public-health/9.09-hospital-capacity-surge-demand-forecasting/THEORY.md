# THEORY — 9.9 Hospital Capacity & Surge Demand Forecasting

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

### 9.9 Hospital Capacity & Surge Demand Forecasting 🟡 · Active R&D

- **Deep dive:** Predicts short-term hospital admission volumes, ICU occupancy, and ventilator demand to enable proactive resource allocation during epidemic surges or seasonal peaks. GPU-accelerated LSTM, Transformer, and ensemble models trained on EHR admission records, regional case counts, wastewater signals, and mobility data produce rolling 14-day forecasts. The volume of hospital time series (thousands of hospitals × dozens of admission types × 365 days/year) is processed in parallel on GPU; each hospital's time series is a separate batch element. Real-time retraining on streaming data requires frequent mini-batch SGD on GPU to adapt to evolving epidemic waves.
- **Key algorithms:** LSTM/GRU multi-step forecasting, Temporal Fusion Transformers, N-BEATS, Prophet (Bayesian decomposition), Gaussian process regression for uncertainty, hierarchical reconciliation (MinT), ensemble averaging, ARIMA + neural hybrids, conformal prediction intervals.
- **Datasets:**
  - HHS Protect Hospital Capacity Data — US hospital capacity and admissions (https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/6xf2-c3ie)
  - ECDC Hospital Data — European hospital admissions and ICU occupancy (https://www.ecdc.europa.eu/en/covid-19/data)
  - NHS England Situation Reports — UK hospital admissions and bed occupancy (https://www.england.nhs.uk/statistics/)
  - COVID-19 Forecast Hub submissions — ensemble of >50 models (https://covid19forecasthub.org/)
- **Starter repos/tools:**
  - PyTorch-Forecasting (https://github.com/jdb78/pytorch-forecasting) — TFT, LSTM, N-BEATS on GPU
  - Darts (https://github.com/unit8co/darts) — multi-model time series forecasting with GPU backend
  - COVID-19 Forecast Hub (https://github.com/reichlab/covid19-forecast-hub) — ensemble model aggregation infrastructure
  - GluonTS (https://github.com/awslabs/gluonts) — probabilistic time series on GPU via MXNet/PyTorch
- **CUDA libraries & GPU pattern:** cuDNN for temporal model training, JAX XLA for parallelised Gaussian process forecasting, NCCL for multi-GPU ensemble training; pattern: panel data parallel — each hospital's time series as a batch element.

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
