# THEORY — 9.10 Mobility-Based Epidemic Nowcasting

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

### 9.10 Mobility-Based Epidemic Nowcasting 🟡 · Active R&D

- **Deep dive:** Infers current epidemic state and short-term trajectory from human mobility data (mobile phone GPS, retail foot traffic, transit ridership) using data assimilation methods that combine mobility signals with epidemiological models. GPU enables rapid sequential Monte Carlo (particle filter) updates as new mobility observations arrive hourly, running thousands of particles simultaneously. Graph neural networks learn spatial transmission patterns from mobility flow matrices — a GPU-parallelised sparse graph convolution. The bottleneck is the batched epidemic ODE integration for all particles in the ensemble simultaneously.
- **Key algorithms:** Sequential Monte Carlo (particle filtering), ensemble Kalman filter (EnKF), graph convolutional networks on mobility graphs, LSTM encoder-decoder for mobility sequence learning, MAP estimation for transmission rate, community mobility indices as predictors (Google CMR).
- **Datasets:**
  - Google Community Mobility Reports — country/region mobility indices during COVID-19 (https://www.google.com/covid19/mobility/)
  - SafeGraph/Dewey POI visit data — US retail foot traffic (verify access terms)
  - Apple Mobility Trends — routing request data by transit type (verify URL)
  - Citymapper Mobility Index — urban mobility across 40 cities (verify URL)
- **Starter repos/tools:**
  - GLEAM mobility pipeline (https://www.gleamviz.org/) — global airline + commuting mobility for epidemic modelling
  - CuPy (https://github.com/cupy/cupy) — GPU NumPy for particle filter implementation
  - Epiforecast (verify URL) — real-time epidemic nowcasting framework
  - PYMC (https://github.com/pymc-devs/pymc) — probabilistic programming with GPU JAX/Numba backend for data assimilation
- **CUDA libraries & GPU pattern:** cuRAND for particle resampling, cuBLAS for ensemble matrix operations, cuGraph for mobility graph convolutions; pattern: particle filter with GPU-parallel ODE integration and resampling.

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
