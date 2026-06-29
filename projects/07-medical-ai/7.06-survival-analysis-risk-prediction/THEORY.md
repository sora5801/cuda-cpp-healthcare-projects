# THEORY — 7.6 Survival Analysis & Risk Prediction

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

### 7.6 Survival Analysis & Risk Prediction 🟡 · Active R&D

- **Deep dive:** Estimates time-to-event outcomes (death, hospital readmission, disease progression) from longitudinal EHR data, imaging, or omics using neural extensions of Cox proportional hazards (DeepSurv), discrete-time survival models (DeepHit), and dynamic variational approaches (DySurv). The GPU bottleneck is batched forward passes through deep neural networks that process long time-series of irregular clinical observations. Computing the partial likelihood loss (Cox) requires sorting survival times and summing risk sets, which can be parallelised as a GPU prefix-sum operation. Large cohort training (>100k patients with hundreds of clinical features) sustains high GPU utilisation throughout.
- **Key algorithms:** Cox Proportional Hazards (DeepSurv), Discrete Survival (DeepHit), Dynamic Bayesian survival (DySurv), random survival forests on GPU, competing risks (Fine-Gray), Concordance Index (C-index) optimisation, deep conditional transformation models, inverse probability of censoring weighting (IPCW).
- **Datasets:**
  - UK Biobank — 500k participant longitudinal cohort with genetic, imaging, and EHR data (https://www.ukbiobank.ac.uk/)
  - SEER Database — cancer incidence and survival from US National Cancer Institute (https://seer.cancer.gov/)
  - eICU Collaborative Research Database — 200k+ critical care admissions across 200 hospitals (https://eicu-crd.mit.edu/)
  - TCGA clinical outcomes — survival labels linked to molecular profiling (https://www.cancer.gov/tcga)
- **Starter repos/tools:**
  - PyCox (https://github.com/havakv/pycox) — GPU-accelerated DeepSurv, DeepHit, MTLR implementations in PyTorch
  - Lifelines (https://github.com/CamDavidsonPilon/lifelines) — classical survival library (CPU); pairs with GPU model backends
  - DySurv (verify URL) — CVAE-based dynamic survival from EHR time series
  - scikit-survival (https://github.com/sebp/scikit-survival) — ensemble survival methods; GPU ensemble via XGBoost integration
- **CUDA libraries & GPU pattern:** cuDNN for network forward/backward, custom CUDA prefix-sum kernels for Cox risk set computation, Thrust for efficient sorting of survival times; pattern: data-parallel minibatch with custom loss kernel.

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
