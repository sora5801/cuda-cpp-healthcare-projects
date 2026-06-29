# THEORY — 7.12 Sepsis Early Warning System

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

### 7.12 Sepsis Early Warning System 🟡 · Active R&D

- **Deep dive:** Predicts the onset of sepsis 3–6 hours before clinical recognition from streaming ICU vitals, lab values, and medication records using recurrent or transformer architectures. The GPU bottleneck is batched forward passes through temporal models (LSTM, GRU, Transformer-XL) over thousands of patient time series simultaneously. Real-time deployment requires sub-second latency over continuously appended EHR streams. Processing irregular time-series (lab values arrive at non-uniform intervals) requires attention mechanisms that weigh observations by recency and relevance — these attention operations are CUDA-accelerated. Large training cohorts (>100k ICU admissions) sustain continuous GPU utilisation throughout training.
- **Key algorithms:** LSTM/GRU temporal classifiers, Transformer-XL for long EHR sequences, Temporal Fusion Transformers (TFT), missing-value imputation via learned decay, AUROC-calibrated threshold selection, early stopping with Clinical Early Warning Scores (qSOFA, SOFA) as baselines, conformal prediction for uncertainty.
- **Datasets:**
  - MIMIC-Sepsis benchmark (https://arxiv.org/abs/2510.24500) — curated sepsis trajectory subset of MIMIC-IV
  - eICU-CRD — 200k+ admissions, multi-site for generalisation testing (https://eicu-crd.mit.edu/)
  - PhysioNet/Computing in Cardiology Challenge 2019 — sepsis prediction from ICU time series (https://physionet.org/content/challenge-2019/)
  - HiRID — high-resolution ICU dataset from Bern University Hospital (https://physionet.org/content/hirid/)
- **Starter repos/tools:**
  - MIMIC-Extract (https://github.com/MLforHealth/MIMIC_Extract) — standardised MIMIC ICU feature tables
  - PyHealth (https://github.com/sunlabuiuc/PyHealth) — healthcare AI library with ICU prediction tasks on GPU
  - ETHOS (verify URL) — transformer-based sepsis prediction on EHR tokens
  - Temporal Fusion Transformer (https://github.com/jdb78/pytorch-forecasting) — multi-horizon temporal model with GPU support
- **CUDA libraries & GPU pattern:** cuDNN for LSTM/GRU cells, Flash Attention for transformer EHR models, Thrust for sorting irregular timestamps; pattern: padded minibatch of patient time series with masking, GPU-resident rolling window inference for real-time alerting.

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
