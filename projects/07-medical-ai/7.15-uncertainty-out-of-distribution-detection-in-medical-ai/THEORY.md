# THEORY — 7.15 Uncertainty & Out-of-Distribution Detection in Medical AI

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

### 7.15 Uncertainty & Out-of-Distribution Detection in Medical AI 🔴 · Frontier/Theoretical

- **Deep dive:** Quantifies model uncertainty for medical predictions to flag distribution shift (e.g., a new imaging protocol, rare pathology) and prevent silent failures in deployed AI. Bayesian approximations (MC Dropout, Deep Ensembles, SWAG) require multiple stochastic forward passes — naturally parallelised across ensemble members on GPU. Conformal prediction calibrates coverage guarantees without distribution assumptions, requiring GPU-parallelised score computation over large calibration sets. Energy-based OOD scoring on medical images computes a scalar energy per sample in a parallelised batch. The research challenge is calibrating uncertainty to actual clinical risk without access to labelled OOD data.
- **Key algorithms:** MC Dropout (Gal & Ghahramani), Deep Ensembles, SWAG (SWA-Gaussian), Normalising Flows for density estimation, Energy-Based Models, Mahalanobis OOD detection, Conformal Prediction (split, cross-conformal), Temperature Scaling, Label Smoothing.
- **Datasets:**
  - Camelyon17 (WILDS) — histopathology with explicit hospital distribution shift (https://wilds.stanford.edu/datasets/#camelyon17)
  - MIMIC-CXR — train/test splits across demographic strata for OOD evaluation (https://physionet.org/content/mimic-cxr/)
  - RSNA Pneumonia Detection — Kaggle competition dataset for OOD robustness benchmarks (verify URL)
  - MedMNIST — 18 standardised 2D/3D medical classification tasks for OOD benchmarking (https://medmnist.com/)
- **Starter repos/tools:**
  - WILDS (https://github.com/p-lambda/wilds) — distribution shift benchmark with CausalML support
  - PyTorch-Uncertainty (https://github.com/ENSTA-U2IS-AI/torch-uncertainty) — uncertainty methods on GPU
  - ConformalCI (https://github.com/aangelopoulos/conformal-prediction) — conformal prediction calibration
  - Laplace-Redux (https://github.com/AlexImmer/Laplace) — post-hoc Laplace approximation for pretrained NNs
- **CUDA libraries & GPU pattern:** cuDNN for parallel ensemble member forward passes, cuBLAS for Mahalanobis computation over feature covariance matrix; pattern: batch-parallel ensemble evaluation with stacked model weights.

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
