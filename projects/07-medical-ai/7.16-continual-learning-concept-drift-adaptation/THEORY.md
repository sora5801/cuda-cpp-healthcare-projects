# THEORY — 7.16 Continual Learning & Concept Drift Adaptation

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

### 7.16 Continual Learning & Concept Drift Adaptation 🔴 · Frontier/Theoretical

- **Deep dive:** Enables deployed medical AI models to continuously incorporate new clinical data (new patient cohorts, updated imaging protocols, population shifts) without catastrophic forgetting of previously learned tasks. Experience replay stores old data in a GPU-resident memory buffer; elastic weight consolidation (EWC) computes Fisher information diagonals — a batched gradient-squared operation on GPU. Gradient Episodic Memory (GEM) requires projecting gradients to the feasible cone defined by old-task gradients, a GPU-parallelised quadratic program. Healthcare settings impose strict constraints: models cannot forget rare disease patterns seen only in early training.
- **Key algorithms:** Elastic Weight Consolidation (EWC), Progressive Neural Networks, PackNet, Gradient Episodic Memory (GEM), Experience Replay (ER), Dark Experience Replay (DER++), Learning Without Forgetting (LwF), Online EWC.
- **Datasets:**
  - MIMIC-IV — temporal partitioning by year to simulate concept drift (https://physionet.org/content/mimiciv/)
  - CheXpert / MIMIC-CXR — multi-cohort splits for sequential task training (https://stanfordmlgroup.github.io/competitions/chexpert/)
  - MedMNIST — 18-task sequential benchmark (https://medmnist.com/)
  - Skin Lesion datasets (ISIC archive) — year-stratified splits for drift simulation (https://www.isic-archive.com/)
- **Starter repos/tools:**
  - Avalanche (https://github.com/ContinualAI/avalanche) — continual learning library with GPU support and medical imaging plugins
  - Mammoth (https://github.com/aimagelab/mammoth) — GPU continual learning framework with DER++, GEM, EWC
  - FACIL (https://github.com/mmasana/FACIL) — class-incremental learning on GPU for image classifiers
  - CLMNIST / MedicalCL (verify URL) — medical imaging continual learning benchmarks
- **CUDA libraries & GPU pattern:** cuBLAS for Fisher diagonal computation, CUDA replay buffer sampling with pinned memory; pattern: gradient projection via CUDA-parallelised QP over constraint matrices.

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
