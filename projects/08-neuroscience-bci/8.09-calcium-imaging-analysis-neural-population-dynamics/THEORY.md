# THEORY — 8.9 Calcium Imaging Analysis & Neural Population Dynamics

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

### 8.9 Calcium Imaging Analysis & Neural Population Dynamics 🟢 · Established
- **Deep dive:** Two-photon calcium imaging records fluorescence from GCaMP-expressing neurons (~1 000–100 000 cells per session at 30 Hz) but requires computationally intensive post-processing: rigid/non-rigid motion correction (GPU-accelerated phase-correlation), ROI detection (NMF or CNN-based source separation), neuropil subtraction, and deconvolution of calcium transients to infer spike timing. Suite2p's GPU pipeline reduces processing of a 60-minute session from hours to minutes. Simultaneous GPU inference of population state dynamics (LFADS, CEBRA, Pi-VAE) ties calcium activity to behavior.
- **Key algorithms:** Phase-correlation motion correction (cuFFT), constrained NMF (CNMF) for source separation, graph-based ROI detection (Suite2p), OASIS/FOOPSI spike deconvolution (LASSO), LFADS latent factor analysis via dynamical systems (LSTM encoder-decoder), t-SNE/UMAP for population visualization.
- **Datasets:** Allen Brain Observatory calcium imaging (https://portal.brain-map.org); DANDI calcium imaging datasets (https://dandiarchive.org); OpenNeuro two-photon datasets (https://openneuro.org); CaImAn demo datasets (https://github.com/flatironinstitute/CaImAn).
- **Starter repos/tools:** Suite2p (https://github.com/MouseLand/suite2p) — fast GPU calcium imaging pipeline (registration + detection + deconvolution); CaImAn (https://github.com/flatironinstitute/CaImAn) — Flatiron CNMF with GPU motion correction; CellProfiler (https://github.com/CellProfiler/CellProfiler) — general imaging analysis applicable to calcium phenotyping; LFADS (https://github.com/google-research/google-research/tree/master/lfads) — GPU latent factor analysis for population dynamics.
- **CUDA libraries & GPU pattern:** cuFFT for phase-correlation motion correction (all frames batch-FFT); cuDNN for CNN ROI detection; custom CUDA NMF solver (multiplicative update with shared-memory A^T A computation); pattern: frame-parallel GPU processing pipeline with pinned memory ring buffer for continuous acquisition.

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
