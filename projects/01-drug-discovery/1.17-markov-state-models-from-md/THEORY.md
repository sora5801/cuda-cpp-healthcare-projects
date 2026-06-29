# THEORY — 1.17 Markov State Models from MD

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

### 1.17 Markov State Models from MD 🟡 · Active R&D

- **Deep dive:** Markov State Models (MSMs) discretize MD conformational space into metastable states and estimate transition probabilities from long or many-short trajectories. Building MSMs requires: (1) featurization of millions of MD frames, (2) dimensionality reduction (tICA/PCA), (3) clustering (k-means/mini-batch k-means), and (4) transition matrix estimation. Steps 1–3 are GPU-acceleratable via cuML or custom CUDA kernels. The payoff is extraction of thermodynamics and kinetics (kon, koff, binding pathways) from aggregated μs-ms of GPU MD.
- **Key algorithms:** Time-lagged independent component analysis (tICA), mini-batch k-means clustering, transition matrix MLE/Bayesian, PCCA+ for state coarse-graining, Chapman-Kolmogorov test, variational approach to Markov processes (VAMP).
- **Datasets:** MDCATH — 5 μs MD trajectories for 272 proteins (https://huggingface.co/datasets/compsciencelab/mdcath); Fast-folder benchmark trajectories (chignolin, Trp-cage, Villin — publicly shared by Piana/Shaw); GPCRmd (https://gpcrmd.org); D. E. Shaw millisecond trajectories (accessible via RCSB deposition).
- **Starter repos/tools:** PyEMMA (https://github.com/markovmodel/PyEMMA) — MSM construction with CUDA-accelerated k-means; MSMBuilder (https://github.com/msmbuilder/msmbuilder) — statistical models for biomolecular dynamics; deeptime (https://github.com/deeptime-ml/deeptime) — VAMPnets and modern MSM tools on GPU; cuML (https://github.com/rapidsai/cuml) — GPU-accelerated k-means and PCA via RAPIDS.
- **CUDA libraries & GPU pattern:** cuML k-means for GPU clustering; custom CUDA kernels for pairwise RMSD featurization; cuBLAS for tICA covariance matrix; GPU-parallel trajectory loading via RAPIDS cuDF.

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
