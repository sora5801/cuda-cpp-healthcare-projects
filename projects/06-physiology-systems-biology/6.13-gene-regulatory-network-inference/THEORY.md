# THEORY — 6.13 Gene Regulatory Network Inference

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

### 6.13 Gene Regulatory Network Inference 🟡 · Active R&D
- **Deep dive:** Infers the directed causal graph of transcription factor-gene interactions from single-cell RNA-seq (scRNA-seq) time-series or perturbation data. State-of-the-art methods use mutual information, GENIE3 random forests, or neural ODE formulations. Computing pairwise mutual information across 20 000 genes requires O(N²) comparisons—a 200-million-pair problem on a GPU. Bayesian network structure learning and variational inference for large graph posteriors are also GPU-amenable.
- **Key algorithms:** GENIE3 random forest (feature importance), ARACNE mutual information + data processing inequality, PANDA message-passing network inference, neural ODE (torchdiffeq) for dynamics, variational autoencoder (scVI) for expression latent space, LASSO/elastic-net for linear GRN, Granger causality.
- **Datasets:** Gene Expression Omnibus (GEO) — tens of thousands of scRNA-seq datasets (https://www.ncbi.nlm.nih.gov/geo/); ENCODE TF binding ChIP-seq (https://www.encodeproject.org); BEELINE benchmark GRN datasets (https://github.com/Murali-group/BEELINE); Human Cell Atlas scRNA-seq (https://www.humancellatlas.org).
- **Starter repos/tools:** BEELINE GRN benchmark (https://github.com/Murali-group/BEELINE) — benchmarking framework for GRN inference methods; scVI (https://github.com/scverse/scvi-tools) — deep generative models for scRNA-seq on GPU via PyTorch; torchdiffeq (https://github.com/rtqichen/torchdiffeq) — GPU neural ODE for dynamics inference; Scanpy (https://github.com/scverse/scanpy) — scRNA-seq analysis with GPU-accelerated backends (rapids-singlecell).
- **CUDA libraries & GPU pattern:** cuBLAS for pairwise correlation matrix (N×N outer product); CUDA Thrust for per-gene sort/rank (MI estimation); GPU neural ODE via PyTorch autograd + custom CUDA adjoint; pattern: tiled matrix multiply for pairwise MI, one-tile-per-gene-pair block.

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
