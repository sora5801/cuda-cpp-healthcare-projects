# THEORY — 3.15 Hi-C / 3D Genome Contact Analysis

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

### 3.15 Hi-C / 3D Genome Contact Analysis 🟡 · Active R&D
- **Deep dive:** Hi-C maps chromatin contacts genome-wide, producing sparse contact matrices of size (genome_bins × genome_bins) at 1–10 kb resolution. Downstream analysis—matrix normalisation (ICE/KR balancing), TAD boundary calling, compartment A/B classification, and loop detection—involves iterative matrix operations on matrices with 3×10⁶ bins (3 Gb of data at 1 kb). GPU acceleration of the ICE iterative correction algorithm (repeated sparse matrix-vector products) and the 2D convolution-based loop caller (HiCCUPS) is particularly impactful. ChromaFold (2024) trains a lightweight CNN on a GPU to predict 3D contact maps from 1D accessibility signals.
- **Key algorithms:** ICE / KR iterative matrix balancing (sparse MVM); eigendecomposition for A/B compartments; 1D insulation score for TAD boundary detection; HiCCUPS 2D Gaussian peak calling; Donut kernel convolution for loop enrichment; 3D polymer simulation constrained by Hi-C.
- **Datasets:** 4DN (4D Nucleome) Data Portal — Hi-C across cell types and time (https://data.4dnucleome.org/); ENCODE Hi-C datasets — cell-line 3D contacts (https://www.encodeproject.org/); GEO Hi-C studies (GSE63525 Rao 2014 etc.) (https://www.ncbi.nlm.nih.gov/geo/); OpenChromatin Consortium ATAC/Hi-C (https://www.ncbi.nlm.nih.gov/geo/).
- **Starter repos/tools:** Higashi (https://github.com/ma-compbio/Higashi) — single-cell Hi-C GPU-accelerated hypergraph model; HiCCUPS (part of Juicer, https://github.com/aidenlab/juicer) — GPU-accelerated loop caller; ChromaFold (https://www.nature.com/articles/s41467-024-53628-0) — GPU CNN for contact prediction; cooler (https://github.com/open2c/cooler) — cool format Hi-C I/O (CPU, GPU matrix ops as next step).
- **CUDA libraries & GPU pattern:** cuSPARSE for sparse ICE/KR matrix balancing; cuBLAS for dense compartment eigendecomposition; custom 2D convolution kernels (HiCCUPS); cuDNN for CNN-based contact prediction; GPU-resident contact matrix as CSR/CSC sparse format.

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
