# THEORY — 12.14 Peptide De Novo Sequencing

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

### 12.14 Peptide De Novo Sequencing 🟡 · Active R&D
- **Deep dive:** De novo peptide sequencing infers amino acid sequences directly from MS/MS spectra without a protein database, critical for non-model organisms, immunopeptidomics, and modified peptides. Algorithms generate candidate sequences by traversing a spectrum graph (nodes = fragment ions, edges = amino acid mass differences) via beam search or dynamic programming. GPU acceleration applies to: (1) GPU-parallel beam search over thousands of candidate sequences simultaneously, (2) batched transformer/LSTM scoring of candidate sequences, and (3) the CUDA-accelerated knapsack DP ensuring precursor mass consistency. NovoBench (NeurIPS 2024) benchmarks GPU-accelerated deep learning de novo sequencers.
- **Key algorithms:** Spectrum graph construction (b/y-ion nodes); beam-search decoding with GPU-parallel branches; CUDA knapsack DP for precursor mass constraint; seq2seq transformer (Casanovo, PointNovo); bidirectional LSTM encoder; attention over fragment ion sequence; PTM-tolerant open search.
- **Datasets:** PRIDE ProteomeXchange benchmark de novo datasets (https://www.ebi.ac.uk/pride/); NovoBench benchmark (https://github.com/jingbo02/NovoBench) — standardised deep learning de novo benchmark; MassIVE (https://massive.ucsd.edu/); PeptideAtlas synthetic peptide datasets (https://www.peptideatlas.org/).
- **Starter repos/tools:** Casanovo (https://github.com/Noble-Lab/casanovo) — transformer-based GPU de novo sequencer; NovoBench (https://github.com/jingbo02/NovoBench) — NeurIPS 2024 benchmark suite; PointNovo (verify URL, from Ma et al.) — deep learning de novo with GPU inference; DeepNovo (https://github.com/nh2tran/DeepNovo) — original LSTM-based GPU de novo sequencer.
- **CUDA libraries & GPU pattern:** cuDNN transformer/LSTM inference; CUDA knapsack DP (shared-memory DP table per spectrum); batched beam search with GPU-parallel candidate scoring; Tensor Core BF16 for transformer scoring; one CUDA stream per spectrum batch.

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
