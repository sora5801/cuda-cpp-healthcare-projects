# THEORY — 3.8 Multiple Sequence Alignment (MSA)

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

### 3.8 Multiple Sequence Alignment (MSA) 🟡 · Active R&D
- **Deep dive:** MSA aligns N sequences simultaneously, core to phylogenetics, variant analysis, and as input to protein structure prediction. Progressive MSA (ClustalW, MAFFT PartTree) first computes an N×N pairwise distance matrix (O(N²) SW comparisons), then builds a guide tree and folds sequences in. On GPU, the distance matrix computation is embarrassingly parallel—each thread block computes one pair—yielding reported 6× speedup for the MAFFT-PartTree distance phase on GPU. CUK-Band (2024) implements center-star MSA on GPU using banded DP. For protein MSA in AlphaFold2 pipelines, MMseqs2-GPU now accelerates the iterative search that builds deep MSAs, the most time-consuming preprocessing step.
- **Key algorithms:** Progressive alignment via guide tree (Neighbor-Joining); center-star alignment reduction; banded Smith-Waterman pairwise DP; profile-profile alignment; Sum-of-Pairs scoring; MAFFT Parttree distance matrix; iterative MSA refinement.
- **Datasets:** BAliBASE — benchmark MSA reference set (https://www.lbgi.fr/balibase/); HomFam — large homologous family MSA benchmark (verify URL); OXFam benchmark (verify URL); Pfam seed alignments (https://www.ebi.ac.uk/interpro/download/).
- **Starter repos/tools:** MAFFT (https://mafft.cbrc.jp/alignment/software/) — fastest large-scale CPU MSA with GPU-accelerated distance phase prototype; CUDA-ClustalW — parallel GPU progressive MSA (https://github.com/topics/multiple-sequence-alignment); CUK-Band (https://link.springer.com/chapter/10.1007/978-981-97-5692-6_8) — 2024 CUDA center-star MSA; MMseqs2 GPU (https://github.com/soedinglab/MMseqs2) — GPU-accelerated MSA search for structure prediction pipelines.
- **CUDA libraries & GPU pattern:** One CUDA thread block per pairwise alignment (distance matrix phase); shared-memory banded DP; thrust for distance matrix sort; cuBLAS GEMM for profile-profile scoring; CUDA streams for guide-tree-ordered batch alignments.

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
