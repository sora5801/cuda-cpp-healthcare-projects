# THEORY — 3.9 Phylogenetic Likelihood / Tree Inference

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

### 3.9 Phylogenetic Likelihood / Tree Inference 🟡 · Active R&D
- **Deep dive:** Maximum-likelihood phylogenetic inference evaluates the Felsenstein pruning recursion—computing site likelihood at each internal node by multiplying branch transition probability matrices (4×4 or 20×20 per site, per node) up the tree—for millions of alignment columns and hundreds of tree search moves (NNI, SPR). For large trees (thousands of taxa, genome-scale alignments), the log-likelihood computation is the bottleneck and is embarrassingly parallel across alignment sites. Bayesian phylogenetics (MrBayes) runs thousands of MCMC steps each requiring full-tree likelihood evaluation; GPU acceleration reported 63× speedup vs. serial CPU by assigning each site to a thread. RAxML-NG and IQ-TREE GPU are active development targets.
- **Key algorithms:** Felsenstein pruning / Felsinstein's pruning recursion; substitution model matrix exponentiation (GTR, WAG, LG); nearest-neighbor interchange (NNI) and subtree pruning/regrafting (SPR) tree search; Metropolis-Hastings MCMC (Bayesian); bootstrap resampling.
- **Datasets:** TreeBASE — curated phylogenetic alignments and trees (https://www.treebase.org/); SILVA rRNA database — large rRNA alignment for phylogenetics (https://www.arb-silva.de/); NCBI CDD — conserved domain alignments (https://www.ncbi.nlm.nih.gov/Structure/cdd/cdd.shtml); OpenTreeOfLife — aggregated phylogenetic data (https://opentreeoflife.github.io/).
- **Starter repos/tools:** IQ-TREE 2 (https://iqtree.github.io/) — state-of-the-art ML tree inference (GPU extension in development); RAxML-NG (https://github.com/amkozlov/raxml-ng) — fast ML inference with GPU acceleration hooks; MrBayes (https://github.com/NBISweden/MrBayes) — Bayesian inference with CUDA-accelerated site likelihood; BeagleLib (https://github.com/beagle-dev/beagle-lib) — GPU-accelerated phylogenetic likelihood library used by MrBayes/BEAST.
- **CUDA libraries & GPU pattern:** BeagleLib uses custom CUDA kernels for 4×4/20×20 matrix-vector products per site per node; one CUDA thread per alignment site within a likelihood pass; cuBLAS for transition matrix exponentiation; multi-GPU over tree partitions.

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
