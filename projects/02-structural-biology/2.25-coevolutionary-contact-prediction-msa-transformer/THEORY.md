# THEORY — 2.25 Coevolutionary Contact Prediction & MSA Transformer

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

### 2.25 Coevolutionary Contact Prediction & MSA Transformer 🟡 · Active R&D

- **Deep dive:** Coevolutionary analysis of MSAs (correlated mutations between residue positions) reveals protein contact maps that drive structure prediction. EVcouplings uses PLMC (pseudolikelihood-maximized direct coupling analysis) — an L×L matrix inversion and optimization problem where L is sequence length. GPU acceleration via direct CUDA implementation or PyTorch autograd parallelizes the DCA learning over position pairs. MSA Transformer (ESM-MSA-1b) processes MSA rows and columns via tied axial attention on GPU, producing contact predictions and rich evolutionary embeddings for downstream tasks.
- **Key algorithms:** Mutual information (MI), Direct Coupling Analysis (DCA), pseudolikelihood-maximized DCA (PLMC), message-passing DCA (mpDCA), MSA Transformer (axial row/column attention), coevolutionary coupling score to contact map.
- **Datasets:** UniRef50/UniRef90 for MSA construction (https://www.uniprot.org); Pfam MSA database (https://pfam.xfam.org); EVcouplings benchmark contact sets (https://github.com/debbiemarkslab/EVcouplings); CASP14 contact prediction benchmarks (https://predictioncenter.org).
- **Starter repos/tools:** EVcouplings (https://github.com/debbiemarkslab/EVcouplings) — DCA-based coevolutionary analysis; ESM-MSA-1b (https://github.com/facebookresearch/esm) — GPU MSA Transformer; CCMpred (https://github.com/soedinglab/CCMpred) — GPU-accelerated DCA with CUDA implementation; HHpred (https://toolkit.tuebingen.mpg.de/tools/hhpred) — profile-profile alignment for MSA.
- **CUDA libraries & GPU pattern:** CCMpred custom CUDA kernels for DCA gradient computation; cuBLAS for L×L coupling matrix products; PyTorch CUDA axial attention for MSA Transformer; GPU-parallel MSA column featurization.

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
