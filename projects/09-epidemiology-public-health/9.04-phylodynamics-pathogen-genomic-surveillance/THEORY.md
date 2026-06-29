# THEORY — 9.4 Phylodynamics & Pathogen Genomic Surveillance

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

### 9.4 Phylodynamics & Pathogen Genomic Surveillance 🟡 · Active R&D

- **Deep dive:** Infers the evolutionary and epidemiological history of pathogens from genomic sequences using Bayesian phylodynamic models (BEAST2, TreeTime). The computational bottleneck is evaluating the phylogenetic likelihood across millions of trees sampled by MCMC — each likelihood evaluation requires computing evolutionary substitution probabilities across thousands of sequence sites and tree branches. BEAGLE (Broad-platform Evolutionary Analysis General Likelihood Evaluator) provides a GPU-accelerated library for this core computation, delivering 20–50× speedup over CPU BEAST. GPU-accelerated variant calling pipelines (DNAnexus, NVIDIA Parabricks) feed surveillance outputs into phylodynamic pipelines.
- **Key algorithms:** Bayesian phylogenetic MCMC (Metropolis-Hastings), HKY/GTR nucleotide substitution models, Kingman's coalescent, birth-death diversification models, skyline population size estimation, ancestral state reconstruction, phylogeographic diffusion, TreeTime maximum-likelihood dating.
- **Datasets:**
  - GISAID — 15M+ SARS-CoV-2 and influenza sequences with metadata (https://www.gisaid.org/)
  - NCBI Pathogen Detection Database — real-time foodborne pathogen genomics (https://www.ncbi.nlm.nih.gov/pathogens/)
  - GenBank — nucleotide sequence archive for all pathogens (https://www.ncbi.nlm.nih.gov/genbank/)
  - Nextstrain data pipelines — curated SARS-CoV-2, influenza, mpox builds (https://nextstrain.org/)
- **Starter repos/tools:**
  - BEAST 2 (https://www.beast2.org/) — Bayesian phylogenetic inference; GPU via BEAGLE library
  - BEAGLE (https://github.com/beagle-dev/beagle-lib) — GPU-accelerated phylogenetic likelihood library (CUDA/OpenCL)
  - Nextstrain (https://github.com/nextstrain/augur) — real-time pathogen genomic surveillance pipeline
  - NVIDIA Parabricks (https://github.com/clara-parabricks) — GPU-accelerated variant calling and genome analysis
- **CUDA libraries & GPU pattern:** BEAGLE CUDA kernels for transition probability matrix exponentiation across tree branches, cuBLAS for substitution rate matrix multiplies; pattern: embarrassingly parallel site-likelihood computation across sequence columns, aggregated with parallel prefix products across tree branches.

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
