# THEORY — 14.8 Real-Time Genomic Pathogen Surveillance Networks

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

### 14.8 Real-Time Genomic Pathogen Surveillance Networks 🟡 · Active R&D

- **Deep dive:** Epidemic genomic surveillance sequences thousands of viral/bacterial isolates per day, requiring near-real-time genome assembly, variant calling, phylogenetic placement, and transmission cluster detection. GPU-accelerated genome assembly (GPU-MEGAHIT) and variant calling (GPU Parabricks) reduce per-sample analysis from hours to minutes, enabling next-flight sequencing decisions during outbreak response. Phylogenetics on GPU (iqtree GPU, PhyML-CUDA) computes maximum likelihood trees on thousands of taxa. Real-time cluster detection via GPU-accelerated pairwise SNP distance matrices (all-vs-all on N×N matrix) parallelizes naturally over GPU threads.
- **Key algorithms:** GPU-accelerated de novo assembly (BWT-based, de Bruijn graph), GPU variant calling (Parabricks Haplotypecaller), maximum likelihood phylogenetics (GTR+Γ model), pairwise SNP distance matrix, Bayesian temporal phylogenetics (BEAST GPU backend), epidemic growth rate estimation (SEIR model on GPU).
- **Datasets:** GISAID EpiCoV — 17M+ SARS-CoV-2 genomes (https://gisaid.org/); NCBI SRA — all short-read sequencing submissions (https://www.ncbi.nlm.nih.gov/sra); Nextstrain builds — curated SARS-CoV-2 / influenza phylogenies (https://nextstrain.org/); PHA4GE pathogen genomics standards datasets (https://pha4ge.org/).
- **Starter repos/tools:** NVIDIA Clara Parabricks (https://www.nvidia.com/en-us/clara/parabricks/) — GPU genome assembly/variant calling (40× speedup over GATK); Nextstrain (https://github.com/nextstrain/ncov) — phylogenetic outbreak analysis pipeline; IQ-TREE (https://github.com/Cibiv/IQ-TREE) — ML phylogenetics (multi-GPU via CUDA); GPU-MEGAHIT (https://github.com/GPU-MEGAHIT/GPU-MEGAHIT) — GPU-accelerated metagenomics assembly.
- **CUDA libraries & GPU pattern:** CUDA BWT for GPU read alignment (BWA-MEM on CUDA), cuBLAS for SNP distance matrix computation, cuFFT for k-mer frequency analysis; pattern: raw reads → GPU assembly → GPU variant calling → pairwise SNP matrix on GPU → transmission cluster detection → phylogenetic placement → epidemiological alert.

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
