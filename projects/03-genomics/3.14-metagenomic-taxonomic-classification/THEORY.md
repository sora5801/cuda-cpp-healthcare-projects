# THEORY — 3.14 Metagenomic Taxonomic Classification

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

### 3.14 Metagenomic Taxonomic Classification 🟡 · Active R&D
- **Deep dive:** Metagenomic classification assigns every sequencing read to a taxon by matching k-mers against a database of reference genomes (Kraken2 uses an exact k-mer LCA hash map; Centrifuge uses FM-index). At clinical sequencing throughput (millions of reads/minute), the hash look-up and LCA traversal become the bottleneck. MetaCache-GPU parallelises the k-mer-to-taxon hash look-up on the GPU, batching thousands of reads simultaneously, each read's k-mers queried via parallel hash table probes. Real-time GPU classification is critical for point-of-care diagnostics and pandemic surveillance.
- **Key algorithms:** K-mer exact hash matching (Kraken2 minimiser-LCA); lowest common ancestor (LCA) traversal; FM-index backward search (Centrifuge); Jaccard / MinHash distance (Mash Screen); Clark discriminative k-mer selection; GPU cuckoo hash table probing.
- **Datasets:** NCBI RefSeq complete microbial genomes — standard Kraken2 database (https://ftp.ncbi.nlm.nih.gov/refseq/); CAMI challenge benchmark datasets — simulated metagenomes (https://data.cami-challenge.org/); HMP (Human Microbiome Project) reads (https://www.hmpdacc.org/); SRA metagenomics projects (https://www.ncbi.nlm.nih.gov/sra).
- **Starter repos/tools:** MetaCache-GPU (https://arxiv.org/pdf/2106.08150) — GPU k-mer classification, ultra-fast; Kraken2 (https://github.com/DerrickWood/kraken2) — CPU reference, GPU hash port target; Centrifuge (https://github.com/DaehwanKimLab/centrifuge) — FM-index based, GPU extension possible; Bracken (https://github.com/jenniferlu717/Bracken) — Bayesian abundance re-estimation downstream of Kraken2.
- **CUDA libraries & GPU pattern:** Custom GPU cuckoo / robin-hood hash tables for k-mer look-up; thrust for k-mer sort and dedup; atomic CAS for concurrent hash insertions; one CUDA thread block per read, threads per k-mer; persistent kernel pattern for streaming reads.

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
