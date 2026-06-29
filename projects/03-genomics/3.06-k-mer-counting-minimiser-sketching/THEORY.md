# THEORY — 3.6 k-mer Counting & Minimiser Sketching

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

### 3.6 k-mer Counting & Minimiser Sketching 🟢 · Established
- **Deep dive:** k-mer counting determines the frequency of every length-k substring in a read set, foundational to genome-size estimation, error detection, assembly, and metagenomics. For a 30× human genome (~270 Gb of sequence, k=21), the table has ~4 billion distinct k-mers; efficient parallel hashing and atomic counting saturate GPU memory bandwidth. Gerbil uses GPU-resident hash tables and achieves >10× speed over Jellyfish. Minimiser sketching (selecting a canonical subset of k-mers per window) reduces data by ~5× and enables the MinHash / HyperMinHash distance computations used in species typing; all operations parallelise across reads with one GPU thread per minimiser.
- **Key algorithms:** Radix-sort-based k-mer canonicalisation; GPU hash table with cuckoo / Robin Hood probing; count-min sketch for approximate counting; minimiser extraction (window function); MinHash / Jaccard distance estimation; HyperLogLog cardinality estimation.
- **Datasets:** Illumina WGS of NA12878 — human reference dataset (https://www.ncbi.nlm.nih.gov/sra/SRR622457); GAGE benchmark — multi-species short reads for assembly tools (http://gage.cbcb.umd.edu/); GenomeTrakr pathogen WGS — bacterial surveillance reads (https://www.ncbi.nlm.nih.gov/bioproject/PRJNA183844); Sequence Read Archive (SRA) — global repository (https://www.ncbi.nlm.nih.gov/sra).
- **Starter repos/tools:** Gerbil (https://github.com/uni-halle/gerbil) — k-mer counter with GPU support; KMC3 (https://github.com/refresh-bio/KMC) — disk-I/O efficient CPU k-mer counter (GPU comparison baseline); Jellyfish (https://github.com/gmarcais/Jellyfish) — lock-free hash k-mer counter; GenomeScope2 (https://github.com/tbenavi1/genomescope2.0) — genome profiling from k-mer histograms.
- **CUDA libraries & GPU pattern:** CUDA atomic operations (atomicAdd for count tables); thrust::sort_by_key for radix sort; warp-level ballot and shuffle for minimiser window reduction; cuRAND for sketch initialisation.

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
