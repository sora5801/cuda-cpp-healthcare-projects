# THEORY — 3.27 Suffix Array / BWT / FM-Index Construction

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

### 3.27 Suffix Array / BWT / FM-Index Construction 🟡 · Active R&D
- **Deep dive:** The BWT (Burrows-Wheeler Transform) and its associated FM-index enable sub-linear text search and are the backbone of short-read aligners (BWA, Bowtie2), assemblers (string graphs), and text compression. Constructing the BWT of a 3 Gb genome involves building the suffix array (SA) then applying the BWT permutation. GPU suffix array construction via parallel prefix-doubling achieves 7.9× speedup over prior GPU skew algorithms, with all n suffixes sorted simultaneously using (log n) radix-sort rounds. At metagenomics or pangenome scale (terabases), GPU construction of a BWT over millions of reads (Big-BWT / ropebwt2) is a research frontier, with CUDA CUDPP's parallel BWT used as a primitives baseline.
- **Key algorithms:** Prefix-doubling suffix array construction (DC3/skew algorithm adapted for GPU); radix sort by 2k-character rank pairs; Burrows-Wheeler permutation; FM-index backward step (LF mapping); wavelet tree construction for rank/select; Big-BWT external-memory algorithm.
- **Datasets:** GRCh38 human reference genome — 3 Gb target for BWT construction (https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.40/); 1000 Genomes read collections for pan-read BWT (https://www.internationalgenome.org/data); NCBI RefSeq complete microbial genomes (https://ftp.ncbi.nlm.nih.gov/refseq/); Human Pangenome sequences for pan-BWT (https://humanpangenome.org/).
- **Starter repos/tools:** GPU suffix array prefix-doubling (https://www.researchgate.net/publication/303594470) — fast parallel SA construction on GPU; ropebwt2 (https://github.com/lh3/ropebwt2) — incremental BWT construction (CPU, GPU K40 tested); CUDPP BWT (https://devblogs.nvidia.com/cutting-edge-parallel-algorithms-research-cuda/) — CUDA Data Parallel Primitives BWT; Big-BWT (https://github.com/alshai/Big-BWT) — external-memory BWT for terabase strings.
- **CUDA libraries & GPU pattern:** thrust::sort_by_key for radix-sort based SA construction; parallel prefix sums (CUB) for rank array update; GPU-resident suffix-rank arrays; custom LF-mapping kernel; persistent warp pattern for backward search.

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
