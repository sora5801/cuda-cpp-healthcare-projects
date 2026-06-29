# THEORY — 3.26 GPU BAM Sorting & Deduplication

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

### 3.26 GPU BAM Sorting & Deduplication 🟡 · Active R&D
- **Deep dive:** Post-alignment BAM sorting (by genomic coordinate) and duplicate read marking are canonical bottlenecks in sequencing pipelines processing terabyte-scale BAM files. Coordinate sort is a radix sort on (chromosome, position, strand) keys; GPU radix sort via thrust achieves far higher throughput than samtools CPU sort. Duplicate marking requires grouping reads by (start, end, orientation) and keeping only the highest-base-quality copy; this is a parallel hash-aggregation problem ideal for GPU hash maps. Parabricks integrates GPU sort and markdup in its fq2bam tool, running in the same 6-minute wall time as the alignment step by overlapping GPU sort with alignment I/O.
- **Key algorithms:** Radix sort by (chromosome, position) key; hash-based read grouping for duplicate detection; Picard MarkDuplicates scoring (sum base quality); UMI-aware duplicate collapsing; coordinate index (BAI/CSI) construction via parallel prefix.
- **Datasets:** 1000 Genomes WGS BAM archives (https://www.internationalgenome.org/data); TCGA cancer WGS BAM files (https://portal.gdc.cancer.gov/); ENCODE ChIP-seq BAM (https://www.encodeproject.org/); ICGC PCAWG BAMs (https://dcc.icgc.org/).
- **Starter repos/tools:** Parabricks fq2bam / bamsort (https://docs.nvidia.com/clara/parabricks/latest/) — integrated GPU BAM sort + dedup; biobambam2 (https://github.com/gt1/biobambam2) — CPU sort/dedup reference with parallel threads; Samtools (https://github.com/samtools/samtools) — CPU BAM toolkit; FastDup (https://arxiv.org/pdf/2505.06127) — speculation-and-test GPU duplicate marking.
- **CUDA libraries & GPU pattern:** thrust::sort_by_key for radix coordinate sort; GPU robin-hood hash map for duplicate grouping; thrust::reduce_by_key for per-group best-quality selection; CUDA managed memory for BAM record streaming; multi-GPU shard-and-merge pattern.

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
