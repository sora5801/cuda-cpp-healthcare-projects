# THEORY — 3.23 Splice-Aware RNA Alignment

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

### 3.23 Splice-Aware RNA Alignment 🟡 · Active R&D
- **Deep dive:** Splice-aware aligners (STAR, HISAT2) map RNA-seq reads across exon-exon junctions, requiring the aligner to simultaneously find the best gapped alignment across multi-exon gene models. STAR uses a suffix array for ultra-fast seeding, then extends seeds across splice junctions; HISAT2 uses a graph FM-index encoding known splice sites. GPU acceleration targets the seed-extension step (banded SW across exon pairs) and the loading/querying of the large (28 Gb for STAR human genome) suffix arrays from a GPU-resident or page-locked memory index. For long-read transcriptomics (minimap2 -ax splice), GPU wavefront alignment handles much longer reads across complex splicing.
- **Key algorithms:** Suffix array seeding (STAR); graph HISAT index with splice-site encoded BWT; banded SW for exon extension; maximum-entropy splice-site scoring; CIGAR encoding with N (intron) operations; Hamming-distance seed extension; minimap2 chaining across introns.
- **Datasets:** ENCODE RNA-seq FASTQs (https://www.encodeproject.org/); GENCODE annotation (https://www.gencodegenes.org/); SRA RNA-seq benchmarks (SEQC/MAQC) (https://www.ncbi.nlm.nih.gov/sra); GTEx tissue RNA-seq (https://gtexportal.org/).
- **Starter repos/tools:** STAR (https://github.com/alexdobin/STAR) — fastest spliced RNA aligner (GPU suffix-array querying target); HISAT2 (https://github.com/DaehwanKimLab/hisat2) — graph-index RNA aligner; minimap2 (https://github.com/lh3/minimap2) — long-read splice-aware (GPU wavefront extension target); AGAThA — GPU-accelerated guided sequence alignment for long-read mapping (verify URL).
- **CUDA libraries & GPU pattern:** Page-locked host memory for suffix array loaded by GPU; custom banded-SW CUDA kernels for exon-exon extension; GPU hash tables for splice-junction index; thrust sort for seed clustering; CUDA streams for multi-sample pipelining.

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
