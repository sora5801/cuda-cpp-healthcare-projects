# THEORY — 3.2 Short-Read Mapping / Alignment

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

### 3.2 Short-Read Mapping / Alignment 🟢 · Established
- **Deep dive:** Short-read mapping (50–300 bp Illumina reads) first seeds candidate positions in a reference genome index (FM-index or hash table), then extends seeds with banded SW. At whole-genome scale (30× coverage ≈ 900 M reads for human), the seed-extension and CIGAR-string generation steps dominate runtime. GPU acceleration batches thousands of read-to-reference extensions simultaneously, each assigned a CUDA thread block with shared-memory score matrix, while FM-index backward search runs as a parallel BFS across thread groups. NVIDIA Parabricks (v4.7, 2025) completes a 30× WGS in under 10 minutes on an H100, vs. >30 hours CPU BWA-MEM, by reimplementing BWA-MEM's seed-chain-extend pipeline in CUDA.
- **Key algorithms:** FM-index / BWT backward search; seed chaining (sparse DP); banded Smith-Waterman extension; CIGAR encoding; markduplicates hashing; Burrows-Wheeler transform; seeding by minimisers.
- **Datasets:** 1000 Genomes Project — 2504 human WGS samples, short reads (https://www.internationalgenome.org/data); Genome in a Bottle (GiaB) NA12878 / HG002 — benchmark short-read WGS datasets (https://www.nist.gov/programs-projects/genome-bottle); SRA FASTQ archives — petabyte-scale short reads (https://www.ncbi.nlm.nih.gov/sra); ENCODE ChIP/RNA-seq FASTQs — curated short-read functional data (https://www.encodeproject.org/).
- **Starter repos/tools:** NVIDIA Parabricks (https://docs.nvidia.com/clara/parabricks/latest/) — GPU-accelerated BWA-MEM + GATK pipeline, 50× faster than CPU; CUSHAW2-GPU (https://github.com/asbschmidt/CUSHAW3) — banded SW seed extension on GPU; Scrooge (https://github.com/CMU-SAFARI/Scrooge) — GPU/CPU co-designed aligner; GenomeWorks (https://github.com/NVIDIA-Genomics-Research/GenomeWorks) — pairwise overlap kernels underpinning mapping.
- **CUDA libraries & GPU pattern:** cuSPARSE (index look-ups); thrust (sorting seeds); custom banded-SW kernels with shared-memory tiling; persistent warp-per-read extension; multi-GPU data parallelism via NCCL.

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
