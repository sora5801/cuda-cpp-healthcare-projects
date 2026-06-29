# THEORY — 3.16 Sequence Error Correction

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

### 3.16 Sequence Error Correction 🟡 · Active R&D
- **Deep dive:** Error correction removes sequencing artefacts before assembly. For short reads, the dominant method is k-mer spectrum analysis: k-mers below a coverage threshold are likely errors; correcting a base changes the read k-mer into a trusted one. For long reads (ONT, PacBio CLR), self-correction aligns multiple raw reads against each other and computes a consensus. CARE (https://github.com/fkallen/CARE) is a CUDA-accelerated short-read error corrector that keeps the k-mer hash table in GPU memory and processes millions of reads per second. GPU-accelerated partial-order alignment (POA) for long-read correction is implemented in GenomeWorks racon-GPU.
- **Key algorithms:** K-mer spectrum analysis (trusted-k-mer correction); Bloom filter for inexact k-mer membership; multiple sequence alignment (POA / MSA) for long-read consensus; BFC (BWT-based correction); de Bruijn graph compaction for error pruning; expectation-maximisation for error model learning.
- **Datasets:** GAGE short-read datasets — benchmark reads with known errors (http://gage.cbcb.umd.edu/); GiaB HG001-HG007 — truth-set comparison for corrected reads (https://www.nist.gov/programs-projects/genome-bottle); ONT long-read SRA archives (https://www.ncbi.nlm.nih.gov/sra); PacBio CLR SRA datasets — high-error long reads (https://www.ncbi.nlm.nih.gov/sra).
- **Starter repos/tools:** CARE (https://github.com/fkallen/CARE) — CUDA short-read error corrector, GPU hash tables, Pascal+ required; racon-GPU (https://github.com/NVIDIA-Genomics-Research/racon-gpu) — GPU POA polishing/correction; CONSENT (https://github.com/morispi/CONSENT) — long-read self-correction via local De Bruijn graphs (CPU, GPU POA target); Medaka (https://github.com/nanoporetech/medaka) — RNN-based long-read correction with GPU inference.
- **CUDA libraries & GPU pattern:** GPU hash tables with atomic CAS for k-mer counting; warp-level vote for consensus base determination; cuBLAS / custom GEMM for MSA scoring; one CUDA block per read during POA alignment; batched kernel launches across millions of reads.

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
