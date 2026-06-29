# THEORY — 12.13 Real-Time Pathogen Identification (Clinical)

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

### 12.13 Real-Time Pathogen Identification (Clinical) 🟡 · Active R&D
- **Deep dive:** Clinical metagenomic next-generation sequencing (mNGS) for pathogen identification requires processing millions of reads within 1–2 hours of sample collection to guide antibiotic therapy. The critical path is: GPU basecalling (Dorado) → GPU k-mer classification (MetaCache/GPU Kraken2) → GPU AMR gene annotation (DIAMOND vs. CARD) → statistical confidence scoring. A 2024 MDPI paper describes a GPU-integrated nanopore workstation running CUDA-accelerated basecalling and classification in real time, enabling same-day bloodstream infection pathogen identification. GPU parallelism is the enabling technology for clinical mNGS turnaround within therapeutic decision windows.
- **Key algorithms:** GPU CTC basecalling; GPU k-mer LCA classification; minimap2 GPU alignment to pathogen reference panel; GPU AMR gene DIAMOND search; Bayesian abundance estimation (Bracken); clinical decision threshold scoring; antimicrobial susceptibility genotype prediction.
- **Datasets:** NCBI RefSeq pathogen reference sequences (https://ftp.ncbi.nlm.nih.gov/refseq/); CARD AMR database (https://card.mcmaster.ca/); IDseq / Chan Zuckerberg clinical mNGS data (https://czid.org/); NCBI Pathogen Detection (https://www.ncbi.nlm.nih.gov/pathogens/).
- **Starter repos/tools:** Dorado (https://github.com/nanoporetech/dorado) — GPU basecaller; MetaCache-GPU (https://arxiv.org/pdf/2106.08150) — GPU real-time classification; DIAMOND (https://github.com/bbuchfink/diamond) — fast AMR gene annotation; CZID/IDseq (https://github.com/chanzuckerberg/czid-workflows) — cloud mNGS pipeline.
- **CUDA libraries & GPU pattern:** TensorRT for low-latency basecalling; GPU hash tables for k-mer classification; CUDA streams for read-by-read pipeline; cuBLAS for alignment score matrices; real-time CUDA ring buffer for streaming POD5 signal; multi-GPU pipelining of basecall → classify → annotate.

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
