# THEORY — 3.3 Variant Calling Acceleration

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

### 3.3 Variant Calling Acceleration 🟢 · Established
- **Deep dive:** Germline variant calling applies the Haplotype Caller algorithm: local de novo assembly of active regions, PairHMM forward-algorithm computation of read-haplotype likelihoods, and genotype likelihood calculation. PairHMM is by far the dominant runtime cost—each read must be compared against every candidate haplotype via an O(R×H) DP table. GPU parallelism fills an entire PairHMM table per thread block, running thousands of read-haplotype pairs simultaneously. Parabricks GPU HaplotypeCaller reduces 30× WGS germline calling from ~9 hours CPU to under 10 minutes on an H100 using GATK-identical math. DeepVariant's CNN pileup scoring is a further candidate for batched GPU inference.
- **Key algorithms:** PairHMM forward algorithm; local de novo assembly (De Bruijn graph over active regions); Viterbi realignment; genotype likelihood calculation (GL/PL); base quality score recalibration (BQSR); DeepVariant convolutional inference.
- **Datasets:** GiaB truth sets HG001–HG007 — gold-standard variant calls for benchmarking (https://www.nist.gov/programs-projects/genome-bottle); ClinVar — clinically interpreted variants (https://www.ncbi.nlm.nih.gov/clinvar/); gnomAD v4 — population allele frequencies (https://gnomad.broadinstitute.org/); 1000 Genomes high-coverage WGS (https://www.internationalgenome.org/data).
- **Starter repos/tools:** NVIDIA Parabricks HaplotypeCaller / DeepVariant module (https://docs.nvidia.com/clara/parabricks/latest/) — GATK-identical GPU variant calling; DeepVariant (https://github.com/google/deepvariant) — CNN-based caller deployable on GPU; GATK (https://github.com/broadinstitute/gatk) — CPU reference for parity testing; Clairvoyante / Clair3 (https://github.com/HKU-BAL/Clair3) — deep learning variant caller with GPU inference.
- **CUDA libraries & GPU pattern:** cuDNN (DeepVariant CNN inference); custom PairHMM CUDA kernels with one block per read-haplotype pair; shared-memory DP tables; multi-GPU pipeline parallelism (BQSR → alignment → calling); CUDA streams for pipelining I/O and compute.

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
