# THEORY — 3.25 Base Quality Score Recalibration (BQSR)

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

### 3.25 Base Quality Score Recalibration (BQSR) 🟢 · Established
- **Deep dive:** BQSR models and corrects systematic machine errors in Illumina base quality scores by regressing quality on covariates: read group, cycle position, sequence context (dinucleotide), and current reported quality. It requires scanning every base of every read (~1 trillion bases for a population study) against a known-variants database, computing covariate tables, then recalibrating scores. NVIDIA Parabricks GPU BQSR reimplements GATK's BaseRecalibrator in CUDA, processing a 30× WGS BQSR step in ~6 minutes on a DGX system vs. 4–9 hours on CPU, by parallelising covariate collection across reads in GPU thread blocks.
- **Key algorithms:** Log-linear regression over quality covariates; covariate table accumulation (parallel prefix sums); known-variant masking via hash look-up; empirical quality recalibration via quantised count table; dbSNP interval tree querying.
- **Datasets:** dbSNP build 155 — known variant positions for masking (https://www.ncbi.nlm.nih.gov/snp/); GiaB known-variant VCFs (https://www.nist.gov/programs-projects/genome-bottle); Mills and 1000G indels — GATK bundle known indels (https://storage.googleapis.com/genomics-public-data/); 1000 Genomes high-coverage WGS (https://www.internationalgenome.org/data).
- **Starter repos/tools:** NVIDIA Parabricks BQSR (https://docs.nvidia.com/clara/parabricks/latest/documentation/tooldocs/man_bqsr.html) — GPU BQSR, GATK-identical output; GATK4 BaseRecalibrator (https://github.com/broadinstitute/gatk) — CPU reference implementation; DeepVariant (https://github.com/google/deepvariant) — alternative CNN caller that bypasses BQSR need; Parabricks fq2bam (https://docs.nvidia.com/clara/parabricks/latest/documentation/tooldocs/man_fq2bam.html) — integrated BWA+BQSR+dedup pipeline.
- **CUDA libraries & GPU pattern:** Parallel covariate table reduction via atomicAdd; GPU hash table for known-variant look-up; shared-memory read buffers; cuBLAS for regression solve; one CUDA thread block per read batch; CUDA streams for pipelined I/O.

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
