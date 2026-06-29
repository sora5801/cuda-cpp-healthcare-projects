# THEORY — 3.28 Profile HMM (Viterbi / Forward)

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

### 3.28 Profile HMM (Viterbi / Forward) 🟡 · Active R&D
- **Deep dive:** Profile HMMs (pHMMs) model protein families as position-specific probability distributions; HMMER3 searches databases by applying a cascade: MSV/SSV (Multi-Segment Viterbi) filter, P7Viterbi, and Forward-Backward scoring. MSV/SSV alone consumes 72% of runtime. CUDAMPF parallelises the MSV/Viterbi recurrence across database sequences: each CUDA thread block processes one query-profile versus one database sequence, computing the N×M score matrix in shared memory. For very deep database scans (>10⁹ sequences in metagenomics), GPU pHMM search reduces days to hours.
- **Key algorithms:** MSV/SSV Multi-Segment Viterbi; P7Viterbi DP over profile-sequence grid; Forward-Backward algorithm (sum-product); Viterbi traceback; plan-7 profile HMM architecture; hit reporting with E-value calculation.
- **Datasets:** Pfam-A — 20 k protein family profiles (https://www.ebi.ac.uk/interpro/download/); UniRef50 — protein sequences for database search (https://www.uniprot.org/help/uniref); Rfam — RNA family profiles (https://rfam.org/); JGI metagenome proteins — environmental pHMM targets (https://genome.jgi.doe.gov/).
- **Starter repos/tools:** CUDAMPF (https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-016-0946-4) — multi-tiered CUDA HMMER acceleration; HMMER3 (https://github.com/EddyLab/hmmer) — CPU reference, CUDA port target; MMseqs2 profile search (https://github.com/soedinglab/MMseqs2) — faster alternative using k-mer prefilter; GPU-HMMER speculative search (verify URL) — speculative HMMER implementation on GPU.
- **CUDA libraries & GPU pattern:** Custom shared-memory MSV/Viterbi kernel (one block per sequence); vectorised score matrix with CUDA float4; CUB warp-level max for Viterbi path; multi-GPU sequence database partitioning; CUDA streams for I/O and compute overlap.

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
