# THEORY — 2.13 MSA Generation Acceleration

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

### 2.13 MSA Generation Acceleration 🟡 · Active R&D

- **Deep dive:** Multiple sequence alignment (MSA) construction for AlphaFold2 is a major bottleneck: HHblits and Jackhmmer search the UniRef90 database (210GB) requiring hours of CPU time. GPU acceleration of profile hidden Markov model (HMM) search is an active area: GPU-HMMER uses CUDA to parallelize the Viterbi/Forward-Backward dynamic programming recursion over thousands of sequence targets simultaneously. Accelerating MSA generation could remove one of the last CPU-bound steps in the AF2 prediction pipeline, enabling rapid large-scale proteome annotation.
- **Key algorithms:** Profile HMM Viterbi algorithm, Forward-Backward DP, Smith-Waterman alignment, position-specific scoring matrix (PSSM) search, k-mer seed hashing, HHblits iterated profile-profile alignment.
- **Datasets:** UniRef90 — 210GB protein sequence database (https://www.uniprot.org/help/uniref); UniClust30 (https://uniclust.mmseqs.com); MGnify metagenomics sequences (https://www.ebi.ac.uk/metagenomics/); BFD — Big Fantastic Database (https://bfd.mmseqs.com).
- **Starter repos/tools:** MMseqs2 (https://github.com/soedinglab/MMseqs2) — ultra-fast protein search and clustering (GPU-capable via SIMD/GPU versions); ColabFold MSA server (https://github.com/sokrypton/ColabFold) — GPU-accelerated MSA for AlphaFold2; GPU-HMMER (verify URL) — CUDA Viterbi HMM search; Linclust (https://github.com/soedinglab/MMseqs2) — GPU-accelerated sequence clustering.
- **CUDA libraries & GPU pattern:** CUDA DP recursion for HMM Viterbi (row-parallel); GPU parallel Smith-Waterman via CUDASW++; warp-parallel query-vs-target scoring; GPU hash tables for k-mer seed lookup.

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
