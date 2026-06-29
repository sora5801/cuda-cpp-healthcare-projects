# THEORY — 3.1 Smith-Waterman / Needleman-Wunsch Alignment

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

### 3.1 Smith-Waterman / Needleman-Wunsch Alignment 🟢 · Established
- **Deep dive:** Smith-Waterman (SW) computes the optimal local alignment between two sequences via a dynamic-programming (DP) score matrix filled cell-by-cell; at protein-database scale this means quadratic work per query against millions of targets. GPUs collapse this into anti-diagonal wavefront parallelism: all cells on the same anti-diagonal are independent and can be computed simultaneously across thousands of CUDA threads, eliminating the serial dependency that cripples CPUs. CUDASW++4.0 (2024) achieves up to 5.71 TCUPS on an H100 by exploiting Hopper's DPX integer-DP instructions, hardware-native to the architecture, alongside tile-based matrix partitioning and sequence-database chunking for maximal occupancy. The specific bottleneck parallelised is the per-cell recurrence max(H[i-1,j-1]+s, H[i,j-1]-g, H[i-1,j]-g) across the anti-diagonal frontier.
- **Key algorithms:** Smith-Waterman anti-diagonal DP wavefront; Needleman-Wunsch global DP; striped SIMD inter-sequence parallelism; affine gap scoring; DPX hardware DP instructions (Hopper); sequence-database tiling and batched kernel launch.
- **Datasets:** UniProtKB/Swiss-Prot — curated protein sequence database, ~570 k entries (https://www.uniprot.org/downloads); NCBI nr (non-redundant protein) — comprehensive protein database, 100 M+ sequences (https://ftp.ncbi.nlm.nih.gov/blast/db/); PDB sequences — structural protein sequences for benchmarking alignments (https://www.rcsb.org/downloads); NCBI RefSeq — reference nucleotide and protein sequences (https://ftp.ncbi.nlm.nih.gov/refseq/).
- **Starter repos/tools:** CUDASW4 (https://github.com/asbschmidt/CUDASW4) — CUDASW++4.0, H100/A100/L40S optimised, DPX, up to 5.71 TCUPS; GenomeWorks / ClaraGenomics SDK (https://github.com/NVIDIA-Genomics-Research/GenomeWorks) — NVIDIA CUDA pairwise alignment primitives for both protein and nucleotide; WFA-GPU (verify URL: github.com/quim0/WFA-GPU) — wavefront alignment algorithm on GPU, gap-affine, ultra-fast for long DNA; Parasail (https://github.com/jeffdaily/parasail) — SIMD/CUDA pairwise alignment library used as reference.
- **CUDA libraries & GPU pattern:** cuBLAS (score accumulation); thrust (sort, scan); CUB (warp-level reduction); custom anti-diagonal kernels with shared memory tiling; inter-sequence batching (one CUDA block per query–target pair or striped across warps); DPX integer instructions on Hopper SM90.

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
