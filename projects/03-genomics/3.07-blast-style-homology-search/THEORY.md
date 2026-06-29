# THEORY — 3.7 BLAST-Style Homology Search

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

### 3.7 BLAST-Style Homology Search 🟢 · Established
- **Deep dive:** Homology search finds sequences in a database that are evolutionarily related to a query, using seed-filter-extend logic (BLAST) or k-mer prefiltering + ungapped alignment (MMseqs2 / DIAMOND). At the scale of AlphaFold2 structure prediction (MSA search dominates 70–90% of total inference time), GPU acceleration is transformative. MMseqs2-GPU (2025, Nature Methods) replaces the CPU k-mer prefilter with a GPU-parallel gapless scoring pass across all database sequences simultaneously, achieving 20× speedup and 71× cost reduction vs. 128-core CPU. The bottleneck parallelised is the embarrassingly parallel pairwise k-mer match scanning across millions of database sequences per query batch.
- **Key algorithms:** K-mer prefilter seeding; gapless diagonal scoring; Smith-Waterman extension (affine gaps); profile-profile scoring (PSI-BLAST); iterative profile construction; DIAMOND's double-indexed seed matching.
- **Datasets:** UniRef50/90 — clustered UniProt sequences for homology (https://www.uniprot.org/help/uniref); NCBI nr protein database (https://ftp.ncbi.nlm.nih.gov/blast/db/); PDB70 — representative PDB sequences (https://www.rcsb.org/downloads); Pfam — protein family HMM database (https://www.ebi.ac.uk/interpro/download/).
- **Starter repos/tools:** MMseqs2 + GPU branch (https://github.com/soedinglab/MMseqs2) — official repo with GPU support in 2025 release; DIAMOND (https://github.com/bbuchfink/diamond) — ultra-fast protein aligner (CPU baseline); CUDASW4 (https://github.com/asbschmidt/CUDASW4) — full SW on GPU for deep alignments; NVIDIA NIM MMseqs2 microservice (https://developer.nvidia.com/blog/accelerated-sequence-alignment-for-protein-design-with-mmseqs2-and-nvidia-nim/) — cloud-API GPU search.
- **CUDA libraries & GPU pattern:** Custom CUDA gapless scoring kernels (one warp per query-target pair); batched SW extension with shared memory; GPU hash table for seed look-ups; multi-GPU data parallelism across database shards; CUDA streams for overlapping I/O and compute.

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
