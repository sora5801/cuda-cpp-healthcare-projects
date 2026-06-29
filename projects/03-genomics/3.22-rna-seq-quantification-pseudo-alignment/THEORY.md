# THEORY — 3.22 RNA-seq Quantification / Pseudo-alignment

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

### 3.22 RNA-seq Quantification / Pseudo-alignment 🟢 · Established
- **Deep dive:** Pseudo-alignment (kallisto, Salmon) bypasses full read alignment by mapping k-mers directly to equivalence classes of transcripts, then running the EM algorithm to estimate transcript abundances. GPU acceleration of kallisto redesigns the k-mer compatibility look-up and EM optimisation for GPU throughput: k-mer hash table queries map naturally to parallel GPU hash probes, and the EM update over millions of reads is a dense GEMV. A 2026 study ("RNA-seq analysis in seconds using GPUs," Melsted et al.) demonstrates GPU kallisto completing quantification in seconds vs. minutes on CPU. Salmon's variational Bayes EM is similarly GPU-amenable.
- **Key algorithms:** K-mer de Bruijn graph construction for transcriptome index; pseudoalignment compatibility class assignment; expectation-maximisation (EM) for abundance estimation; variational Bayes EM (Salmon); bootstrap resampling for uncertainty; quasi-mapping hash-based alignment.
- **Datasets:** GENCODE human transcriptome — reference transcript index (https://www.gencodegenes.org/); ENCODE RNA-seq FASTQs — diverse cell-type transcriptomes (https://www.encodeproject.org/); GTEx v9 — tissue RNA-seq compendium (https://gtexportal.org/); SRA RNA-seq studies (https://www.ncbi.nlm.nih.gov/sra).
- **Starter repos/tools:** kallisto GPU branch (https://github.com/pachterlab/kallisto) — GPU branch for pseudo-alignment; Salmon (https://github.com/COMBINE-lab/salmon) — quasi-mapping quantification (GPU EM target); bustools (https://github.com/BUStools/bustools) — BUS file manipulation for scRNA-seq downstream; alevin-fry (https://github.com/COMBINE-lab/alevin-fry) — fast single-cell quantification, GPU-amenable.
- **CUDA libraries & GPU pattern:** GPU hash table for k-mer to equivalence class look-up; custom EM kernel (sparse GEMV per read per EM iteration); warp-level reduction for abundance accumulation; cuSPARSE for sparse equivalence class matrices; CUDA streams for I/O and compute overlap.

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
