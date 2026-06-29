# THEORY — 11.8 CRISPR System Design & Modeling

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

### 11.8 CRISPR System Design & Modeling 🟡 · Active R&D

- **Deep dive:** CRISPR guide RNA (gRNA) design requires genome-wide off-target site enumeration (all 20-mer matches with ≤4 mismatches in 3 billion base pairs), scoring each off-target's likelihood based on mismatch position and type. GPU-accelerated exact string matching (GPU BWT/FM-index) reduces the genome scanning from hours to minutes. Deep learning off-target predictors (CNN, BiGRU, BERT-based LLMs) run on GPU over millions of candidate gRNAs in parallel. The CRISOT tool suite derives RNA-DNA molecular interaction fingerprints from GPU-accelerated MD simulations of Cas9-gRNA-DNA ternary complexes to compute structural off-target scores.
- **Key algorithms:** FM-index / BWT genome search on GPU, CNN/BiGRU/Transformer off-target classifiers, molecular dynamics of Cas9 R-loop formation, energy minimization for gRNA thermodynamic stability, seqmap-style GPU hash table for rapid k-mer matching, CRISOT molecular fingerprinting.
- **Datasets:** CRISPOR Guide RNA Dataset — experimentally validated on/off-target activities (https://crispor.tefor.net/); CIRCLE-seq Off-Target Dataset (Tsai et al., Nature Methods) — unbiased off-target identification; Genome-wide CRISPR off-target benchmark (https://www.nature.com/articles/s41467-023-42695-4); ClinVar — disease-relevant on-target loci for therapeutic gRNA selection (https://www.ncbi.nlm.nih.gov/clinvar/).
- **Starter repos/tools:** CRISPOR (https://github.com/maximilianh/crisporWebsite) — GPU-accelerated guide design pipeline; CRISPRscan (https://www.crisprscan.org/) — on/off-target prediction (verify GitHub URL); DeepCRISPR (https://github.com/jieccccc/DeepCRISPR) — CNN off-target prediction with GPU inference; GROMACS (https://github.com/gromacs/gromacs) — GPU MD of Cas9 R-loop for CRISOT-style fingerprinting.
- **CUDA libraries & GPU pattern:** CUDA BWT string index for GPU genome scanning, cuDNN for CNN/Transformer off-target scoring over batches of gRNAs, cuRAND for MD trajectory generation; pattern: 20-mer gRNA → GPU BWT scan of genome → candidate off-target list → batch GPU DL scoring → filter by specificity score → MD fingerprint for top candidates.

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
