# THEORY — 3.17 CRISPR Guide Design & Off-Target Scoring

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

### 3.17 CRISPR Guide Design & Off-Target Scoring 🟡 · Active R&D
- **Deep dive:** Designing effective CRISPR guide RNAs requires genome-wide off-target assessment: every 20-mer protospacer must be compared against all near-matches in the genome (allowing mismatches and bulges). For a 3 Gb human genome, this is ~300 M potential off-target sites per guide; Cas-OFFinder uses GPU to enumerate all combinations of mismatches in parallel. Scoring each off-target for actual cutting probability requires a learned model (CFD score, CNN, transformer), which GPU inference accelerates in batch over all candidate sites. FlashFry precomputes a compressed binary index enabling fast GPU-scalable off-target database look-ups.
- **Key algorithms:** Exact/approximate string matching with bounded mismatches (BFS over mismatch graph); CFD (cutting frequency determination) scoring; CNN/RNN on-target efficiency prediction; protein language model (PLM) for Cas9 variant activity (PLM-CRISPR); off-target enumeration via FM-index or hash-based inexact search.
- **Datasets:** CRISPOR benchmark — validated guide efficiencies and off-targets (https://crispor.gi.ucsc.edu/); GeCKO v2 library — genome-scale CRISPR knockout screen guides (https://www.addgene.org/pooled-library/leczkowski-gecko-v2/); Azimuth / Rule Set 2 training data — published guide efficiency datasets (verify URL); hg38/mm10 reference genomes — for off-target genome scanning (https://genome.ucsc.edu/).
- **Starter repos/tools:** Cas-OFFinder (https://github.com/snugel/cas-offinder) — GPU-accelerated off-target search, mismatch + RNA bulge enumeration; FlashFry (https://github.com/aaronmck/FlashFry) — scalable CRISPR target design with binary index; CRISPOR (https://github.com/maximilianh/crisporPaper) — comprehensive on/off-target scoring pipeline; PLM-CRISPR (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12254127/) — protein LM for Cas9 variant activity prediction with GPU inference.
- **CUDA libraries & GPU pattern:** Custom CUDA mismatch enumeration kernels (parallel BFS across mismatch positions); GPU-resident genome index in constant/global memory; cuDNN for CNN on-target scoring; batched transformer inference (ESM / PLM) on GPU; one CUDA thread per genome position.

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
