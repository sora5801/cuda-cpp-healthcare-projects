# THEORY — 3.29 Motif Finding in Genomic Sequences

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

### 3.29 Motif Finding in Genomic Sequences 🟡 · Active R&D
- **Deep dive:** Transcription factor motif discovery from ChIP-seq peaks searches for over-represented sequence patterns (IUPAC or position weight matrices) against a background model. Expectation-Maximisation over all N×W sequence windows (N peaks × W-k+1 positions per peak) is O(N×W×4^k) for exhaustive search; GPU parallelism assigns one thread to each window position, computing the PWM score via a parallel dot product. mCUDA-MEME achieves orders-of-magnitude speedup by distributing MEME's EM steps across GPU cores and GPU clusters. For genome-scale ChIP-seq (millions of peaks), this turns multi-day CPU runs into hours.
- **Key algorithms:** MEME expectation-maximisation over sequence windows; position weight matrix (PWM) scoring; ZOOPS/OOPS/TCM motif occurrence models; FIMO discrete log-sum-over-PWM scoring; Gibbs sampling for motif discovery; JASPAR database PWM matching.
- **Datasets:** ENCODE ChIP-seq peak BED files — thousands of TF experiments (https://www.encodeproject.org/); JASPAR 2024 — curated PWM database (https://jaspar.elixir.no/); ReMap 2022 — regulatory elements from 5 k ChIP-seq experiments (https://remap.univ-amu.fr/); GEO ChIP-seq datasets (https://www.ncbi.nlm.nih.gov/geo/).
- **Starter repos/tools:** CUDA-MEME / mCUDA-MEME (https://cuda-meme.sourceforge.io/homepage.htm) — GPU cluster MEME, ultrafast motif discovery; Argo_CUDA (https://pubmed.ncbi.nlm.nih.gov/29281953/) — exhaustive GPU motif discovery for large datasets; MEME Suite (https://meme-suite.org/) — reference CPU motif toolkit; HOMER (https://github.com/samtools/homer — verify URL, originally http://homer.ucsd.edu/) — CPU ChIP-seq motif enrichment tool.
- **CUDA libraries & GPU pattern:** One CUDA thread per sequence window for PWM scoring; shared-memory PWM matrix loaded once per kernel; warp-level sum for log-probability accumulation; thrust for top-k motif score extraction; batched EM outer loops with inter-GPU synchronisation.

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
