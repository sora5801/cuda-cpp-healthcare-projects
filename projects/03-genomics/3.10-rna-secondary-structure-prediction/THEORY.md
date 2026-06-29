# THEORY — 3.10 RNA Secondary-Structure Prediction

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

### 3.10 RNA Secondary-Structure Prediction 🟡 · Active R&D
- **Deep dive:** RNA folds into hairpins and stems governed by free-energy minimisation via the Zuker algorithm (O(n³) time, O(n²) space). For sequences >10 kb (rRNA, lncRNA), the cubic cost is prohibitive on CPU. GPU parallelism exploits the diagonal wavefront of the DP table: all cells (i,j) on the same diagonal d=j-i are independent and can be updated simultaneously by CUDA threads, similar to SW alignment. CUDA RNAfold achieves 14× speedup for sequences up to 30 kb. LinearFold reduces the complexity to O(n) using a beam-search approximation and lends itself to GPU batch processing of thousands of short RNAs in parallel.
- **Key algorithms:** Zuker free-energy minimisation (partition function DP); McCaskill partition function (base-pair probabilities); anti-diagonal wavefront parallelism; LinearFold beam-search O(n); Vienna RNA thermodynamic model; stochastic context-free grammar (SCFG) parsing.
- **Datasets:** Rfam — RNA family alignments and secondary structures (https://rfam.org/); RNAcentral — comprehensive RNA sequence database (https://rnacentral.org/); PDB RNA structures — known 3D-validated secondary structures (https://www.rcsb.org/); ArchiveII benchmark — curated RNA secondary structure data (verify URL).
- **Starter repos/tools:** CUDA RNAfold (https://www.biorxiv.org/content/10.1101/298885v1.full) — GPU-parallelised Vienna RNAfold, 14× speedup; LinearFold (https://github.com/LinearFold/LinearFold) — O(n) RNA folding with GPU batch variant; LinearAlifold (https://github.com/LinearFold/LinearAlifold) — consensus structure prediction; EternaFold (https://github.com/eternagame/EternaFold) — ML-trained folding model for GPU inference.
- **CUDA libraries & GPU pattern:** Anti-diagonal wavefront kernel (custom CUDA, shared-memory tiling of DP triangle); one warp per diagonal cell group; thrust for energy table initialization; cuFFT (not standard here, but used in some spectral RNA analyses); batch RNA folding with one CTA per sequence.

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
