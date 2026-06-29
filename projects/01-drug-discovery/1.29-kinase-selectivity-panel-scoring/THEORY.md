# THEORY — 1.29 Kinase Selectivity Panel Scoring

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

### 1.29 Kinase Selectivity Panel Scoring 🟡 · Active R&D

- **Deep dive:** Kinases share highly similar binding pockets, making selectivity a central challenge in kinase drug discovery. A GPU MD + ML pipeline can score a compound across 500+ kinase structures simultaneously: (1) GPU-parallel docking against all kinase homology models, (2) ML scoring using kinase-specific fingerprints (KLIFS features), (3) MM-GBSA rescoring. GPU acceleration allows a compound to be profiled against the entire kinome in minutes rather than days. Selectivity fingerprinting using ensemble docking with GPU makes this tractable.
- **Key algorithms:** Ensemble docking, kinase-ligand interaction fingerprints (KLIFS/IFP), selectivity scoring (SFP), homology model generation, structural kinome alignment, ML kinase activity prediction (KinaseML).
- **Datasets:** KLIFS — kinase-ligand interaction fingerprinting database (https://klifs.net); KinomeScan — 468-kinase selectivity data (verify URL); ChEMBL kinase activity data (https://www.ebi.ac.uk/chembl/); DTC drug-target commons kinase panel (https://dtcommons.ai).
- **Starter repos/tools:** AutoDock-GPU (https://github.com/ccsb-scripps/AutoDock-GPU) — GPU docking against kinase panels; KLIFS Python API (https://github.com/volkamerlab/kissim) — kinase structural fingerprints; KinoML (https://github.com/openkinome/kinoml) — ML for kinase drug discovery; HTMD (https://github.com/Acellera/htmd) — GPU-based kinome docking workflows.
- **CUDA libraries & GPU pattern:** GPU-parallel docking against kinase model array; GPU-batched IFP featurization; cuML for kinase activity ML training; Thrust for topK selectivity ranking.

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
