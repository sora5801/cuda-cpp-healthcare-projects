# THEORY — 6.25 Liver & Kidney Perfusion Modeling

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

### 6.25 Liver & Kidney Perfusion Modeling 🟡 · Active R&D
- **Deep dive:** Liver lobules and kidney nephrons are structurally repetitive functional units that process blood to clear metabolites, drugs, and toxins. GPU simulation of drug/toxin clearance across millions of sinusoidal segments in a liver or tubular segments in a nephron network enables virtual pharmacotoxicology and organ-on-chip digital twins. Oxygen-zone-specific metabolism (periportal vs. centrilobular) and countercurrent exchange in the renal medullary vasa recta add physiological complexity.
- **Key algorithms:** Zonal liver sinusoid transport model, countercurrent exchange (renal medullary), convection-diffusion-reaction along network segments, Michaelis-Menten hepatic clearance, filtration-reabsorption-secretion nephron model, 3D lobular vascular network flow, PBPK liver/kidney sub-model.
- **Datasets:** Human Protein Atlas liver expression (https://www.proteinatlas.org); HMDB liver metabolomics (https://hmdb.ca); Open Systems Pharmacology PBPK model library (https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library); PhysioNet renal function datasets (https://physionet.org).
- **Starter repos/tools:** Open Systems Pharmacology Suite (https://github.com/Open-Systems-Pharmacology) — organ-level PBPK with liver/kidney compartments; mrgsolve (https://github.com/metrumresearchgroup/mrgsolve) — ODE-based organ pharmacokinetics; SimVascular (https://github.com/SimVascular/svFSI) — vascular tree flow for portal vein; HemeLB (https://github.com/hemelb-codes/hemelb) — microvessel LBM for sinusoidal flow.
- **CUDA libraries & GPU pattern:** Batch ODE (one thread per lobule unit or nephron segment); cuSPARSE for lobular network linear system; custom CUDA kernels for Michaelis-Menten reaction in each zone; pattern: hierarchical parallelism—CUDA blocks per lobule, threads per sinusoidal segment.

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
