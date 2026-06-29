# THEORY — 13.15 Drug-Induced Liver Injury (DILI) & Quantitative Systems Toxicology

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

### 13.15 Drug-Induced Liver Injury (DILI) & Quantitative Systems Toxicology 🟡 · Active R&D

- **Deep dive:** Predicts and mechanistically explains drug-induced liver injury using multi-scale QST models (DILIsym) that integrate intracellular mitochondrial function, bile acid synthesis/transport, oxidative stress, and innate immune response with drug concentration-dependent perturbations. The stiff ODE system (300+ equations for intracellular biochemistry × hepatocyte populations × liver zonation) requires GPU-parallel stiff integration for virtual patient simulations. Graph convolutional networks on drug molecular graphs (BioGL-GCN) trained on hepatotoxicity labels enable rapid screening of new compounds. Combining GCN screening with mechanistic QST validation on GPU covers both speed and interpretability.
- **Key algorithms:** QST ODE integration (CVODE, RODAS4), mitochondrial membrane potential dynamics ODEs, bile acid transport ODE system, NF-κB signalling cascade, GCN/GNN on molecular graphs for hepatotoxicity classification, random forest + physicochemical feature DILI prediction, multiscale coupling of PBPK with intracellular QST.
- **Datasets:**
  - DILIst — curated DILI positive/negative drug list (verify URL; NCATS)
  - LiverTox — NIH database of drug-induced liver disease (https://www.ncbi.nlm.nih.gov/books/NBK547852/)
  - Tox21 — 12,000+ compounds with hepatotoxicity assay data (https://tox21.gov/)
  - DILIsym virtual patient database (Simulations Plus) — calibrated virtual liver population (verify URL)
- **Starter repos/tools:**
  - DILIsym (https://www.simulations-plus.com/software/dilisym/) — commercial QST DILI platform (Simulations Plus)
  - BioGL-GCN (verify URL) — graph convolutional network for DILI prediction from drug structures
  - DeepTox (https://github.com/bioinf-jku/tox21_networks) — deep learning Tox21 prediction baseline
  - nvQSP (https://github.com/NVIDIA-Digital-Bio/nvQSP) — GPU stiff ODE solver for QST models
- **CUDA libraries & GPU pattern:** Custom CUDA CVODE/RODAS4 stiff ODE kernels for QST integration, DGL for hepatotoxicity GCN, cuBLAS for bile acid flux Jacobians; pattern: virtual patient batch — one CUDA block per patient, intracellular biochemistry compartments in shared memory.

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
