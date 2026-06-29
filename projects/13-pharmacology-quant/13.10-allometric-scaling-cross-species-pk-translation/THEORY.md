# THEORY — 13.10 Allometric Scaling & Cross-Species PK Translation

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

### 13.10 Allometric Scaling & Cross-Species PK Translation 🟢 · Established

- **Deep dive:** Translates preclinical animal PK parameters to human predictions using allometric power laws, species-specific physiological scaling, and mechanistic PBPK bridging. When applied at scale — scoring thousands of drug candidates from an in vivo animal study to prioritise compounds for human trials — the PBPK-based cross-species translation requires solving complete animal and human PBPK ODE systems for each candidate. GPU batch ODE integration across thousands of candidates simultaneously is the core acceleration; each candidate requires solving ~15-compartment human and rat/mouse PBPK systems in parallel. Machine learning models (trained on ChEMBL animal-to-human PK datasets) that predict human CL, Vd, and t½ from molecular features are GPU-accelerated via neural forward passes.
- **Key algorithms:** Simple allometry (body weight power law), Maximum Lifespan Potential (MLP) correction, Rule of Exponents, PBPK-based cross-species translation, in vitro-in vivo extrapolation (IVIVE), machine-learning regression from molecular descriptors to PK parameters, QSAR-PK modelling.
- **Datasets:**
  - ChEMBL PK dataset — 18k+ compounds with preclinical and human PK data (https://www.ebi.ac.uk/chembl/)
  - Lombardo et al. drug PK dataset — 1352 drugs with CL, Vd, t½ in humans and animals (verify URL)
  - Obach et al. clearance dataset — metabolic clearance measurements (verify URL)
  - Open Systems Pharmacology species parameter databases (https://github.com/Open-Systems-Pharmacology/PK-Sim)
- **Starter repos/tools:**
  - PK-Sim (https://github.com/Open-Systems-Pharmacology/PK-Sim) — PBPK allometric scaling built-in
  - pkNCA (https://github.com/billdenney/pknca) — non-compartmental PK analysis in R
  - DeepPK (verify URL) — deep learning PK prediction for allometric scaling
  - ADMET-AI (https://github.com/swansonk14/admet_ai) — ML-based ADME/PK prediction pipeline
- **CUDA libraries & GPU pattern:** CUDA batched RK4 for species ODE systems, cuBLAS for regression model forward pass, cuDNN for molecular graph encoder; pattern: batch-parallel PBPK translation — one compound per CUDA thread group.

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
