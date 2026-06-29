# THEORY — 1.34 Amyloid / Aggregation Propensity Prediction

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

### 1.34 Amyloid / Aggregation Propensity Prediction 🟡 · Active R&D

- **Deep dive:** Protein aggregation drives diseases (Alzheimer's, Parkinson's, ALS) and is a major liability in biologic drug development. GPU-accelerated coarse-grained and atomistic MD can directly simulate fibril nucleation and extension, but requires microsecond-to-millisecond timescales accessible only with GPU enhanced sampling. ML aggregation predictors (AGGRESCAN3D, CamSol) train on experimental aggregation rates; GPU-trained GNNs on protein sequence+structure outperform sequence-only models. Amyloid fibril cryo-EM structures from EMDB drive validation.
- **Key algorithms:** β-aggregation propensity scoring, coarse-grained MD of oligomerization (MARTINI), REMD/MetaD of early aggregation, GNN aggregation predictor, solubility prediction neural networks.
- **Datasets:** AmyPro — curated amyloidogenic sequence database (https://amypro.net); FoldAmyloid prediction database (verify URL); ThT fluorescence assay aggregation kinetics datasets; EMDB fibril EM maps (https://www.ebi.ac.uk/emdb/).
- **Starter repos/tools:** AGGRESCAN3D server (https://biocomp.chem.uw.edu.pl/A3D2/) — structure-based aggregation prediction; CamSol (https://www-cohsoftware.ch.cam.ac.uk/index.php/camsolmethod) — solubility prediction; WALTZ-DB 2.0 (verify URL) — aggregation kinetics; GROMACS+PLUMED fibril simulation stack (https://github.com/gromacs/gromacs).
- **CUDA libraries & GPU pattern:** GPU MARTINI CG-MD for large oligomerization systems; metadynamics enhanced sampling via PLUMED on GPU; GPU-trained GNN inference for sequence-based aggregation; CUDA-accelerated contact map tracking during aggregation.

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
