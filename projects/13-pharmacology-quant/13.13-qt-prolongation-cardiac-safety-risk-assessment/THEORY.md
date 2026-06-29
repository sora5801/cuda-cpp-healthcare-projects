# THEORY — 13.13 QT-Prolongation & Cardiac Safety Risk Assessment

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

### 13.13 QT-Prolongation & Cardiac Safety Risk Assessment 🟡 · Active R&D

- **Deep dive:** Predicts drug-induced QT interval prolongation — a surrogate for fatal arrhythmia (Torsade de Pointes) — from drug structure, hERG channel IC50 measurements, and clinical ECG data. The CardioGenAI framework uses GPU-accelerated molecular graph neural networks to predict hERG block and re-engineer drug structures for reduced liability. Clinical ECG-based deep learning (3DRECON-QT) reconstructs 3D spatial QTc from single-lead recordings using CNN on GPU. Mechanistic cardiac action potential models (O'Hara-Rudy, Paci human iPSC-CM) simulate drug effects on ion channels at thousands of drug concentrations simultaneously on GPU — each simulation is an ODE stiff system on the 40+ state Hodgkin-Huxley-type action potential model.
- **Key algorithms:** GNN-based hERG IC50 prediction from SMILES, 3DRECON-QT spatial reconstruction, O'Hara-Rudy action potential ODE, voltage-clamp state machine (Markov model for hERG), torsade de pointes risk classification (TdP risk categories), dynamic clamp simulation on GPU, QTc Fridericia/Bazett correction.
- **Datasets:**
  - CiPA (Comprehensive in vitro Pro-arrhythmia Assay) ion channel datasets — multi-channel IC50 for 28 reference drugs (verify URL via FDA)
  - hERGCentral database — hERG patch-clamp measurements (verify URL)
  - MIMIC-IV-ECG — clinical QTc measurements linked to medication data (https://physionet.org/content/mimic-iv-ecg/)
  - CardioNet ECG database (verify URL) — large annotated ECG dataset for QT analysis
- **Starter repos/tools:**
  - CardioGenAI (https://github.com/mgreenig/CardioGenAI) — ML framework for re-engineering drugs for reduced hERG liability
  - myokit (https://github.com/myokit/myokit) — cardiac action potential ODE modelling; GPU via CUDA backend
  - OpenCARP (https://opencarp.org/) — cardiac electrophysiology simulator with GPU support
  - DeepHERG (verify URL) — deep learning hERG inhibition prediction
- **CUDA libraries & GPU pattern:** DGL for hERG GNN, custom CUDA Hodgkin-Huxley ODE kernels for action potential batch simulation, cuRAND for Monte Carlo drug concentration sweeps; pattern: one CUDA thread per drug concentration × cell simulation, with shared memory for ion channel state variables.

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
