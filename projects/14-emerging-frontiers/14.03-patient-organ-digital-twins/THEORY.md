# THEORY — 14.3 Patient / Organ Digital Twins

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

### 14.3 Patient / Organ Digital Twins 🟡 · Active R&D

- **Deep dive:** A patient digital twin integrates genomics, imaging, hemodynamics, metabolism, and pharmacokinetics into a continuously updated computational model that predicts disease progression and treatment response for a specific individual. GPU-accelerated cardiac digital twins (npj Systems Biology 2023) solve full multi-physics cardiac electromechanics in a few hours per heartbeat on 4 GPU cards — opening systematic cohort-scale simulation campaigns. Cancer digital twins ("cancer avatars") couple tumor growth PDE models with pharmacodynamic ODEs, parameterized from serial liquid biopsies, enabling adaptive treatment optimization. The GPU bottleneck is the repeated FEM/CFD solve at each patient update cycle.
- **Key algorithms:** Multi-physics cardiac electromechanics (bidomain + passive/active material), tumor growth (reaction-diffusion PDE, Go-or-Grow model), pharmacokinetic-pharmacodynamic (PKPD) ODE integration, Bayesian data assimilation (ensemble Kalman filter), physics-informed neural network surrogates, mesh morphing for anatomy personalization.
- **Datasets:** UK Biobank Imaging — cardiac MRI + genomics on 100 K subjects (https://www.ukbiobank.ac.uk/); TCIA — cancer imaging archive (https://www.cancerimagingarchive.net/); ClinicalTrials.gov synthetic patient cohorts; Digital Twin Cardiovascular Cohort (GPU-accelerated, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10203142/).
- **Starter repos/tools:** NVIDIA PhysicsNeMo (https://github.com/NVIDIA/physicsnemo) — physics-informed neural network surrogates for organ modeling; OpenCMISS-Iron (https://github.com/OpenCMISS/iron) — GPU-capable cardiac electromechanics finite-element solver; SimVascular (https://github.com/SimVascular/SimVascular) — patient-specific cardiovascular CFD; CHASTE (https://github.com/Chaste/Chaste) — cardiac and tumor multiscale simulation.
- **CUDA libraries & GPU pattern:** cuSPARSE for bidomain cardiac FEM, cuDNN for surrogate model inference, NCCL for multi-GPU organ assembly; pattern: patient imaging segmented → geometry personalized → GPU multi-physics solve → Bayesian assimilation of new biomarker measurements → treatment response prediction → update digital twin.

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
