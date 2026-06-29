# THEORY — 6.2 Whole-Heart Digital Twin

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

### 6.2 Whole-Heart Digital Twin 🟡 · Active R&D
- **Deep dive:** Integrates patient-specific cardiac geometry (from CMR segmentation), fiber orientation (rule-based or DTI), EP simulation, and mechanical contraction into a unified virtual organ calibrated to clinical measurements. Building the twin requires iterative parameter estimation loops—thousands of forward simulations of the EP+mechanics PDE system—making GPU acceleration critical not just for each simulation but for the ensemble inference step. Differentiable simulators (e.g., TorchCor) allow gradient-based parameter fitting through the forward model. Hemodynamic boundary conditions couple the twin to a lumped Windkessel circulation model.
- **Key algorithms:** Bidomain/monodomain EP, active-strain / active-stress cardiac mechanics (nonlinear elasticity), Windkessel 3-element lumped circulation, rule-based fiber assignment (Bayer-Blake-Plank), adjoint-based or ensemble Kalman filter parameter estimation, finite element method (FEM) with tetrahedral meshes.
- **Datasets:** UK Biobank Cardiac MRI — 100 000+ cine CMR (https://www.ukbiobank.ac.uk); Zenodo Synthetic Biventricular Heart Meshes — 1 000 virtual cohort meshes (https://zenodo.org/records/4506930); Visible Human Project — full-body cryosection + CT + MRI (https://www.nlm.nih.gov/research/visible/visible_human.html); ACDC MICCAI — 100-patient CMR segmentations (https://www.creatis.insa-lyon.fr/Challenge/acdc/).
- **Starter repos/tools:** openCARP (https://git.opencarp.org/openCARP/openCARP) — EP component of twins; simcardems (https://github.com/ComputationalPhysiology/simcardems) — FEniCS-based cardiac electromechanics coupling; TorchCor (https://github.com/sagebei/torchcor) — PyTorch GPU cardiac EP FEM for differentiable twin fitting; Awesome-Cardiac-Digital-Twins list (https://github.com/lileitech/Awesome-Cardiac-Digital-Twins) — curated resource index.
- **CUDA libraries & GPU pattern:** cuSPARSE + cuSOLVER (FEM assembly/solve), cuBLAS (adjoint vector ops), custom CUDA kernels (ionic ODE batch); pattern: batched forward solves across ensemble members for parameter inference; mixed precision (FP16 forward, FP32 gradient accumulation).

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
