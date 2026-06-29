# THEORY — 6.1 Cardiac Electrophysiology Simulation

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

### 6.1 Cardiac Electrophysiology Simulation 🟡 · Active R&D
- **Deep dive:** Simulates transmembrane voltage propagation across cardiac tissue by solving the monodomain or bidomain reaction-diffusion PDE coupled to stiff ODEs representing ionic channel kinetics (e.g., ten Tusscher-Panfilov, O'Hara-Rudy). Each voxel integrates 50–200 state variables per time step at sub-millisecond temporal resolution; a whole-heart simulation at 0.1 mm spatial resolution yields ~10⁸ nodes, making the per-node ODE update embarrassingly parallel. The GPU eliminates the otherwise serial per-cell Rush-Larsen / RL2 exponential gating integration. Operator splitting decouples the reaction (GPU-parallel ODE) from diffusion (sparse linear solve), and CUDA kernels saturate memory bandwidth on the former while cuSPARSE handles the latter.
- **Key algorithms:** Monodomain/bidomain reaction-diffusion, operator splitting (Strang/Godunov), Rush-Larsen explicit gating, Crank-Nicolson implicit diffusion, conjugate gradient with ILU(0) preconditioning, finite volume/finite element spatial discretization.
- **Datasets:** PhysioNet MIT-BIH & MIMIC-III Waveform — 40 000+ ICU ECG/hemodynamic waveforms (https://physionet.org); CellML Physiome Repository — curated ionic cell models in CellML/SBML format importable by openCARP (https://models.physiomeproject.org); UK Biobank Cardiac MRI — 100 000+ cine CMR studies, access via application (https://www.ukbiobank.ac.uk); ACDC MICCAI Cardiac Challenge — 100-patient CMR with LV/RV/myocardium ground truth (https://www.creatis.insa-lyon.fr/Challenge/acdc/).
- **Starter repos/tools:** openCARP (https://git.opencarp.org/openCARP/openCARP) — MPI+CUDA cardiac EP solver, CARPutils Python scripting, v19.0 April 2026; MonoAlg3D_C (https://github.com/rsachetto/MonoAlg3D_C) — finite-volume GPU monodomain solver with Purkinje coupling and MPI batch dispatch; Cardioid/LLNL (https://github.com/llnl/cardioid) — multiscale cardiac suite (EP + mechanics + ECG), CUDA optional, Gordon Bell finalist; Chaste (https://github.com/Chaste/Chaste) — Oxford bidomain solver with cardiac mechanics module.
- **CUDA libraries & GPU pattern:** cuSPARSE (diffusion SpMV), cuSOLVER (linear system), CUDA custom kernels (per-cell ODE Rush-Larsen); pattern: fine-grained thread-per-cell ODE + coarse SpMV for diffusion; streams for overlapping compute and halo exchange.

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
