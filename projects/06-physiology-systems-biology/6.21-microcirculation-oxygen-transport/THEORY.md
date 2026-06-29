# THEORY — 6.21 Microcirculation & Oxygen Transport

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

### 6.21 Microcirculation & Oxygen Transport 🟡 · Active R&D
- **Deep dive:** Oxygen delivery from red blood cells to tissue parenchyma involves convection in capillaries, diffusion through capillary walls and interstitium (Krogh cylinder / Green's function models), and intracellular O₂ reaction/consumption (Michaelis-Menten kinetics). A realistic tissue volume (~1 mm³) contains thousands of capillaries forming a 3D network; GPU parallelism is applied to the per-segment convection-diffusion solves and the volumetric Green's function integrals (which are an O(N²) operation accelerated to O(N log N) via multipole or GPU-NUFFT).
- **Key algorithms:** Krogh cylinder O₂ transport, Green's function method (Secomb Hsu), 1D convection-diffusion along capillary segments, Michaelis-Menten O₂ consumption, fast multipole method (FMM) for Green's function sums, hemoglobin saturation curve (Hill equation), hematocrit-dependent RBC flux partitioning.
- **Datasets:** Vascular Model Repository (http://www.vascularmodel.com); two-photon microscopy microvascular datasets from Allen Institute (https://portal.brain-map.org); PhysioNet oxygen saturation waveforms (https://physionet.org); published microvascular network datasets (Secomb group, verify at secomb.org).
- **Starter repos/tools:** HemeLB (https://github.com/hemelb-codes/hemelb) — sparse LBM for capillary flow; USERMESO-2.0 (https://github.com/AnselGitAccount/USERMESO-2.0) — GPU red blood cell hemodynamics with deformable membranes; APBS (https://github.com/Electrostatics/apbs) — electrostatics solver repurposable for O₂ diffusion; OpenFOAM (https://github.com/OpenFOAM/OpenFOAM-dev) — volume-average tissue oxygenation.
- **CUDA libraries & GPU pattern:** CUDA NUFFT or FMM (cuFMM) for Green's function O₂ sums; custom CUDA kernels for per-segment RBC oxygen release; cuSPARSE for network flow solve; pattern: segment-parallel threads for convection update + shared-memory reduction for junction mass balance.

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
