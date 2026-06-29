# THEORY — 6.3 Hemodynamics / Blood-Flow CFD

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

### 6.3 Hemodynamics / Blood-Flow CFD 🟡 · Active R&D
- **Deep dive:** Solves the incompressible Navier-Stokes equations on patient-specific vascular geometries (aorta, coronary arteries, cerebral vasculature) reconstructed from CT/MRI angiography. Non-Newtonian blood rheology (Carreau-Yasuda model) and fluid-structure interaction (FSI) with compliant vessel walls add computational stiffness. Wall shear stress (WSS) and oscillatory shear index (OSI) fields—risk factors for atherosclerosis—require temporally resolved solutions across the cardiac cycle (~1000 time steps). GPU parallelism maps naturally onto the unstructured mesh cell updates.
- **Key algorithms:** Incompressible Navier-Stokes (fractional-step / SIMPLE / PISO), ALE formulation for FSI, non-Newtonian viscosity (Carreau-Yasuda), arbitrary Lagrangian-Eulerian mesh motion, finite volume method on unstructured polyhedral meshes, multigrid pressure solver, RBF mesh morphing.
- **Datasets:** PhysioNet MIMIC-III waveforms — invasive pressure/flow recordings (https://physionet.org/content/mimiciii/1.4/); Vascular Model Repository — patient-specific vascular geometries (http://www.vascularmodel.com); Zenodo Cardiac Mechanics Emulation dataset (https://zenodo.org/records/7075055); UK Biobank aortic flow (4D flow MRI subset) (https://www.ukbiobank.ac.uk).
- **Starter repos/tools:** SimVascular/svFSI (https://github.com/SimVascular/svFSI) — open-source image-to-simulation pipeline with GPU-capable parallel solver; OpenFOAM-dev (https://github.com/OpenFOAM/OpenFOAM-dev) — general CFD with biomedical application via custom boundary conditions; Chaste (https://github.com/Chaste/Chaste) — includes vascular network flow module; HemeLB (https://github.com/hemelb-codes/hemelb) — sparse vascular lattice-Boltzmann alternative.
- **CUDA libraries & GPU pattern:** AmgX (GPU multigrid pressure solver), cuSPARSE (SpMV for flux assembly), NVIDIA RAPIDS for mesh preprocessing; pattern: domain decomposition with MPI+CUDA, halo-exchange via NCCL, time-stepping loop with async memory transfers.

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
