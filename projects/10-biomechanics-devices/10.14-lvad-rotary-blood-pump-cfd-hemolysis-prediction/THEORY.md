# THEORY — 10.14 LVAD / Rotary Blood Pump CFD & Hemolysis Prediction

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

### 10.14 LVAD / Rotary Blood Pump CFD & Hemolysis Prediction 🟡 · Active R&D

- **Deep dive:** Left ventricular assist devices (LVADs) expose blood to high shear stress at impeller blades, triggering hemolysis and thrombus formation. Patient-specific CFD with a rotating reference frame and moving mesh requires GPU-accelerated Navier-Stokes solutions on unstructured grids with ~5 M cells. The hemolysis index (power-law Giersiepen-Wurzinger model) is integrated along particle pathlines, computed by GPU-resident Lagrangian particle tracking. The 2024 simulation study demonstrated that hyperadhesion of activated platelets plays a dominant role in LVAD thrombosis at high rotor speeds. Design variants (impeller blade count, tip clearance) are evaluated in batches on GPU to build surrogate response surfaces for optimization.
- **Key algorithms:** Rotating reference frame Navier-Stokes (MRF/sliding mesh), Lagrangian particle tracking for hemolysis integration, platelet activation and thrombosis model (7-agonist biochemical cascade), power-law hemolysis (GKM), Euler-Euler two-phase (plasma + RBC) formulation, immersed boundary for rotor blades.
- **Datasets:** FDA Benchmark Pump Dataset — PIV-measured flow in centrifugal/axial blood pumps (https://www.fda.gov/science-research/about-science-research-fda/computational-modeling-biomedical-devices); Multi-GPU IB Hemodynamics Benchmark (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7402620/); LVAD Thrombosis Simulation Archive (see https://arxiv.org/abs/2312.04761); HeartMate 3 geometry (anonymized, verify via Frontiers Cardiovasc Med).
- **Starter repos/tools:** OpenFOAM (https://github.com/OpenFOAM) — rotating machinery solvers (MRFSimpleFoam) with GPU linear-algebra backends; HemeLB (https://github.com/UCL/hemelb) — GPU LBM for cardiovascular flows; IBM at Extreme Scale (https://arxiv.org/html/2605.04335) — OpenACC+CUDA+NCCL IBM solver; CUDA particle tracking kernel templates (https://github.com/NVIDIA/CUDALibrarySamples).
- **CUDA libraries & GPU pattern:** CUDA rotating-frame velocity interpolation kernels, cuSPARSE for pressure-velocity coupling, Thrust for particle trajectory integration; pattern: GPU unstructured CFD mesh → MRF velocity correction per cell → Lagrangian particle release → CUDA pathline integration → per-particle hemolysis accumulation → thrombosis probability map.

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
