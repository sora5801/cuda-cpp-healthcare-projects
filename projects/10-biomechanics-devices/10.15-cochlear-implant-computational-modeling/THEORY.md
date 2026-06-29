# THEORY — 10.15 Cochlear Implant Computational Modeling

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

### 10.15 Cochlear Implant Computational Modeling 🟡 · Active R&D

- **Deep dive:** Cochlear implant (CI) electrodes stimulate spiral ganglion neurons via current fields that spread through complex fluid-filled scala tympani geometries. GPU-accelerated FEM on micro-CT-derived cochlear geometries computes the full 3D voltage distribution across the spiral ganglion fiber population in under a second, enabling real-time comparison of electrode array designs. Multi-compartment auditory nerve fiber (ANF) cable models are integrated in parallel on GPU — one thread per fiber per timestep — to predict neural firing patterns from arbitrary stimulation waveforms. Population-model simulations over thousands of virtual patients with varying cochlear anatomy quantify inter-subject variability in electrode coupling.
- **Key algorithms:** Volume-conductor FEM (bidomain), multi-compartment Hodgkin-Huxley cable models for ANF, psychoacoustic loudness growth modeling, Green's function electrode-impedance computation, Monte Carlo sampling over cochlear geometry populations.
- **Datasets:** Cochlear Micro-CT Atlas (25 ANF traced geometries, see https://www.frontiersin.org/articles/10.3389/fnins.2025.1639092); Electrical Stimulation Human Cochlea Dataset (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6915103/); SIMBIOsys Cochlear Models (https://www.upf.edu/web/simbiosys/cochlear-implants); PhysioNet auditory nerve response databases (verify via physionet.org).
- **Starter repos/tools:** FEBio (https://github.com/febiosoftware/FEBio) — bidomain volume conductor FEM; NEURON simulator GPU branch (https://github.com/neuronsimulator/nrn) — parallel ANF cable integration; SimNIBS (https://github.com/simnibs/simnibs) — FEM for electrostimulation (adaptable to cochlear geometry); Cochlear FEM pipeline (SIMBIOsys UPF, verify URL at UPF site) — CI-specific meshing and solving workflow.
- **CUDA libraries & GPU pattern:** cuSPARSE/cuSolver for bidomain FEM voltage solve, CUDA kernels for per-fiber HH cable ODE integration (embarrassingly parallel over ANFs), cuRAND for stochastic threshold variability; pattern: GPU FEM voltage field → per-fiber interpolation of extracellular potential → parallel ODE integration of HH equations → spike-time extraction → population audiogram prediction.

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
