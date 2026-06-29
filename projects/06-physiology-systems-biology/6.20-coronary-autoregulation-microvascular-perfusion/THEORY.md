# THEORY — 6.20 Coronary Autoregulation & Microvascular Perfusion

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

### 6.20 Coronary Autoregulation & Microvascular Perfusion 🟡 · Active R&D
- **Deep dive:** Coronary blood flow is regulated by metabolic (adenosine), myogenic, and neural mechanisms operating across scales from capillaries (5 µm) to epicardial arteries (4 mm). GPU simulation of a microvascular network with 10⁴–10⁶ vessel segments requires solving a large sparse linear system (network Poiseuille flow) coupled to oxygen transport (convection-diffusion along each segment) and auto-regulatory feedback ODEs. Real-time coronary perfusion models support fractional flow reserve (FFR) virtual assessment for stenosis evaluation.
- **Key algorithms:** Network Poiseuille flow (sparse linear system), convection-diffusion oxygen transport along segments, Green's function oxygen transport in tissue, myogenic/metabolic regulation ODE, 1D structured-tree Windkessel for coronary outlet, FFR virtual computation, Fåhræus-Lindqvist effect (hematocrit-dependent viscosity).
- **Datasets:** UK Biobank coronary CTA (subset) (https://www.ukbiobank.ac.uk); PhysioNet coronary pressure/flow waveforms (https://physionet.org); Vascular Model Repository coronary geometries (http://www.vascularmodel.com); MICCAI coronary artery tracking challenge datasets (grand-challenge.org).
- **Starter repos/tools:** SimVascular (https://github.com/SimVascular/svFSI) — coronary flow boundary conditions (structured tree); HemeLB (https://github.com/hemelb-codes/hemelb) — sparse vascular LBM for microvascular beds; APBS (https://github.com/Electrostatics/apbs) — electrostatics analogy for oxygen transport; OpenFOAM (https://github.com/OpenFOAM/OpenFOAM-dev) — coronary CFD with custom UDF.
- **CUDA libraries & GPU pattern:** cuSPARSE for network flow linear system (sparse symmetric positive definite); cuSPARSE SpMV for iterative CG; CUDA Thrust for per-segment oxygen PDE; pattern: one thread per vessel segment for transport update, shared memory for branching connectivity.

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
