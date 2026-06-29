# THEORY — 10.4 Haptic Rendering for Medical Training

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

### 10.4 Haptic Rendering for Medical Training 🟡 · Active R&D

- **Deep dive:** Haptic devices require force updates at 1 kHz or faster; the GPU must solve deformation and contact in under 1 ms per cycle. Energy-based haptic rendering computes virtual coupling forces from the difference between haptic device position and simulated tissue surface, requiring rapid contact detection and signed-distance-field (SDF) queries. GPU-accelerated SDFs pre-computed on volumetric grids enable sub-millisecond closest-point queries. Arterial catheter simulators, endoscopy trainers, and bone-drilling trainers all demand layered material models (mucosa, submucosa, muscle) with distinct stiffness, requiring per-layer GPU FE subsolvers. The bottleneck is contact resolution at the tool-tissue interface, parallelized over candidate contact pairs.
- **Key algorithms:** Energy-based haptic rendering with virtual coupling, signed-distance-field (SDF) contact detection, XPBD constraint projection, layered viscoelastic material models (Kelvin-Voigt), penumbra-based friction, god-object method for haptic proxy.
- **Datasets:** SOFA haptic benchmark scenes (liver puncture, needle insertion) (https://www.sofa-framework.org/); CholecT50 — laparoscopic cholecystectomy video for ground-truth tissue interaction reference (https://github.com/CAMMA-public/cholect50); Hamlyn Centre Laparoscopic / Robotic Video Dataset (http://hamlyn.doc.ic.ac.uk/vision/); Human Tissue Mechanical Properties Database (Picinbono et al., verify via SpringerLink).
- **Starter repos/tools:** SOFA Framework (https://github.com/sofa-framework/sofa) — modular GPU haptic-enabled simulator with OpenHaptics integration; Haptics-Medical-Simulation (https://github.com/HarrisKomn/Haptics-Medical-Simulation) — SOFA-based lung/bronchus haptic trainer with Geomagic Touch; Open-Source Visuo-Haptic Simulator (https://github.com/ChiaraSapo/Open-Source-Visuo-Haptic-Simulator-for-Surgical-Training) — SOFA-based multi-task haptic trainer; CHAI3D (https://www.chai3d.org) — haptic rendering framework with GPU geometry kernel support.
- **CUDA libraries & GPU pattern:** CUDA kernels for SDF ray marching and contact pair query, cuSPARSE for tissue stiffness subsolve, Thrust for collision broadphase; pattern: GPU-resident SDF updated each deformation step → parallel contact pair generation → energy-gradient force computation → CPU haptic device readout at 1 kHz via shared ring buffer.

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
