# THEORY — 5.14 GPU-Accelerated Adaptive MR-Linac Workflow

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

### 5.14 GPU-Accelerated Adaptive MR-Linac Workflow 🟡 · Active R&D
- **Deep dive:** MR-Linac (MRL) systems (Elekta Unity, ViewRay MRIdian) combine MRI with simultaneous radiation delivery, enabling online adaptive radiotherapy (oART) where each fraction's plan is re-optimized based on daily anatomy. The oART workflow must complete all steps within a 30–90 minute treatment slot: (1) real-time MRI reconstruction (GPU NUFFT, <1 s), (2) deformable MR-to-MR registration (GPU Demons/VoxelMorph, <30 s), (3) synthetic CT generation (deep learning CT from MRI, GPU CNN, <10 s), (4) GPU dose recalculation on adapted anatomy (<30 s via collapsed-cone or MC), and (5) re-optimization (<2 min). Every step requires GPU; the entire chain is a GPU pipeline.
- **Key algorithms:** Real-time MRI reconstruction (radial GRASP GPU), MR-to-MR deformable registration (Demons, SyN), synthetic CT generation (CNN: MR→sCT), GPU collapsed-cone dose on sCT, GPU proton or photon dose recalculation, warm-start IMRT fluence re-optimization, plan approval metric computation.
- **Datasets:** MR-Linac Consortium shared datasets (verify URL at mrlinac.org); TCIA MR-guided RT datasets; AAPM MR-Linac WG test cases; MRI-only radiotherapy datasets from published cohorts.
- **Starter repos/tools:** Gadgetron (https://github.com/gadgetron/gadgetron) — real-time GPU MRI reconstruction for MRL; Plastimatch (https://plastimatch.org/) — GPU DIR + sCT generation; matRad (https://github.com/e0404/matRad) — dose re-optimization kernel; MONAI (https://github.com/Project-MONAI/MONAI) — CNN for MR→sCT translation.
- **CUDA libraries & GPU pattern:** CUDA streams pipeline: acquisition → cuFFT NUFFT → cuDNN sCT CNN → GPU dose kernel → cuSPARSE optimizer → display; each stage double-buffered to overlap computation with data transfer; multi-GPU across the 5-stage pipeline.

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
