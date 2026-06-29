# THEORY — 4.33 Real-Time MRI Reconstruction

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

### 4.33 Real-Time MRI Reconstruction 🟡 · Active R&D
- **Deep dive:** Interventional and cardiac MRI require image reconstruction latency <100 ms to enable real-time guidance (catheter navigation, cardiac function monitoring). Online adaptive compressed sensing with sliding window or XD-GRASP (extra-dimensional GRASP) processes continuously acquired non-Cartesian k-space (radial, spiral) with GPU NUFFT and compressed sensing reconstruction running in a locked pipeline with acquisition. Gadgetron, an open-source streaming MR reconstruction framework, pipelines coil compression, NUFFT, GRAPPA, and deep learning inference on GPU with acquisition-synchronous operation. The cardiac cycle adds a gating dimension, requiring 4D (3D + cardiac phase) reconstruction at interactive speeds only feasible on GPU.
- **Key algorithms:** XD-GRASP (multi-dimensional golden-angle radial), sliding-window NUFFT, online GRAPPA, low-rank + sparse reconstruction, compressed sensing NUFFT with TV, cardiac-gated CS (XTREAM, L+S), neural network real-time reconstruction (MoDL-S), real-time MRI with physiological monitoring.
- **Datasets:** Cardiac MRI datasets from ACDC challenge (https://www.creatis.insa-lyon.fr/Challenge/acdc/); CMRxRecon 2023 challenge (https://cmrxrecon.github.io/); dynamic cardiac MRI from MRXCAT simulation (verify URL); real-time fetal MRI from research groups.
- **Starter repos/tools:** Gadgetron (https://github.com/gadgetron/gadgetron) — GPU streaming MRI reconstruction server, GRAPPA/NUFFT/DL plugins; BART (https://github.com/mrirecon/bart) — GPU GRASP/CS-MRI for batch; MRzero (https://github.com/MRsimulator/MRzero) — differentiable real-time MR simulation; SigPy (https://github.com/mikgroup/sigpy) — Python NUFFT/CUDA for real-time prototyping.
- **CUDA libraries & GPU pattern:** cuFFT for NUFFT gridding; CUDA streams for acquisition-synchronous pipeline (double-buffering: acquire on CPU/scanner, reconstruct on GPU simultaneously); cuDNN for online DL inference; CUDA thrust for dynamic radial k-space sorting; multi-GPU for parallel cardiac phase reconstruction.

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
