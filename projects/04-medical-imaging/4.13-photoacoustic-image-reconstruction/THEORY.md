# THEORY — 4.13 Photoacoustic Image Reconstruction

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

### 4.13 Photoacoustic Image Reconstruction 🟡 · Active R&D
- **Deep dive:** Photoacoustic imaging (PAI) generates ultrasound waves by pulsed laser absorption in tissue; images are reconstructed from time-series pressure data on a sensor surface. Delay-and-sum backprojection is analogous to ultrasound but in 3D; for 1,024 sensors and a 256³ volume, ~68 billion delay-and-sum operations are required per image — tractable only on GPU. Model-based iterative reconstruction solves the wave equation numerically (k-space pseudospectral method via cuFFT), enabling quantitative PAI with accurate acoustic attenuation and heterogeneous speed-of-sound modelling. Real-time 3D PA imaging for interventional guidance requires GPU throughput of multiple frames/second.
- **Key algorithms:** Delay-and-sum backprojection, time-reversal reconstruction, universal back-projection, k-space pseudo-spectral wave propagation (k-Wave), iterative model-based PA reconstruction, compressed sensing PAI, deep learning end-to-end PA reconstruction.
- **Datasets:** k-Wave simulation datasets (generated locally); USCT (Ultrasound Computed Tomography) benchmark data (verify URL); in vivo photoacoustic datasets from Nature Communications publications (open access); PASCAA challenge datasets (verify URL at photoacoustics.org).
- **Starter repos/tools:** k-Wave (http://www.k-wave.org/, CUDA C++ version at https://github.com/klepo/k-Wave-Fluid-CUDA) — industry-standard PA/US simulation and reconstruction toolbox; OpenMSOT (open multi-spectral optoacoustic tomography framework, verify URL); k-Wave MATLAB + CUDA backend for fast GPU wave simulation; PyTomography (https://github.com/lukepolson/pytomography) — Python GPU tomographic reconstruction including photoacoustic.
- **CUDA libraries & GPU pattern:** cuFFT for k-space wave propagation; custom CUDA kernel for DAS (one thread per voxel, loop over sensors); CUDA texture for time-series data interpolation; shared memory for sensor geometry LUT; multi-GPU decomposition over k-space planes.

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
