# THEORY — 4.6 Ultrasound Beamforming

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

### 4.6 Ultrasound Beamforming 🟢 · Established
- **Deep dive:** Delay-and-sum (DAS) beamforming reconstructs B-mode images by computing time-delayed sums of per-element receive signals for every pixel in the image grid. For a 128-element linear array, a 512×512 image, and 4,000 scan lines per second, DAS requires ~3.4 × 10¹⁰ multiply-accumulate operations per second — far beyond real-time CPU capability. GPU parallelism maps each output pixel to a CUDA thread, computes focal delays from element geometry, interpolates raw RF data, and sums across elements; a single RTX-class GPU achieves interactive frame rates for 3D volumetric beamforming. Coherence-based techniques (DMAS, CF) add per-pixel statistics but remain embarrassingly parallel.
- **Key algorithms:** Delay-and-sum (DAS), f-k migration, synthetic aperture focusing (SAFT), coherence factor (CF), DMAS (delay-multiply-and-sum), compressed sensing beamforming, Fourier domain reconstruction, adaptive minimum variance beamforming.
- **Datasets:** Plane-Wave Imaging Challenge in Medical Ultrasound (PICMUS, https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/) — RF data for beamforming evaluation; UltraSound SegLab dataset; IQ ultrasound datasets from open research groups (verify URL at creatis.insa-lyon.fr).
- **Starter repos/tools:** GPU-accelerated US beamforming repos on GitHub (search "CUDA ultrasound beamforming"); MUST (MATLAB Ultrasound Toolbox, https://www.biomecardio.com/MUST/) — reference DAS + GPU wrappers; Field II (https://field-ii.dk/) — simulation toolbox (CPU, but generates RF data for GPU DAS); k-Wave CUDA (https://github.com/klepo/k-Wave-Fluid-CUDA) — CUDA time-domain acoustic propagation for full-wave ultrasound.
- **CUDA libraries & GPU pattern:** cuBLAS for element-weighted summation; custom CUDA kernel: one thread per image pixel, loads element positions into shared memory, vectorized delay computation via `__fmaf_rn`; texture fetch for interpolated RF data; coalesced global memory access across scan-line dimension.

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
