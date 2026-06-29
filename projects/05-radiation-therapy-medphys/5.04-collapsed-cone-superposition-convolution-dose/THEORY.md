# THEORY — 5.4 Collapsed-Cone / Superposition-Convolution Dose

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

### 5.4 Collapsed-Cone / Superposition-Convolution Dose 🟢 · Established
- **Deep dive:** Superposition-convolution (SC) dose computation convolves Monte Carlo-derived photon energy-deposition kernels (polyenergetic dose-spread arrays, DSAs) with the TERMA (total energy released per unit mass) computed from CT. Collapsed-cone convolution (CCC) discretizes the kernel into angular cones and propagates dose along ray paths at each angle. For a 512³ CT volume and ~400 cone directions, each cone sweep is a 1D scan along the CT in that direction — embarrassingly parallel across cones and voxels. GPU parallelization across cone directions and voxel planes reduces a CCC plan from ~10 min to <10 s. This algorithm underlies most commercial photon dose engines (Eclipse AXB, RayStation).
- **Key algorithms:** Superposition/convolution with poly-energetic DSA kernels, collapsed-cone convolution (CCC), anisotropic analytical algorithm (AAA), Acuros XB (linear Boltzmann transport), TERMA ray-tracing (Siddon/ray-voxel), heterogeneity correction via density scaling.
- **Datasets:** AAPM TG-105 test cases (heterogeneous media dose benchmarks); IROC lung phantom CT + dosimetry data; TCIA clinical photon planning datasets; CIRS IMRT verification phantom data.
- **Starter repos/tools:** matRad (https://github.com/e0404/matRad) — photon pencil-beam + CC dose engine; Plastimatch (https://plastimatch.org/) — GPU-accelerated dose engine components; CERR (https://github.com/cerr/CERR) — dose calculation framework; open AAPM TG-105 reference datasets with comparison code (verify URL at aapm.org).
- **CUDA libraries & GPU pattern:** Custom CUDA kernel for TERMA ray-trace (Siddon's algorithm, one thread per ray); cone-direction parallel sweep in CCC (one CUDA block per cone direction); shared memory for density strip along current cone ray; reduction for energy normalization.

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
