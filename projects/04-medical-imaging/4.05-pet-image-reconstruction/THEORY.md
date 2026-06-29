# THEORY — 4.5 PET Image Reconstruction

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

### 4.5 PET Image Reconstruction 🟡 · Active R&D
- **Deep dive:** Positron Emission Tomography (PET) detects coincident 511 keV gamma pairs and inverts the resulting sinogram to recover tracer-distribution volumes. Maximum-likelihood expectation-maximization (MLEM) and its ordered-subsets accelerator OS-EM dominate clinically; each EM iteration requires a full system-matrix forward-projection and backprojection, accounting for detector geometry, attenuation, scatter, and randoms. Modern scanners produce list-mode data at ~10⁸ events and sinograms with ~10⁹ elements; a single MLEM iteration on a clinical dataset takes seconds on CPU, motivating GPU parallelization of the projection step across LORs (lines of response). Dynamic PET adds a time dimension, multiplying reconstruction cost by the number of frames.
- **Key algorithms:** MLEM, OS-EM (Hudson-Larkin), RAMLA, MAP-EM with Gibbs priors, PSF (point spread function) modelling, TOF-PET reconstruction (time-of-flight), list-mode ML-EM, PET/MRI joint reconstruction, penalized likelihood with MR-guided priors.
- **Datasets:** OpenNEURO PET datasets (https://openneuro.org/); TCIA PET collections (https://www.cancerimagingarchive.net/); PETRIC challenge datasets (https://github.com/SyneRBI/PETRIC); Siemens mMR phantom datasets (publicly available through STIR/SIRF).
- **Starter repos/tools:** STIR (Software for Tomographic Image Reconstruction, https://github.com/SyneRBI/STIR) — C++, OS-EM, TOF, scatter, CUDA via parallelproj; SIRF (Synergistic Image Reconstruction Framework, https://github.com/SyneRBI/SIRF) — Python/MATLAB wrapper around STIR + Gadgetron for joint PET/MR; parallelproj (https://github.com/gschramm/parallelproj) — CUDA/OpenCL GPU projectors for PET; CASToR (https://castor-project.org/) — multi-threaded/GPU-capable PET/SPECT reconstruction (verify URL).
- **CUDA libraries & GPU pattern:** Custom CUDA kernels for LOR-parallel projection; cuBLAS for correction factors; warp-level reduction for scatter estimation; one thread per LOR in forward/back projection; CUDA streams for overlapping compute and host-device transfer.

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
