# THEORY — 10.10 Spinal Biomechanics & Intervertebral Disc Modeling

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

### 10.10 Spinal Biomechanics & Intervertebral Disc Modeling 🟡 · Active R&D

- **Deep dive:** The lumbar spine involves poroelastic disc mechanics, facet-joint contact, and large deformation under combined flexion-compression loads, requiring multi-physics FEA with >500 K DOF per motion segment. GPU parallelism compresses the 97.9% time-reduction already demonstrated in automated MRI-to-FEM pipelines (Frontiers 2024) by further accelerating the PCG solver for the full lumbar assembly. Population virtual trials — evaluating thousands of patient-specific spinal constructs after fusion surgery — run overnight on GPU clusters, replacing months of cadaveric testing. GPU-resident bone-density maps updated with DXA-calibrated HU values enable patient-specific fracture risk prediction on clinical timescales.
- **Key algorithms:** Biphasic/poroelastic FEM (Mow-Holmes disc model), hyperelastic anulus fibrosus (fiber-reinforced), penalty facet-joint contact, bone-remodeling, automated mesh generation (Laplacian smoothing + decimation), shape correspondence via non-rigid ICP.
- **Datasets:** VerSe Challenge — 374 CT scans with vertebral shape annotation (https://verse-challenge.github.io/); MICCAI SpineSeg — lumbar MRI segmentation (verify URL via Grand Challenge); CT Spine Dataset (verse2020, Zenodo) — 355 CTs with vertebral instance masks (https://doi.org/10.5281/zenodo.3755323); OrthoLoad Lumbar — in vivo spinal implant forces (https://orthoload.com/).
- **Starter repos/tools:** FEBio (https://github.com/febiosoftware/FEBio) — built-in biphasic and fiber-reinforced disc models; SpineWeb toolkit — vertebral mesh atlas (http://spineweb.digitalimaginggroup.ca/); TotalSegmentator (https://github.com/wasserth/TotalSegmentator) — fast CT organ+vertebra segmentation for mesh input; MRI-to-FEM pipeline (Frontiers 2024, verify Zenodo for code) — automated lumbar FE model generation.
- **CUDA libraries & GPU pattern:** cuSPARSE PCG for multi-physics coupled system, CUDA kernels for fiber-reinforced anisotropic stress update, cuDNN for DXA HU calibration regression; pattern: GPU-resident CT density map → automatic mesh generation → fiber orientation interpolation on GPU → coupled solid-fluid PCG solve → fracture risk post-processing.

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
