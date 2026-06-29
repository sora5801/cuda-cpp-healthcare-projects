# THEORY — 4.1 CT Reconstruction — Filtered Backprojection

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

### 4.1 CT Reconstruction — Filtered Backprojection 🟢 · Established
- **Deep dive:** Computes a 3D volume from a set of 2D X-ray projections by applying a ramp (Ram-Lak) filter in the frequency domain to each sinogram row, then smearing each filtered projection back across the reconstructed volume. The Feldkamp-Davis-Kress (FDK) algorithm extends this to cone-beam geometry used in modern scanners and linac on-board imagers. GPU acceleration is decisive: for a 512³ volume and 1,000 projections, each backprojection step touches ~10⁹ voxel-projection pairs, making serial CPU execution intractable for real-time or high-resolution use. CUDA texture memory provides hardware-interpolated trilinear sampling of projection data at near-zero extra cost, and the entire backprojection kernel saturates GPU memory bandwidth. Achieving sub-second reconstruction at clinical resolutions requires tens of TFLOPS, available only on GPU.
- **Key algorithms:** Feldkamp-Davis-Kress FBP, Ram-Lak / Shepp-Logan ramp filter, Parker short-scan weighting, GPU ray-driven and voxel-driven backprojection, helical cone-beam FDK with Katsevich exact reconstruction.
- **Datasets:** LUNA16/LIDC-IDRI — 888 annotated thoracic CTs from TCIA (https://luna16.grand-challenge.org/); TCIA (The Cancer Imaging Archive) — large multi-collection public CT/MRI archive (https://www.cancerimagingarchive.net/); LoDoPaB-CT — low-dose CT sinogram/reconstruction pairs for benchmarking (https://zenodo.org/record/3384092); 2016 AAPM Low-Dose CT Grand Challenge — paired full-/quarter-dose CT scans (https://www.aapm.org/grandchallenge/lowdosect/).
- **Starter repos/tools:** RTK (RTKConsortium/RTK, https://github.com/RTKConsortium/RTK) — ITK-based, GPU FDK and iterative, multi-GPU, clinical DICOM-RT support; ASTRA Toolbox (https://astra-toolbox.com/, https://github.com/astra-toolbox/astra-toolbox) — MATLAB/Python/C++ GPU forward/back-projection primitives for 2D/3D, supports fan/cone/parallel; TIGRE (https://github.com/CERN/TIGRE) — MATLAB/Python CUDA toolbox with FDK plus 10+ iterative algorithms, real-dataset focus; Plastimatch (https://plastimatch.org/) — GPU FDK, deformable registration, DRR; open-source, clinical-grade C++.
- **CUDA libraries & GPU pattern:** cuFFT (ramp filter in k-space), CUDA texture memory (hardware trilinear backprojection interpolation), cuBLAS; kernel pattern: one CUDA thread per output voxel, loops over projections; multi-GPU split over projection subsets.

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
