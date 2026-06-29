# THEORY — 4.27 Radiomics Feature Extraction

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

### 4.27 Radiomics Feature Extraction 🟡 · Active R&D
- **Deep dive:** Radiomics extracts hundreds of quantitative features (shape, first-order statistics, texture: GLCM, GLRLM, GLSZM, NGTDM) from 3D segmented ROIs in CT/PET/MRI. For a cohort of 10,000 patients with large ROIs (~10⁶ voxels each), CPU-based PyRadiomics takes 10–30 min per patient; GPU-accelerated cuRadiomics and PyRadiomics-CUDA achieve 143× speedup by parallelizing all histogram and co-occurrence matrix computations across voxels on GPU. Texture features require computing co-occurrence matrices from 26 3D neighbor directions simultaneously — each direction's computation is independent, enabling massive GPU parallelism. Radiomics biomarker discovery pipelines must process thousands of scans for statistical power.
- **Key algorithms:** GLCM (gray-level co-occurrence matrix), GLRLM (run-length matrix), GLSZM (size-zone matrix), NGTDM (neighborhood gray-tone difference matrix), first-order statistics, 3D shape descriptors, wavelet-decomposition features, multi-scale radiomics, IBSI (Image Biomarker Standardization Initiative) compliant features.
- **Datasets:** TCIA NSCLC-Radiomics (https://www.cancerimagingarchive.net/collection/nsclc-radiomics/) — 422 lung CTs with survival; RIDER Breast MRI (via TCIA); QIN-HEADNECK (via TCIA) — head and neck RT; TCGA collections (https://portal.gdc.cancer.gov/).
- **Starter repos/tools:** PyRadiomics-CUDA (https://arxiv.org/abs/2510.02894 — code on https://github.com/mis-wut/pyradiomics-CUDA) — GPU radiomics, 143× speedup; cuRadiomics (verify URL — published in AAPM proceedings) — CUDA texture/GLCM GPU extraction; PyRadiomics CPU baseline (https://github.com/AIM-Harvard/pyradiomics) — IBSI-compliant reference; MONAI (https://github.com/Project-MONAI/MONAI) — integrated GPU radiomics pipeline.
- **CUDA libraries & GPU pattern:** Custom CUDA for co-occurrence matrix (atomic add into per-direction GLCM per thread block); shared memory for voxel neighborhood; parallel histogram across all voxels (CUB block histogram); warp-level reductions for matrix statistics.

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
