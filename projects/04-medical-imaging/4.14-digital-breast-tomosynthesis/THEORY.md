# THEORY — 4.14 Digital Breast Tomosynthesis

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

### 4.14 Digital Breast Tomosynthesis 🟡 · Active R&D
- **Deep dive:** Digital breast tomosynthesis (DBT) acquires 9–25 low-dose projections over a limited angular range (~15–50°), then reconstructs thin slabs through compressed breast tissue. The limited-angle geometry makes analytical FBP unstable, so iterative methods (OS-EM, SART, ASD-POCS) with total-variation regularization dominate for artifact reduction. The breast is a low-contrast, soft-tissue object where noise and blur from the limited angle severely reduce lesion conspicuity, making statistical reconstruction critical. A single DBT volume (~800 × 700 × 60 slices at 85 µm) represents ~30 GB of raw projection data; GPU acceleration reduces OS-EM reconstruction from hours to under a minute. Deep learning methods (U-Net denoising on FBP outputs) additionally require GPU for inference.
- **Key algorithms:** FBP with limited-angle filter, OS-EM (ordered-subsets EM), SART, ASD-POCS with total variation, model-based iterative reconstruction (MBIR), DBT-specific PSF/MTF modelling, deep learning denoising and artifact reduction, mass detection CNNs.
- **Datasets:** OPTIMAM Mammography Image Database (OMI-DB, access via ICR UK); CBIS-DDSM (https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM) — 2,620 mammograms via TCIA; VinDr-Mammo (https://physionet.org/content/vindr-mammo/1.0.0/); BCS-DBT (Duke DBT challenge dataset, https://bcs-dbt.grand-challenge.org/).
- **Starter repos/tools:** ASTRA Toolbox (https://github.com/astra-toolbox/astra-toolbox) — GPU forward/back-projection for arbitrary cone-beam geometry; RTK (https://github.com/RTKConsortium/RTK) — FDK and iterative DBT-capable; TIGRE (https://github.com/CERN/TIGRE) — DBT-compatible geometry; OpenDBT (verify URL) — research-focused DBT reconstruction framework.
- **CUDA libraries & GPU pattern:** cuFFT for ramp filter; CUDA voxel-driven backprojection with compressed breast geometry; texture memory for projection interpolation; limited-angle geometry stored in constant memory; ADMM inner loop GPU-resident.

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
