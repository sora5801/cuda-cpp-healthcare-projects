# THEORY — 14.12 Cross-Modal "Virtual Staining" & Label-Free Imaging

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

### 14.12 Cross-Modal "Virtual Staining" & Label-Free Imaging 🟡 · Active R&D

- **Deep dive:** Virtual staining uses deep learning to predict H&E, IHC, or other chemical stain images from label-free optical modalities (autofluorescence, quantitative phase, CARS, FTIR), eliminating destructive sample preparation. Pixel super-resolved virtual staining via diffusion models (Nature Communications 2025) achieves pathologist-grade tissue diagnostics from autofluorescence alone on GPU. GPU acceleration is essential: a single whole-slide image (100,000 × 100,000 pixels) requires tiled U-Net inference over ~10,000 patches per slide. Clinical-grade validation of autofluorescence virtual staining for prostate cancer (medRxiv 2024) demonstrates diagnostic equivalence to H&E. The GPU also enables real-time virtual staining during surgery for fresh frozen section replacement.
- **Key algorithms:** U-Net/ViT image translation (pix2pix, CycleGAN, diffusion model), pixel super-resolution (ESRGAN, diffusion), Fourier ptychographic reconstruction, stimulated Raman spectral unmixing on GPU, multi-modal image registration (DRIT++), diffusion-model inversion for unpaired translation.
- **Datasets:** Virtual Staining Dataset (Ozcan Lab, UCLA) — autofluorescence → H&E paired images (verify URL via nature.com supplementary); LCI-PARIS — unstained label-free vs. H&E pairs (verify URL); TCGA Digital Pathology Whole-Slide Images (https://portal.gdc.cancer.gov/); Human Protein Atlas — multimodal tissue images (https://www.proteinatlas.org/).
- **Starter repos/tools:** MONAI (https://github.com/Project-MONAI/MONAI) — GPU medical image segmentation + translation; pix2pix/CycleGAN (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) — paired/unpaired GPU image translation; Stable Diffusion (huggingface) fine-tuned for pathology virtual staining; HistoStar (https://github.com/TissueImageAnalytics/tiatoolbox) — GPU whole-slide image analysis toolkit.
- **CUDA libraries & GPU pattern:** cuDNN for U-Net/ViT inference, Tensor Core FP16 for batch patch processing, cuFFT for Fourier ptychographic phase reconstruction; pattern: WSI tiled into 256×256 patches → GPU batch U-Net inference → tile stitching → GPU super-resolution → virtual H&E output for pathologist review.

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
