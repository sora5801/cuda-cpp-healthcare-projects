# THEORY — 7.18 Retinal Fundus AI Screening

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

### 7.18 Retinal Fundus AI Screening 🟢 · Established

- **Deep dive:** Classifies diabetic retinopathy, glaucoma, and age-related macular degeneration from colour fundus photographs or OCT scans. High-resolution fundus images (typically 2048×2048) require significant GPU memory for batch processing; ResNet, EfficientNet, and Swin-Transformer backbones fine-tuned on annotated fundus datasets are the standard approach. GPU tensor cores accelerate the backbone convolutions in batch; simultaneous inference across both eyes and multiple pathologies (multi-task heads) doubles effective throughput. Real-world screening pipelines process millions of images annually, making GPU throughput a primary operational concern.
- **Key algorithms:** EfficientNet-B4/B5 (winner of EyePACS 2019), Swin Transformer, Grad-CAM for lesion localisation, multi-task classification (DR grade + glaucoma + AMD), self-supervised pretraining on unlabelled fundus images, uncertainty calibration for referral decisions.
- **Datasets:**
  - EyePACS — 88,000 labelled fundus images, 5-grade DR severity (Kaggle, verify URL)
  - APTOS 2019 — 3,662 fundus images, DR grading competition (Kaggle, verify URL)
  - DRIVE / STARE — retinal vessel segmentation datasets (verify URL)
  - UK Biobank Retinal Imaging — 68k retinal fundus images with linked health records (https://www.ukbiobank.ac.uk/)
- **Starter repos/tools:**
  - EfficientDet / EfficientNet (https://github.com/google/automl/tree/master/efficientnet) — strong fundus baselines
  - MONAI (https://github.com/Project-MONAI/MONAI) — fundus classification pipelines
  - RETFound (https://github.com/rmaphoh/RETFound_MAE) — MAE-pretrained retinal foundation model on 1.6M fundus images
  - DeepDR Plus (verify URL) — end-to-end diabetic retinopathy screening system
- **CUDA libraries & GPU pattern:** cuDNN for EfficientNet/Swin convolutions, TensorRT for clinic deployment; pattern: data-parallel fine-tuning on high-resolution fundus batches with gradient accumulation.

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
