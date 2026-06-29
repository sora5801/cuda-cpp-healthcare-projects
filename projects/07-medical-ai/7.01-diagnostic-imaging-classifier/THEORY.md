# THEORY — 7.1 Diagnostic Imaging Classifier

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

### 7.1 Diagnostic Imaging Classifier 🟢 · Established

- **Deep dive:** Trains convolutional and transformer-based networks to classify pathologies (malignancy, disease grade, anatomical anomaly) from 2D/3D medical images — CT, MRI, X-ray, ultrasound. GPUs provide the tensor-parallel matrix multiply needed to process high-resolution volumetric input in minibatches; a single 512×512 CT slice stack can reach tens of millions of pixels. Backbone convolutions (3D U-Net, ResNet-50, EfficientNet, ViT-B) are the compute-dominant operation, mapping directly onto CUDA tensor cores. Mixed-precision FP16/BF16 training via cuDNN doubles effective throughput versus FP32 while preserving classification accuracy. Inference on edge devices is further accelerated with TensorRT INT8 quantisation.
- **Key algorithms:** 3D convolutional neural networks (ResNet-3D, DenseNet), Vision Transformers (ViT, Swin-T), EfficientNet, data augmentation with random affine/elastic transforms, AUC-optimised losses, Grad-CAM explainability, TTA (test-time augmentation) ensembling.
- **Datasets:**
  - MIMIC-CXR — 227,827 labelled chest X-ray studies with radiology reports from Beth Israel Deaconess (https://physionet.org/content/mimic-cxr/)
  - CheXpert — 224,316 chest X-rays from Stanford, 14 pathology labels (https://stanfordmlgroup.github.io/competitions/chexpert/)
  - LIDC-IDRI — 1,018 CT lung nodule cases with radiologist consensus annotations (https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
  - The Cancer Imaging Archive (TCIA) — multi-modal oncology imaging across dozens of curated collections (https://www.cancerimagingarchive.net/)
- **Starter repos/tools:**
  - MONAI (https://github.com/Project-MONAI/MONAI) — PyTorch-native medical imaging framework with C++/CUDA extensions for resampling and transforms
  - TorchXRayVision (https://github.com/mlmed/torchxrayvision) — pre-trained chest X-ray models, loaders for CheXpert/MIMIC-CXR
  - nnU-Net (https://github.com/MIC-DKFZ/nnUNet) — auto-configuring segmentation/classification baseline that wins most medical imaging benchmarks
  - TotalSegmentator (https://github.com/wasserth/TotalSegmentator) — 104-structure CT segmentation built on nnU-Net
- **CUDA libraries & GPU pattern:** cuDNN for convolution kernels, NCCL for multi-GPU data-parallel training, TensorRT for deployment; pattern: minibatch data parallelism with NCCL all-reduce, optional model parallelism for 3D volumes that exceed single-GPU VRAM.

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
