# THEORY — 4.7 Medical Image Segmentation (Deep Learning)

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

### 4.7 Medical Image Segmentation (Deep Learning) 🟢 · Established
- **Deep dive:** Volumetric segmentation of anatomical structures (organs, tumors, vessels) in CT/MRI using encoder-decoder CNNs operates on 3D patches or whole volumes; a 512×512×200 CT volume processed in a 3D U-Net with standard batch size requires ~16 GB GPU memory and ~200 GFLOPS per forward pass. nnU-Net automatically configures patch size, batch size, network topology, and augmentation to dataset fingerprints, making it a strong universal baseline. Inference of whole-body CT (TotalSegmentator, 117 structures) completes in 20–50 s on a GPU vs. 40–50 min on CPU. Transformer architectures (Swin-UNETR) add self-attention with quadratic memory cost in sequence length, further motivating large-VRAM GPUs.
- **Key algorithms:** 3D U-Net, nnU-Net, Swin-UNETR, TransUNet, DeepMedic, V-Net, residual encoder-decoder, cascaded networks, multi-scale feature pyramid, conditional random fields (CRF) post-processing, semi-supervised pseudo-labeling.
- **Datasets:** Medical Segmentation Decathlon (http://medicaldecathlon.com/) — 10 tasks, ~2,500 volumes total; TotalSegmentator training set (Zenodo, ~1,200 CT with 117 structure labels; https://zenodo.org/record/6802614); KiTS23 kidney tumor challenge (https://kits-challenge.org/kits23/); BraTS brain tumor dataset (https://www.synapse.org/#!Synapse:syn27046444).
- **Starter repos/tools:** nnU-Net (https://github.com/MIC-DKFZ/nnUNet) — self-configuring, handles 2D/3D, GPU training and inference; TotalSegmentator (https://github.com/wasserth/TotalSegmentator) — 117-class whole-body CT segmentation, GPU inference in <1 min; MONAI (https://github.com/Project-MONAI/MONAI) — PyTorch medical AI framework with GPU-optimized transforms and network zoo; Swin-UNETR reference (https://github.com/Project-MONAI/research-contributions) — transformer-based 3D segmentation.
- **CUDA libraries & GPU pattern:** cuDNN (3D convolutions), Tensor Cores (FP16/BF16), CUDA Unified Memory for large volumes; mixed-precision training; patch-based inference with sliding window; multi-GPU via PyTorch DDP + NCCL; GPU-resident data augmentation (random flips, elastic deformations) via MONAI or NVIDIA DALI.

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
