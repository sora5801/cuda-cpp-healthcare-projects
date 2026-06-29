# THEORY — 7.14 Self-Supervised Image Pretraining for Medical Imaging

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

### 7.14 Self-Supervised Image Pretraining for Medical Imaging 🟡 · Active R&D

- **Deep dive:** Pre-trains visual encoders on large unlabelled medical image collections using contrastive or masked image modelling objectives, so downstream tasks (classification, segmentation) require only small labelled datasets. SimCLR, MoCo-v3, DINO, and MAE all reduce to large batched matrix multiplications during projection head training and attention; MoCo avoids the large-batch requirement of SimCLR by using a momentum encoder and GPU-resident queue of negatives. Medical images differ from natural images (greyscale, 3D, complex domain shifts), requiring domain-specific augmentation policies. GPU memory is the primary constraint: SimCLR needs large batch (4096+) to fill the negative pool, demanding multi-GPU with NCCL all-reduce.
- **Key algorithms:** SimCLR, MoCo-v2/v3, BYOL, SimSiam, DINO, MAE (Masked Autoencoder), MoCo-CXR (chest X-ray adaptation), momentum encoder, projection head, cosine-similarity loss, online-vs-target network paradigm.
- **Datasets:**
  - ChestX-ray14 / NIH — 112k chest X-rays with 14 disease labels (https://nihcc.app.box.com/v/ChestXray-NIHCC)
  - MIMIC-CXR — 227k X-rays for self-supervised pretraining (https://physionet.org/content/mimic-cxr/)
  - RadImageNet — 1.35M radiology images across CT/MRI/US for SSL pretraining (verify URL)
  - BraTS + TCIA — unlabelled MRI/CT volumes for 3D MAE pretraining (https://www.cancerimagingarchive.net/)
- **Starter repos/tools:**
  - MoCo-CXR (https://arxiv.org/abs/2010.05352) — MoCo applied to chest X-ray, code available (verify URL)
  - MONAI self-supervised (https://github.com/Project-MONAI/research-contributions) — MAE and contrastive pretraining for 3D medical images
  - DINO (https://github.com/facebookresearch/dino) — self-supervised ViT; adaptable to medical imaging
  - lightly (https://github.com/lightly-ai/lightly) — SSL framework supporting SimCLR/DINO/MoCo on GPU
- **CUDA libraries & GPU pattern:** NCCL all-reduce for multi-GPU negative queue synchronisation, cuDNN for backbone convolutions, cuBLAS for projection head; pattern: large-batch contrastive with NCCL gradient sync, or momentum queue stored in GPU SRAM.

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
