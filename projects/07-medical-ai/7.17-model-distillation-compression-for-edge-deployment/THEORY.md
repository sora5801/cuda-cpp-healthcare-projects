# THEORY — 7.17 Model Distillation & Compression for Edge Deployment

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

### 7.17 Model Distillation & Compression for Edge Deployment 🟡 · Active R&D

- **Deep dive:** Compresses large clinical AI models (ViT-L, GPT-style) into small student networks that fit on embedded devices by matching teacher logits, intermediate features, or attention maps. Knowledge distillation training is GPU-bound: both teacher and student run forward passes in every iteration, doubling compute vs. standard training. Structured pruning removes entire channels or attention heads; the resulting sparse model still benefits from GPU execution through efficient sparse tensor routines. INT8 quantisation-aware training (QAT) uses fake-quantisation operators that are CUDA-kernel-friendly and allow recovery of accuracy on medical benchmarks.
- **Key algorithms:** Soft label distillation (Hinton et al.), feature matching (FitNets), attention transfer, data-free distillation, structured/unstructured pruning, magnitude-based weight pruning, quantisation-aware training (QAT), GPTQ, AWQ, LoRA for student warm-start.
- **Datasets:**
  - ImageNet (pre-training teachers) + CheXpert (domain fine-tuning) — dual-dataset compression pipeline
  - MedMNIST — small-scale medical benchmark for student evaluation (https://medmnist.com/)
  - PTB-XL — ECG dataset for waveform model compression evaluation (https://physionet.org/content/ptb-xl/)
  - LIDC-IDRI — CT nodule dataset for compressed segmentation model evaluation (https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
- **Starter repos/tools:**
  - TensorRT (https://github.com/NVIDIA/TensorRT) — quantisation and layer fusion for medical inference
  - Intel Neural Compressor (https://github.com/intel/neural-compressor) — INT8 QAT and pruning on GPU/CPU
  - Pytorch-Distiller (verify URL) — knowledge distillation toolkit
  - Once-for-All (https://github.com/mit-han-lab/once-for-all) — NAS + distillation for efficient medical model families
- **CUDA libraries & GPU pattern:** cuDNN for both teacher/student simultaneous forward passes, TensorRT PTQ/QAT calibration; pattern: data-parallel joint teacher-student training with teacher frozen on GPU.

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
