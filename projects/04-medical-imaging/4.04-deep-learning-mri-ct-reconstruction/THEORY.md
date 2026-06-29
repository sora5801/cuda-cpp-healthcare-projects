# THEORY — 4.4 Deep-Learning MRI/CT Reconstruction

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

### 4.4 Deep-Learning MRI/CT Reconstruction 🟡 · Active R&D
- **Deep dive:** Learned reconstruction networks replace hand-crafted priors with data-driven mappings from under-sampled/degraded k-space or sinogram to fully-sampled images. End-to-end variational networks (E2E-VarNet) unroll gradient descent iterations as network layers, each with trainable sensitivity maps and refinement modules; these run entirely on GPU during both training (batch gradient descent) and inference. Training on large multi-coil raw k-space datasets (fastMRI) requires TB-scale data loading with GPU-pinned memory and mixed-precision FP16/BF16 tensor cores. Inference at 256² × 32 coils can achieve sub-100 ms per volume on a single A100, enabling real-time clinical deployment.
- **Key algorithms:** E2E-VarNet (variational network with learned sensitivity maps), unrolled ADMM-Net, deep cascade of CNN, U-Net in image domain, score-based diffusion models for MRI (DiffusionMBIR), plug-and-play denoising priors, recurrent unrolled networks.
- **Datasets:** fastMRI (https://fastmri.med.nyu.edu/) — raw multi-coil k-space, knee/brain, gold-standard reference; fastMRI+ with radiologist annotations (https://github.com/StanfordMIMI/fastMRI_plus); 2016 AAPM Low-Dose CT Challenge for CT reconstruction learning.
- **Starter repos/tools:** fastMRI baseline code (https://github.com/facebookresearch/fastMRI) — PyTorch E2E-VarNet, U-Net, evaluation scripts; BART (https://github.com/mrirecon/bart) — Deep MRI reconstruction via BART-learn module; Direct (https://github.com/directgroup/direct) — modular PyTorch framework for DL MRI reconstruction (multiple unrolled architectures); Hugging Face Medical Imaging (https://huggingface.co/datasets?search=mri) — model hub with pretrained MRI reconstruction checkpoints.
- **CUDA libraries & GPU pattern:** cuDNN (conv layers), Tensor Cores (FP16 mixed precision), PyTorch CUDA autograd; pipeline: data → pinned host memory → GPU → network forward pass → loss → backward; multi-GPU DDP training via NCCL.

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
