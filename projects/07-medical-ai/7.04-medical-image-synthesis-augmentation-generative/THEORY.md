# THEORY — 7.4 Medical Image Synthesis & Augmentation (Generative)

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

### 7.4 Medical Image Synthesis & Augmentation (Generative) 🟡 · Active R&D

- **Deep dive:** Generates synthetic medical images to augment scarce annotated datasets or enable domain transfer — e.g., synthesising MRI from CT, generating rare pathology variants, or creating paired segmentation masks. Generative models (GANs, diffusion models, VAEs) are training-compute-intensive: a diffusion UNet iterates 1000 denoising steps at full 3D resolution, with each forward pass bottlenecked by 3D convolutions. GANs require simultaneous forward/backward passes through discriminator and generator on the same GPU batch. GPU parallelism over the spatial dimensions of 3D volumes provides the necessary throughput. Diffusion models in particular benefit from mixed-precision training and gradient checkpointing to fit large 3D UNets in GPU memory.
- **Key algorithms:** Denoising Diffusion Probabilistic Models (DDPM), Score-Based Generative Models (SGBM), CycleGAN, Pix2Pix, VQVAE, Latent Diffusion Models (LDM), FID/FRD evaluation metrics, style-transfer augmentation.
- **Datasets:**
  - BraTS (Brain Tumor Segmentation) — multi-institutional MRI with ground-truth tumour masks (https://www.synapse.org/Synapse:syn51156910/wiki/)
  - ADNI (Alzheimer's Disease Neuroimaging Initiative) — longitudinal MRI/PET with clinical data (https://adni.loni.usc.edu/)
  - TCIA — public CT/MRI collections enabling synthesis training (https://www.cancerimagingarchive.net/)
  - GaNDLF-Synth benchmark (https://arxiv.org/abs/2410.00173) — multi-site synthetic pathology image benchmark (verify URL)
- **Starter repos/tools:**
  - MONAI Generative (https://github.com/Project-MONAI/GenerativeModels) — diffusion, VQVAE, GAN modules on GPU
  - SynthSeg (https://github.com/BBillot/SynthSeg) — label-conditioned MRI synthesis for segmentation
  - MedSynAnalyser / StableDiffusion-Medical (verify URL) — medical image fine-tuning pipelines for latent diffusion
  - HealthyGAN / CycleGAN-3D (verify URL) — unpaired MRI-CT translation
- **CUDA libraries & GPU pattern:** cuDNN 3D grouped convolutions, FlashAttention for diffusion attention blocks, NCCL for multi-GPU; pattern: data-parallel training with gradient checkpointing, NVLink for A100/H100 inter-GPU communication during large 3D batch training.

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
