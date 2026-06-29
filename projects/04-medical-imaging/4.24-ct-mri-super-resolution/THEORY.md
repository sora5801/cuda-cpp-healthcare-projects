# THEORY — 4.24 CT/MRI Super-Resolution

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

### 4.24 CT/MRI Super-Resolution 🟡 · Active R&D
- **Deep dive:** Clinical CT/MRI is acquired at sub-optimal resolution due to dose constraints, scan time, or scanner capability; super-resolution (SR) enhances images 2–4× isotropically using deep neural networks. For MRI, anisotropic SR (thick slice → isotropic) upsamples a 5 mm axial slice to 1 mm coronal/sagittal using networks trained on pairs of high/low-resolution volumes. GANs (ESRGAN-Med, CycleGAN) generate perceptually sharp images; diffusion SR models produce hallucination-free probabilistic outputs. Processing a 512×512×100 CT volume through a 3D ESRGAN requires ~500 GFLOPS per forward pass; clinical deployment requires TensorRT-optimized inference at <5 s/volume on a single GPU.
- **Key algorithms:** ESRGAN (enhanced SRGAN), 3D U-Net SR, CycleGAN for unpaired SR, diffusion model SR (SR3, DDPM), subpixel convolution (ICNR), attention U-Net SR, learned upsampling (LIIF), perceptual and adversarial losses, self-supervised SR.
- **Datasets:** HCP 7T/3T paired brain MRI (https://db.humanconnectome.org/); fastMRI (https://fastmri.med.nyu.edu/) — implicitly used for SR evaluation; IXI brain MRI dataset (https://brain-development.org/ixi-dataset/); MSD CT tasks for resolution enhancement.
- **Starter repos/tools:** MONAI SR examples (https://github.com/Project-MONAI/MONAI) — 3D medical SR reference implementations; BasicSR (https://github.com/XPixelGroup/BasicSR) — general GPU SR framework adaptable to medical; SynthSR (https://github.com/BBillot/SynthSR) — multi-contrast MRI SR/synthesis; MedSRGAN (search GitHub for "medical image super resolution GAN").
- **CUDA libraries & GPU pattern:** cuDNN (3D transposed convolutions, pixel shuffle); Tensor Cores for FP16 SR training; gradient penalty in discriminator (cuBLAS); CuPy for efficient patch extraction; TensorRT for INT8/FP16 inference deployment.

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
