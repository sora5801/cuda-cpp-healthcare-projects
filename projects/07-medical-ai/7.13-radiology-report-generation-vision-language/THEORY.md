# THEORY — 7.13 Radiology Report Generation (Vision-Language)

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

### 7.13 Radiology Report Generation (Vision-Language) 🟡 · Active R&D

- **Deep dive:** Generates free-text radiology reports from chest X-rays, CT, or MRI scans by jointly encoding the image and decoding text autoregressively. This is a vision-language task requiring large cross-modal attention blocks: ViT image encoder + GPT-style decoder with cross-attention, both bottlenecked by CUDA tensor-core matrix multiplies. Generating a 200-word radiology report at inference requires hundreds of autoregressive decoder steps, each a full forward pass through a multi-layer transformer; batch decoding on GPU with KV-caching provides the necessary throughput for clinical deployment. Training requires paired image-report datasets and auxiliary pathology-label supervision, running on multi-GPU clusters.
- **Key algorithms:** Cross-modal attention, Vision Transformer (ViT) encoder, GPT decoder, contrastive image-text pretraining (CLIP), CheXpert labeller for evaluation, RadGraph F1 metric, layer-wise anatomical attention, chain-of-thought report generation, LEAD / LLaVA-TA architectures.
- **Datasets:**
  - MIMIC-CXR — 227,827 chest X-ray + report pairs (https://physionet.org/content/mimic-cxr/)
  - CheXpert — 224k X-rays with pathology labels (https://stanfordmlgroup.github.io/competitions/chexpert/)
  - OpenI — Indiana University chest X-ray + report dataset (https://openi.nlm.nih.gov/)
  - PadChest — 160k chest X-rays with 174-label taxonomy (https://bimcv.cipf.es/bimcv-projects/padchest/)
- **Starter repos/tools:**
  - Awesome-Radiology-Report-Generation (https://github.com/mk-runner/Awesome-Radiology-Report-Generation) — curated paper/dataset/code list
  - R2Gen / R2GenCMN (https://github.com/cuhksz-nlp/R2Gen) — seminal cross-modal radiology generation models
  - MIMIC-CXR multimodal repo (https://github.com/yuanditang/MIMIC-CXR) — ResNet + LLaMA-3.2 vision-instruction pipeline
  - CheXagent (verify URL) — instruction-tuned radiology report generation model
- **CUDA libraries & GPU pattern:** Flash Attention 2 for cross-modal attention, TensorRT for decoder inference acceleration, KV-cache with CUDA persistent memory; pattern: encoder-decoder parallelism with batched beam search on GPU.

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
