# THEORY — 7.11 Medical Foundation-Model Pretraining & Inference

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

### 7.11 Medical Foundation-Model Pretraining & Inference 🟡 · Active R&D

- **Deep dive:** Pretrains large-scale (1B–70B parameter) language, vision, or multimodal models on domain-specific medical corpora — PubMed, MIMIC clinical notes, radiology report databases, pathology image collections — to produce general-purpose medical representations. Pretraining is massively GPU-bound: the matrix multiplications in transformer attention and feed-forward layers constitute >95% of FLOPs. Tensor-parallel and pipeline-parallel model partitioning across hundreds of A100/H100 GPUs (via Megatron-LM or DeepSpeed) is necessary for 70B-parameter models. Inference serving uses Flash Attention, continuous batching (vLLM), and INT8/GPTQ quantisation to handle concurrent clinical queries.
- **Key algorithms:** Autoregressive pretraining (GPT), masked language modelling (BERT), instruction tuning (SFT + RLHF), Vision-Language Contrastive pretraining (CLIP, FLAVA), Mixture-of-Experts (MoE), FlashAttention-2, LoRA/QLoRA fine-tuning, GPTQ quantisation.
- **Datasets:**
  - PubMed Central Open Access — 4M+ full biomedical articles (https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/)
  - MIMIC-IV Notes — 331,794 clinical notes (https://physionet.org/content/mimic-iv-note/)
  - The Pile: Pile-MedMent / S2ORC — broad scientific pretraining corpora (https://pile.eleuther.ai/)
  - OpenPath / PathCap — pathology image-caption pairs for vision-language pretraining (verify URL)
- **Starter repos/tools:**
  - MEDITRON (https://github.com/epfLLM/meditron) — Llama-2 70B adapted for medicine with GPU pretraining scripts
  - Awesome Healthcare Foundation Models (https://github.com/Jianing-Qiu/Awesome-Healthcare-Foundation-Models) — curated model list
  - Awesome Foundation Models in Medical Imaging (https://github.com/xmindflow/Awesome-Foundation-Models-in-Medical-Imaging) — curated vision-language models
  - vLLM (https://github.com/vllm-project/vllm) — continuous batching inference engine for serving medical LLMs on GPU
- **CUDA libraries & GPU pattern:** Megatron-LM tensor parallelism, DeepSpeed ZeRO, Flash Attention 2, NCCL all-reduce; pattern: 3D parallelism (tensor × pipeline × data), NVLink high-bandwidth GPU fabric required.

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
