# THEORY — 3.18 Protein Language Model Inference

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

### 3.18 Protein Language Model Inference 🟡 · Active R&D
- **Deep dive:** Protein language models (PLMs) such as Meta's ESM-2 (650 M–15 B parameters) learn evolutionary constraints from hundreds of millions of protein sequences; their residue embeddings encode structure, function, and mutational effects. ESMFold uses ESM-2 as a trunk to predict 3D structure without MSA, making it dramatically faster than AlphaFold2 for single-sequence predictions. GPU acceleration of the multi-head self-attention layers (O(L²) per layer for sequence length L) is essential—H100 Tensor Cores achieve >3× MFU for these GEMM workloads. Inference of 10 M UniProt proteins via ESMFold required a dedicated GPU cluster; GPU batching of mixed-length proteins with padding optimisation is the key engineering challenge.
- **Key algorithms:** Transformer multi-head self-attention (Q×K^T scaling, softmax, V aggregation); rotary positional embeddings; evoformer-style structure module; invariant point attention (IPA); masked language model (MLM) training; FlashAttention memory-efficient attention.
- **Datasets:** UniRef50/90 — training corpus for PLMs (https://www.uniprot.org/help/uniref); ESM Metagenomic Atlas — 700 M metagenomic protein structures (https://esmatlas.com/); PDB structures — validation set for ESMFold (https://www.rcsb.org/); CATH / SCOP — structural classification databases (https://www.cathdb.info/).
- **Starter repos/tools:** fair-esm (https://github.com/facebookresearch/esm) — Meta's ESM-2 and ESMFold, official CUDA inference code; EvolutionaryScale ESM3 (https://github.com/evolutionaryscale/esm) — latest multimodal protein model; ColabFold (https://github.com/sokrypton/ColabFold) — fast MSA + AlphaFold2 on GPU; xTrimoPGLM (https://huggingface.co/BonjwrAI/xTrimoPGLM-100B) — 100 B protein LM (verify URL).
- **CUDA libraries & GPU pattern:** cuDNN / Apex / FlashAttention-2 for attention; cuBLAS GEMM for feed-forward layers; Tensor Core FP16/BF16 mixed precision; multi-GPU tensor + pipeline parallelism (Megatron-LM / DeepSpeed); dynamic batching by sequence length bucket.

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
