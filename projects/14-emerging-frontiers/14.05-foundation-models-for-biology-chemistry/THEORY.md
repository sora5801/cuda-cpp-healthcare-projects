# THEORY — 14.5 Foundation Models for Biology & Chemistry

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

### 14.5 Foundation Models for Biology & Chemistry 🟡 · Active R&D

- **Deep dive:** Large pre-trained models on biological sequences (ESM-2: 15B parameters on 250M protein sequences), genomic DNA (Nucleotide Transformer, Evo-1), and chemical SMILES (ChemBERTa, MolGPT) are rapidly becoming universal biological encoders. GPU training at scale (thousands of A100s) is the defining infrastructure requirement; GPU inference is the deployment bottleneck for drug discovery pipelines scoring millions of candidates. Fine-tuning foundation models on task-specific biomedical datasets (DMS, HTS, survival data) achieves state of the art across fitness prediction, structure prediction, and clinical outcome forecasting. AMix-1 (2025) demonstrates mixture-of-experts protein foundation models with test-time scaling.
- **Key algorithms:** Masked language modeling (MLM) pre-training, attention with rotary position encoding (RoPE), LoRA/QLoRA fine-tuning, retrieval-augmented generation for protein databases, multi-modal fusion (sequence + structure + expression), model distillation for edge deployment.
- **Datasets:** UniRef90/UniClust30 — protein sequence clusters for pre-training (https://www.uniprot.org/); PDB (https://www.rcsb.org/) — 230K+ structures for structure-aware pre-training; ChEMBL (https://www.ebi.ac.uk/chembl/) — 2.4M bioactive compounds; NCBI RefSeq — genomic DNA pre-training corpus (https://www.ncbi.nlm.nih.gov/refseq/).
- **Starter repos/tools:** ESM2/ESMFold (https://github.com/facebookresearch/esm) — FAIR protein LLM + structure prediction on GPU; Evo (https://github.com/evo-design/evo) — genomic DNA foundation model (Arc Institute); HuggingFace Transformers (https://github.com/huggingface/transformers) — training/fine-tuning infrastructure; NVIDIA BioNeMo (https://www.nvidia.com/en-us/clara/bionemo/) — GPU-optimized biology foundation model platform.
- **CUDA libraries & GPU pattern:** cuDNN FlashAttention-2 for memory-efficient attention, Tensor Core BF16/FP8 matmuls, NCCL for tensor/pipeline parallelism; pattern: sequence tokenization → distributed data-parallel GPU training → gradient checkpointing for memory → task-specific LoRA fine-tuning → GPU batch inference over candidate libraries.

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
