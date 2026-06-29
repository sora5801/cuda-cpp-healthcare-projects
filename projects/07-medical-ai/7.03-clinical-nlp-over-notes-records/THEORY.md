# THEORY — 7.3 Clinical NLP over Notes & Records

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

### 7.3 Clinical NLP over Notes & Records 🟢 · Established

- **Deep dive:** Applies transformer language models to de-identified electronic health record (EHR) free-text — discharge summaries, radiology reports, nursing notes — for named entity recognition, relation extraction, ICD coding, phenotyping, and clinical event prediction. BERT-style pretraining on billions of clinical tokens (MIMIC-IV notes) is highly GPU-bound: multi-head self-attention scales O(n²) in sequence length, making long-document clinical notes particularly expensive. Flash Attention reduces this cost from O(n²) to near-linear in memory, enabling 8192-token context windows. The parallel bottleneck is the batched matrix multiplications in each transformer layer, exploiting GPU tensor cores. Fine-tuning on task-specific clinical benchmarks (NER, RE) requires additional GPU compute for gradient accumulation across long sequences.
- **Key algorithms:** BERT masked language modelling, next-sentence prediction, Flash Attention, Rotary Positional Embeddings (RoPE), BIO-tagging for NER, CRF output layers, relation extraction with span pairs, multi-label ICD classification, instruction-tuning with clinical instruction sets.
- **Datasets:**
  - MIMIC-IV Clinical Notes — 331,794 de-identified patient notes from Beth Israel Deaconess (https://physionet.org/content/mimic-iv-note/)
  - i2b2/n2c2 NLP Challenge Datasets — named entity, coreference, and relation tasks in clinical text (https://n2c2.dbmi.hms.harvard.edu/)
  - MTSamples — 4,999 transcribed medical reports across 40 specialties (https://mtsamples.com/)
  - MedQA / MedMCQA — medical question answering benchmarks for evaluating clinical LLMs (verify URL)
- **Starter repos/tools:**
  - BioClinicalBERT (https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) — BERT pretrained on MIMIC-III notes
  - Clinical ModernBERT (https://github.com/Simonlee711/Clinical_ModernBERT) — ModernBERT fine-tuned on 13B tokens of PubMed + MIMIC-IV with 8192-token context
  - medSpaCy (https://github.com/medspacy/medspacy) — spaCy-based clinical NLP pipeline with GPU inference support
  - GatorTron (https://huggingface.co/UFNLP/gatortron-base) — large clinical LLM pretrained on 82B tokens of clinical text (verify URL)
- **CUDA libraries & GPU pattern:** Flash Attention 2, cuBLAS for GEMM-dominated transformer layers, NCCL for data-parallel pretraining; pattern: data parallelism across multiple A100/H100 GPUs, gradient checkpointing to fit long-context batches in VRAM.

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
