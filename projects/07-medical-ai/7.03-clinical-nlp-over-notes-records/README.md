# 7.3 — Clinical NLP over Notes & Records

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Medical%20AI%20%26%20Clinical%20Deep%20Learning-lightgrey)

> **🟢 Beginner · Established** — Domain 7: Medical AI & Clinical Deep Learning · Catalog ID `7.3`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

<!-- =======================================================================
     SCAFFOLD STATUS: this README was stamped from the catalog. The prose
     fields below (Deep dive / Algorithms / Datasets / Prior art) are filled
     in from the catalog. Sections marked TODO(impl)/TODO(theory) must be
     completed by the project author before this project is "done"
     (see CLAUDE.md §4.1 and tools/verify_project.py).
     ======================================================================= -->

## Summary

TODO(impl): One paragraph, plain language — what this project does and why a
learner should care. (Seed from the deep dive below.)

## What this computes & why the GPU helps

Applies transformer language models to de-identified electronic health record (EHR) free-text — discharge summaries, radiology reports, nursing notes — for named entity recognition, relation extraction, ICD coding, phenotyping, and clinical event prediction. BERT-style pretraining on billions of clinical tokens (MIMIC-IV notes) is highly GPU-bound: multi-head self-attention scales O(n²) in sequence length, making long-document clinical notes particularly expensive. Flash Attention reduces this cost from O(n²) to near-linear in memory, enabling 8192-token context windows. The parallel bottleneck is the batched matrix multiplications in each transformer layer, exploiting GPU tensor cores. Fine-tuning on task-specific clinical benchmarks (NER, RE) requires additional GPU compute for gradient accumulation across long sequences.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

BERT masked language modelling, next-sentence prediction, Flash Attention, Rotary Positional Embeddings (RoPE), BIO-tagging for NER, CRF output layers, relation extraction with span pairs, multi-label ICD classification, instruction-tuning with clinical instruction sets.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/clinical-nlp-over-notes-records.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/clinical-nlp-over-notes-records.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\clinical-nlp-over-notes-records.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the result, shows the
GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/` — a tiny, offline input so the demo runs
  with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (documented, idempotent).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: MIMIC-IV Clinical Notes — 331,794 de-identified patient notes from Beth Israel Deaconess (https://physionet.org/content/mimic-iv-note/) i2b2/n2c2 NLP Challenge Datasets — named entity, coreference, and relation tasks in clinical text (https://n2c2.dbmi.hms.harvard.edu/) MTSamples — 4,999 transcribed medical reports across 40 specialties (https://mtsamples.com/) MedQA / MedMCQA — medical question answering benchmarks for evaluating clinical LLMs (verify URL)

## Expected output

Success looks like `demo/expected_output.txt`. The program computes the result on
both the **GPU** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`)
and asserts they agree within the documented tolerance — that agreement is the
correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
2. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
3. [`src/kernels.cu`](src/kernels.cu) — the kernel(s) and host wrapper.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline.
5. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

BioClinicalBERT (https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) — BERT pretrained on MIMIC-III notes Clinical ModernBERT (https://github.com/Simonlee711/Clinical_ModernBERT) — ModernBERT fine-tuned on 13B tokens of PubMed + MIMIC-IV with 8192-token context medSpaCy (https://github.com/medspacy/medspacy) — spaCy-based clinical NLP pipeline with GPU inference support GatorTron (https://huggingface.co/UFNLP/gatortron-base) — large clinical LLM pretrained on 82B tokens of clinical text (verify URL)

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Flash Attention 2, cuBLAS for GEMM-dominated transformer layers, NCCL for data-parallel pretraining; pattern: data parallelism across multiple A100/H100 GPUs, gradient checkpointing to fit long-context batches in VRAM. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
