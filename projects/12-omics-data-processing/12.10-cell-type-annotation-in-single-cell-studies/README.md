# 12.10 — Cell-Type Annotation in Single-Cell Studies

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Analytical%20%26%20Omics%20Data%20Processing-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 12: Analytical & Omics Data Processing · Catalog ID `12.10`
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

Cell-type annotation assigns biological identity to each sequenced cell by comparing its gene expression profile against reference atlases or marker gene signatures. GPU acceleration applies to: (1) nearest-centroid or KNN classification in high-dimensional gene space (GPU KNN via Faiss or cuML), (2) label transfer via GPU matrix multiplication (Seurat/Harmony), and (3) foundation model inference (scGPT, Geneformer, CellMaster) that takes tokenised gene expression as input to a transformer, with GPU inference on batches of cells. scGPT (2024) fine-tuned on 33 M cells demonstrates that GPU-accelerated transformer inference at cell-type annotation is now at least as accurate as classical marker-based methods.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

KNN label transfer in PCA-reduced gene space; Seurat anchor-based integration (CCA); marker-gene enrichment scoring (GSEA); transformer token attention over expressed genes (scGPT, Geneformer); logistic regression classifiers; hierarchical label propagation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/cell-type-annotation-in-single-cell-studies.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/cell-type-annotation-in-single-cell-studies.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\cell-type-annotation-in-single-cell-studies.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Human Cell Atlas (https://www.humancellatlas.org/); CellxGene Census (https://cellxgene.cziscience.com/); Azimuth reference atlases — curated cell-type references (https://azimuth.hubmapconsortium.org/); PanglaoDB — marker gene database (https://panglaodb.se/).

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

scGPT (https://github.com/bowang-lab/scGPT) — single-cell foundation model, GPU transformer inference; Geneformer (https://huggingface.co/ctheodoris/Geneformer) — transformer pre-trained on 30 M cells; rapids-singlecell (https://github.com/scverse/rapids_singlecell) — GPU KNN label transfer; CellMaster (https://arxiv.org/pdf/2602.13346) — collaborative annotation with LLM reasoning.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuML KNN for label transfer; cuDNN transformer inference for scGPT / Geneformer; batched tokenised cell embedding via GPU; Faiss-GPU for reference atlas similarity search; multi-GPU gradient accumulation for foundation model fine-tuning. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
