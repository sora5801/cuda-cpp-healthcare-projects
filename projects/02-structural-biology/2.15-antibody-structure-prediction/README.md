# 2.15 — Antibody Structure Prediction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.15`
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

Antibody structure prediction is specialized because the CDR-H3 loop is hypervariable and controls antigen specificity. Tools like IgFold, ABodyBuilder3, and IMGT-optimized AlphaFold2 models predict full antibody Fv region structures including flexible CDR loops. GPU inference enables high-throughput prediction for antibody library screening — thousands of sequences per GPU-hour. ABodyBuilder3 uses language model embeddings (ESM-2) and optimized GPU vectorization from OpenFold. Applications include antibody humanization, affinity maturation design, and developability assessment.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Attention-based CDR loop prediction, language model (ESM-2/IgLM) embeddings for antibody sequences, IMGT-numbered structure prediction, CDR-H3 loop sampling via diffusion, disulfide bond geometry constraints.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/antibody-structure-prediction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/antibody-structure-prediction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\antibody-structure-prediction.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: SAbDab — Structural Antibody Database (https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/); OAS (Observed Antibody Space) — 2B antibody sequences (https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/); CASP-Ab benchmarks; Thera-SAbDab — therapeutic antibody database (https://opig.stats.ox.ac.uk/webapps/newsabdab/therasabdab/).

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

IgFold (https://github.com/Graylab/IgFold) — fast antibody structure prediction on GPU; ABodyBuilder3 (verify GitHub URL) — GPU-optimized AF2 antibody model; AbNatiV (verify URL) — antibody naturalness scoring; AbDiffuser (verify URL) — antibody sequence+structure diffusion.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN multi-head attention for ESM-2 backbone; custom CDR attention CUDA kernels; FP16 inference with Flash attention; GPU-batched prediction for antibody library screening; PyTorch distributed for multi-GPU fine-tuning. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
