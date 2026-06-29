# 12.7 — DIA Proteomics Spectral Deconvolution

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Analytical%20%26%20Omics%20Data%20Processing-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 12: Analytical & Omics Data Processing · Catalog ID `12.7`
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

Data-Independent Acquisition (DIA) proteomics (Spectronaut, DIA-NN, FragPipe-DIA) co-isolates and co-fragments all precursors in wide isolation windows, requiring deconvolution of chimeric MS2 spectra containing overlapping fragment ion series. The GPU bottleneck is the inner-loop scoring: for each DIA window, thousands of peptide fragment ion templates must be correlated with the observed chromatographic fragment traces (XIC), a batched sliding-window cross-correlation problem. DIA-BERT (2025) is a GPU-enabled transformer approach treating DIA spectrum sequences analogously to language tokens, enabling improved feature extraction with GPU inference.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Extracted ion chromatogram (XIC) correlation scoring; deconvolution of chimeric spectra via library matching; Gaussian smoothing of chromatographic peaks; semi-empirical spectral library generation; transformer-based DIA spectrum encoding (DIA-BERT); target-decoy FDR estimation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/dia-proteomics-spectral-deconvolution.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/dia-proteomics-spectral-deconvolution.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\dia-proteomics-spectral-deconvolution.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: PRIDE ProteomeXchange DIA datasets (https://www.ebi.ac.uk/pride/); CPTAC DIA cancer proteomics (https://proteomics.cancer.gov/); Proteome profiler benchmark DIA datasets (verify URL); DIA-NN benchmark datasets (https://github.com/vdemichev/DiaNN).

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

DIA-NN (https://github.com/vdemichev/DiaNN) — fast DIA software (GPU inner-loop target); FragPipe (https://github.com/Nesvilab/FragPipe) — MSFragger-based DIA pipeline; DIA-BERT (https://proteomicsnews.blogspot.com/2025/05/dia-bert-gpu-enabled-dia-analysis.html) — GPU transformer for DIA; Spectronaut (commercial, Biognosys) — industry DIA software.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuFFT cross-correlation for XIC fragment trace matching; cuDNN transformer for DIA-BERT; batched sliding-window scoring kernels; GPU tensor for precursor×fragment scoring matrix; thrust for peak apex detection; multi-GPU for large clinical DIA cohorts. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
