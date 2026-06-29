# 12.8 — Isotope Pattern Matching & Charge Deconvolution

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Analytical%20%26%20Omics%20Data%20Processing-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 12: Analytical & Omics Data Processing · Catalog ID `12.8`
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

High-resolution mass spectrometry resolves isotope envelopes (the pattern of ¹²C, ¹³C, ²H, ¹⁸O peaks) that report the charge state and monoisotopic mass of each peptide or metabolite. Matching observed isotope patterns against theoretical Averagine distributions (or exact elemental isotope calculations via IsoSpec) across millions of features per LC-MS run is a quadratic search problem. GPU parallelism assigns one thread per candidate mass window, computing the dot product between observed and theoretical isotope patterns simultaneously across thousands of charge states and masses, replacing the sequential CPU sweep.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Averagine model for average elemental composition; Mercury / IsoSpec exact isotope pattern calculation via Poisson convolution; dot-product / cosine-similarity matching of isotope envelopes; Maximum Likelihood charge state assignment; THRASH deconvolution algorithm; Wavelet transform for isotope detection (IsotopeWavelet).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/isotope-pattern-matching-charge-deconvolution.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/isotope-pattern-matching-charge-deconvolution.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\isotope-pattern-matching-charge-deconvolution.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: PRIDE ProteomeXchange high-resolution datasets (https://www.ebi.ac.uk/pride/); HMDB high-resolution metabolomics spectra (https://hmdb.ca/); MassBank (https://massbank.eu/); CPTAC iTRAQ/TMT quantitative proteomics (https://proteomics.cancer.gov/).

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

OpenMS (https://github.com/OpenMS/OpenMS) — comprehensive LC-MS toolkit with GPU integration hooks; IsoSpec (https://github.com/MatteoLacki/IsoSpec) — exact isotope pattern computation; Xtract (Thermo Fisher, proprietary) — charge deconvolution; pyOpenMS (https://github.com/OpenMS/OpenMS) — Python bindings for proteomics.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Batched dot-product CUDA kernels (one warp per candidate m/z window); cuFFT for wavelet-based isotope detection; shared-memory Averagine lookup tables; thrust for peak list sorting and deduplication; cuBLAS GEMM for charge-state × m/z scoring matrix. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
