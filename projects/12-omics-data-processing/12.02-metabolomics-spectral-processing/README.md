# 12.2 — Metabolomics Spectral Processing

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Analytical%20%26%20Omics%20Data%20Processing-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 12: Analytical & Omics Data Processing · Catalog ID `12.2`
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

Metabolomics LC-MS/MS produces thousands of spectra per sample that must be denoised, deconvoluted, and matched against spectral libraries (e.g., MassBank, HMDB). Key GPU-amenable steps: (1) denoising via 2D Gaussian filtering on the (m/z, retention-time) ion map, (2) spectral library matching via batched dot-product between observed and reference spectra (identical to proteomics search but with small molecule fragmentation patterns), and (3) isotope deconvolution using the Averagine model for charge-state assignment. GPU batch cross-correlation across tens of thousands of library entries per observed spectrum replaces sequential CPU loops.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Gaussian kernel smoothing on MS1 ion maps; isotope deconvolution via Averagine model; dot-product spectral library matching; modified cosine similarity for spectral networking (GNPS); mass-defect filtering; retention time alignment via dynamic time warping (DTW).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/metabolomics-spectral-processing.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/metabolomics-spectral-processing.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\metabolomics-spectral-processing.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: GNPS / MassIVE metabolomics datasets (https://gnps.ucsd.edu/); HMDB — Human Metabolome Database spectral library (https://hmdb.ca/); MetaboLights — metabolomics studies repository (https://www.ebi.ac.uk/metabolights/); MassBank of North America — MS/MS spectral library (https://mona.fiehnlab.ucdavis.edu/).

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

GNPS (https://gnps.ucsd.edu/) — spectral networking platform (GPU matching target); MZmine3 (https://github.com/mzmine/mzmine3) — open-source LC-MS processing (GPU acceleration integration target); SIRIUS (https://github.com/boecker-lab/sirius) — molecular formula / structure prediction; OpenMS (https://github.com/OpenMS/OpenMS) — LC-MS processing suite.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuFFT for cross-correlation in spectral library matching; custom 2D Gaussian smoothing CUDA kernels on ion maps; thrust for m/z sorted spectral vector operations; batched cosine similarity via cuBLAS GEMM (spectra as rows of a matrix); GPU-resident library matrix for parallel dot-product. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
