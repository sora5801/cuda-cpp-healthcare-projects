# 12.1 — Mass-Spectrometry Proteomics Search

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Analytical%20%26%20Omics%20Data%20Processing-lightgrey)

> **🟢 Beginner · Established** — Domain 12: Analytical & Omics Data Processing · Catalog ID `12.1`
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

Database peptide search correlates each observed MS/MS spectrum against thousands of theoretical peptide spectra from a protein sequence database, the most time-consuming step in proteomics. For a dataset of 100 k spectra against a human tryptic database of 1 M peptides (× 100 modifications), the search space is 10¹¹ comparisons; GPU parallelises scoring of thousands of theoretical spectra simultaneously per observed spectrum. GiCOPS (GPU-accelerated HiCOPS) achieves 1.2–5× speedup over CPU HiCOPS and >10× over older GPU tools like Tempest, using fragment-ion indexing on GPU. MSFragger uses hash-based fragment indexing on CPU but its inner scoring loop is a GPU acceleration target.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Fragment-ion indexing (hash/sorted lists of b/y-ions); Xcorr / HyperScore spectral dot product; fragment index mass offset search (open search); XCorr normalised cross-correlation; peptide-spectrum match (PSM) q-value estimation (Percolator); precursor mass matching and charge state deconvolution.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/mass-spectrometry-proteomics-search.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/mass-spectrometry-proteomics-search.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\mass-spectrometry-proteomics-search.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: PRIDE / ProteomeXchange — proteomics data repository (https://www.ebi.ac.uk/pride/); PeptideAtlas — validated human peptide spectral library (https://www.peptideatlas.org/); CPTAC cancer proteomics datasets (https://proteomics.cancer.gov/); MassIVE — mass spectrometry data repository (https://massive.ucsd.edu/).

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

GiCOPS (https://github.com/pcdslab/gicops) — GPU HPC framework for database peptide search; MSFragger (https://github.com/Nesvilab/MSFragger) — ultra-fast hash-index search (CPU, GPU inner loop target); Tempest — CUDA spectral scoring (verify URL; legacy); OpenMS (https://github.com/OpenMS/OpenMS) — proteomics framework with GPU integration potential.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

GPU hash tables for fragment ion indexing; batched dot-product CUDA kernels (one thread per theoretical peptide per observed spectrum); shared-memory spectral vector loading; cuFFT-based cross-correlation; multi-GPU database sharding. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
