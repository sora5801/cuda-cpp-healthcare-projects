# 8.2 — Spike Sorting

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Neuroscience%20%26%20Brain--Computer%20Interfaces-lightgrey)

> **🟢 Beginner · Established** — Domain 8: Neuroscience & Brain-Computer Interfaces · Catalog ID `8.2`
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

Spike sorting identifies the firing times and cellular identities of individual neurons from raw extracellular voltage traces recorded on multi-electrode arrays (MEAs) or Neuropixels probes (384 channels × 30 kHz). The GPU bottleneck is template-matching: cross-correlating detected waveforms against hundreds of neuron templates across all channels simultaneously. Kilosort4 achieves this via GPU-accelerated template convolution, reducing hours of CPU sorting to minutes and enabling automated curation for large-scale Neuropixels datasets.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Whitening and common-average reference (CAR) preprocessing, threshold-based spike detection, PCA dimensionality reduction, template-matching (cross-correlation), expectation-maximization (EM) clustering, drift correction via continuous template registration, Gaussian mixture model (GMM) classification.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/spike-sorting.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/spike-sorting.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\spike-sorting.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: DANDI Archive Neuropixels datasets (https://dandiarchive.org); Allen Brain Observatory Neuropixels visual coding dataset (https://portal.brain-map.org); SpikeInterface benchmark datasets (https://spikeinterface.readthedocs.io); MountainSort benchmark datasets on Zenodo (search zenodo.org "spike sorting benchmark").

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

Kilosort4 (https://github.com/MouseLand/Kilosort) — GPU template-matching spike sorter, Python, CUDA; MountainSort5 (https://github.com/flatironinstitute/mountainsort5) — Flatiron Institute sorter with GPU preprocessing; SpikeInterface (https://github.com/SpikeInterface/spikeinterface) — unified Python framework wrapping 10+ sorters including GPU ones; Phy (https://github.com/cortex-lab/phy) — manual curation GUI for Kilosort output.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuFFT for template convolution (FFT-based cross-correlation); cuBLAS for waveform-template matrix multiply; cuSPARSE for sparse cluster assignment; CUDA Thrust for peak-finding in filtered traces; pattern: sliding-window batch convolution with cuFFT, one FFT per channel-template pair in a batched call. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
