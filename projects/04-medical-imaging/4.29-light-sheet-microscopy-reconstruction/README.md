# 4.29 — Light-Sheet Microscopy Reconstruction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.29`
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

Light-sheet fluorescence microscopy (LSFM / selective plane illumination, SPIM) acquires terabyte-scale datasets of developing embryos or cleared organs by illuminating a thin optical plane; the resulting multi-view 3D stacks must be: (1) registered across views/illuminations, (2) fused via multi-view deconvolution, and (3) stitched from tiled acquisitions. Multi-view deconvolution (Richardson-Lucy per view, Gaussian PSF model) on a 10³ × 10³ × 10³ sub-volume requires ~10¹² multiply-accumulates per outer iteration — GPU essential. BigStitcher (Fiji/ImageJ) uses GPU-accelerated image correlation for tile alignment and multi-GPU deconvolution for simultaneous multi-view fusion.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Multi-view Richardson-Lucy deconvolution (GPU), entropy-based content-weighted fusion, phase correlation tile stitching, BigStitcher alignment, iterative PSF estimation (blind deconvolution), SPIM dual-illumination fusion, 4D cell tracking (convolutional tracker).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/light-sheet-microscopy-reconstruction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/light-sheet-microscopy-reconstruction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\light-sheet-microscopy-reconstruction.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: OpenOrganelle (https://openorganelle.janelia.org/) — FIB-SEM and light-sheet neuroscience; EMBL LSFM public datasets (https://www.embl.org/); Zebrafish SPIM atlas data from Nature Methods papers; BioImage Archive LSFM collections (https://www.ebi.ac.uk/biostudies/bioimages).

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

BigStitcher (https://github.com/PreibischLab/BigStitcher) — GPU-accelerated LSFM stitching/fusion; CSBDeep/CARE (https://github.com/CSBDeep/CSBDeep) — deep learning LSFM denoising/restoration; N2V (https://github.com/juglab/n2v) — self-supervised GPU denoising for LSFM; DeconvolutionLab2 (https://github.com/Biomedical-Imaging-Group/DeconvolutionLab2) — multi-algorithm deconvolution with GPU.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuFFT for Fourier-domain deconvolution (Richardson-Lucy in k-space); cuBLAS for view-weight matrix products; custom CUDA for phase-correlation peak detection; multi-GPU domain decomposition across z-planes for large volumes; pinned host memory for streaming TB-scale data. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
