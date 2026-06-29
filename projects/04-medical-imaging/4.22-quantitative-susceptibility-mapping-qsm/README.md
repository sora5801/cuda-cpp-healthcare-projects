# 4.22 — Quantitative Susceptibility Mapping (QSM)

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.22`
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

QSM reconstructs tissue magnetic susceptibility (χ) from gradient-echo phase data in a 3D volume. The pipeline involves phase unwrapping (PUROR, ROMEO), background field removal (PDF, SHARP, VSHARP), and dipole inversion (MEDI, TKD, iLSQR, deep learning). The dipole inversion is the computational bottleneck: the forward model in k-space is a multiplication by a dipole kernel (analytically known), but inversion is ill-posed at the magic angle (cone of zero crossing). Iterative MEDI minimization requires O(100) iterations of 3D FFT + gradient updates on a 256³ volume, each costing ~30 ms GPU vs. seconds CPU. Deep learning QSM (QSMnet, xQSM) replaces MEDI with a single GPU network forward pass (<1 s).

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Phase unwrapping (PUROR, ROMEO, BEST path), SHARP/V-SHARP background removal, MEDI (morphology-enabled dipole inversion), TKD (threshold-based k-space division), iterative least-squares (iLSQR), deep learning dipole inversion (QSMnet, xQSM), total-variation regularized inversion.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/quantitative-susceptibility-mapping-qsm.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/quantitative-susceptibility-mapping-qsm.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\quantitative-susceptibility-mapping-qsm.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: QSM Reconstruction Challenge 2.0 (https://doi.org/10.1101/2020.11.25.397695 — data on Zenodo); HCP 7T multiecho GRE data (https://db.humanconnectome.org/); AHEAD dataset (Amsterdam Ultra-high field Adult lifespan Database); BioBank UKB (https://www.ukbiobank.ac.uk/).

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

QSMnet (https://github.com/SNU-LIST/QSMnet) — deep learning QSM on GPU; MEDI toolbox (http://pre.weill.cornell.edu/mri/pages/qsm.html — verify URL) — MATLAB MEDI + GPU options; ROMEO (https://github.com/korbinian90/ROMEO) — fast phase unwrapping; STISuite (verify URL) — STI + QSM MATLAB toolbox.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuFFT for dipole kernel multiplication in k-space per MEDI iteration; custom CUDA gradient/divergence operators for TV regularization; cuBLAS for conjugate gradient solver; memory layout: complex float32 arrays, FFT in-place. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
