# 2.22 — Electron Density Map Analysis & Model Validation

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟢 Beginner · Established** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.22`
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

Crystallographic and cryo-EM electron density maps must be validated for model-to-map fit quality before deposition. GPU-accelerated real-space correlation coefficient (RSCC) and Fourier shell correlation (FSC) calculations over millions of voxels enable rapid quality assessment. Phenix, CCP4, and GEMMI provide GPU-accelerated map manipulation. Structure factor calculation (Fcalc vs. Fobs difference maps in crystallography) requires GPU FFT over large reciprocal-space datasets. For cryo-EM, local resolution estimation (MonoRes, ResMap) computes local FSC across the map in sliding windows — GPU-parallelized.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Real-space correlation coefficient (RSCC), Fourier shell correlation (FSC), difference map calculation (Fo-Fc, 2Fo-Fc), R-factor / R-free crystallographic validation, local resolution estimation, model-to-map fit scoring.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/electron-density-map-analysis-model-validation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/electron-density-map-analysis-model-validation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\electron-density-map-analysis-model-validation.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: EMDB validation maps (https://www.ebi.ac.uk/emdb/); PDB structure factors (https://www.rcsb.org); IUCr validation standards datasets (verify URL); wwPDB OneDep validation pipeline (https://deposit.wwpdb.org).

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

Phenix (https://phenix-online.org) — crystallography and cryo-EM refinement with GPU acceleration; CCP4 (https://www.ccp4.ac.uk) — crystallographic computing suite; GEMMI (https://github.com/project-gemmi/gemmi) — GPU-friendly CIF/map library; EMAN2 (https://blake.bcm.edu/emanwiki/EMAN2) — GPU cryo-EM processing suite.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuFFT for structure factor FFT; custom CUDA correlation coefficient computation over map voxels; GPU FSC computation via batched FFT ring averaging; GPU local resolution sliding window in parallel. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
