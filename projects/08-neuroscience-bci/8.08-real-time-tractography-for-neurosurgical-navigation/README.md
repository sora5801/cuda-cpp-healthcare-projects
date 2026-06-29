# 8.8 — Real-Time Tractography for Neurosurgical Navigation

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Neuroscience%20%26%20Brain--Computer%20Interfaces-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 8: Neuroscience & Brain-Computer Interfaces · Catalog ID `8.8`
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

Diffusion tensor imaging (DTI) tractography traces white matter fiber bundles from seed ROIs by integrating principal diffusion directions through the 3D DTI field (streamline tracking). Intraoperative real-time tractography updates the fiber map as brain shift occurs during surgery, requiring sub-second computation. GPU parallelizes thousands of independent streamline integrations (CUDA: one thread per seed). Probabilistic tractography (FSL BEDPOSTX) samples from diffusion parameter posteriors—thousands of Monte Carlo streamlines per seed—is also GPU-amenable.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Deterministic streamline tractography (FACT, Runge-Kutta 4th order), probabilistic tractography (FSL BEDPOSTX ball-and-stick model), fiber orientation distribution (FOD) from HARDI (spherical deconvolution), constrained spherical deconvolution (CSD), DSI/Q-ball imaging, anatomical tract atlas registration (MNI-space), curvature-limited streamline termination.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/real-time-tractography-for-neurosurgical-navigation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/real-time-tractography-for-neurosurgical-navigation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\real-time-tractography-for-neurosurgical-navigation.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Human Connectome Project DT-MRI (https://db.humanconnectome.org); ADNI diffusion MRI (https://adni.loni.usc.edu); ISMRM 2015 Tractography Challenge dataset (verify URL — tractometer.org or ismrm.org); OpenNeuro diffusion MRI datasets (https://openneuro.org).

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

DIPY (https://github.com/dipy/dipy) — Python DTI/HARDI tractography with GPU acceleration via CuPy; FSL GPU tractography (GPU BEDPOSTX) (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT) — CUDA-accelerated probabilistic tractography; MRtrix3 (https://github.com/MRtrix3/mrtrix3) — constrained spherical deconvolution + tractography; TrackVis/DiffusionTool (verify URL) — surgical navigation-oriented fiber display.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA kernel for parallel streamline integration (one thread per seed, RK4 over DTI field in texture memory); cuBLAS for tensor field operations; cuFFT for spherical harmonic convolution in CSD; pattern: texture-memory DTI field for fast interpolation, warp-level thread divergence handled by fixed-step integration. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
