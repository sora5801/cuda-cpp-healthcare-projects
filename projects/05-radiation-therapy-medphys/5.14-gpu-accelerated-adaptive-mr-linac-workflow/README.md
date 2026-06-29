# 5.14 — GPU-Accelerated Adaptive MR-Linac Workflow

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.14`
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

MR-Linac (MRL) systems (Elekta Unity, ViewRay MRIdian) combine MRI with simultaneous radiation delivery, enabling online adaptive radiotherapy (oART) where each fraction's plan is re-optimized based on daily anatomy. The oART workflow must complete all steps within a 30–90 minute treatment slot: (1) real-time MRI reconstruction (GPU NUFFT, <1 s), (2) deformable MR-to-MR registration (GPU Demons/VoxelMorph, <30 s), (3) synthetic CT generation (deep learning CT from MRI, GPU CNN, <10 s), (4) GPU dose recalculation on adapted anatomy (<30 s via collapsed-cone or MC), and (5) re-optimization (<2 min). Every step requires GPU; the entire chain is a GPU pipeline.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Real-time MRI reconstruction (radial GRASP GPU), MR-to-MR deformable registration (Demons, SyN), synthetic CT generation (CNN: MR→sCT), GPU collapsed-cone dose on sCT, GPU proton or photon dose recalculation, warm-start IMRT fluence re-optimization, plan approval metric computation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/gpu-accelerated-adaptive-mr-linac-workflow.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/gpu-accelerated-adaptive-mr-linac-workflow.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\gpu-accelerated-adaptive-mr-linac-workflow.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: MR-Linac Consortium shared datasets (verify URL at mrlinac.org); TCIA MR-guided RT datasets; AAPM MR-Linac WG test cases; MRI-only radiotherapy datasets from published cohorts.

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

Gadgetron (https://github.com/gadgetron/gadgetron) — real-time GPU MRI reconstruction for MRL; Plastimatch (https://plastimatch.org/) — GPU DIR + sCT generation; matRad (https://github.com/e0404/matRad) — dose re-optimization kernel; MONAI (https://github.com/Project-MONAI/MONAI) — CNN for MR→sCT translation.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA streams pipeline: acquisition → cuFFT NUFFT → cuDNN sCT CNN → GPU dose kernel → cuSPARSE optimizer → display; each stage double-buffered to overlap computation with data transfer; multi-GPU across the 5-stage pipeline. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
