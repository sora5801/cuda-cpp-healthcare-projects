# 10.5 — Gait & Motion-Capture Biomechanics

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Biomechanics%2C%20Biomedical%20Devices%20%26%20Surgery-lightgrey)

> **🟢 Beginner · Established** — Domain 10: Biomechanics, Biomedical Devices & Surgery · Catalog ID `10.5`
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

Musculoskeletal gait analysis solves inverse kinematics (IK) and inverse dynamics (ID) to compute joint torques, followed by static optimization or forward-dynamics muscle recruitment minimizing metabolic cost. With 80+ muscles per limb and 200+ time frames per trial, the problem scales linearly with subjects in a cohort, making GPU batch-parallelism over trials the key acceleration strategy. Forward-dynamics predictive simulation using direct collocation (Moco) parallelizes across the collocation mesh nodes. GPU acceleration of Jacobian evaluation in trajectory optimization can achieve 7.7× speedup. Real-time IMU-based gait analysis on edge GPUs allows clinic-floor biomechanics without motion-capture labs.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Inverse kinematics (damped least-squares), inverse dynamics (Newton-Euler recursive), static optimization (bounded quadratic programming), direct collocation optimal control (Hermite-Simpson), musculotendon Hill-type models, contact detection in foot–ground models, Kalman-filter IMU fusion.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/gait-motion-capture-biomechanics.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/gait-motion-capture-biomechanics.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\gait-motion-capture-biomechanics.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: GaitRec — 2,084 patient bilateral ground reaction force (GRF) walking trials + 211 healthy controls (https://www.nature.com/articles/s41597-020-0481-z); CMU Motion Capture Database — 2500+ mocap sequences across diverse activities (http://mocap.cs.cmu.edu/); PhysioNet Gait/Posture Database — multi-camera + 17-IMU multimodal gait (https://physionet.org/content/multi-gait-posture/1.0.0/); Gait120 — comprehensive EMG + kinematic dataset (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12177048/).

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

OpenSim (https://github.com/opensim-org/opensim-core) — gold-standard musculoskeletal simulation; OpenSim Moco (https://github.com/opensim-org/opensim-moco) — direct collocation optimal control with multicore parallelism; Awesome-Biomechanics (https://github.com/modenaxe/awesome-biomechanics) — curated dataset/software index; PyBiomech (https://github.com/felixlb/pybiomech) — Python IMU processing pipeline (GPU-extensible).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuBLAS for batch matrix inversion in IK Jacobians, Thrust for parallel over-trial static optimization QP, CUDA kernels for Hill-model force-velocity lookup tables; pattern: batch subject/trial parallelism → per-frame Jacobian assembly on GPU → CPU-side IPOPT/CasADi optimal-control solve with GPU Jacobian callbacks. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
