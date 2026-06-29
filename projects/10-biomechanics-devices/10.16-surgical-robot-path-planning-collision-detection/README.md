# 10.16 — Surgical Robot Path Planning & Collision Detection

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biomechanics%2C%20Biomedical%20Devices%20%26%20Surgery-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 10: Biomechanics, Biomedical Devices & Surgery · Catalog ID `10.16`
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

Robotic-assisted surgery (e.g., da Vinci, Mako) requires real-time collision-free trajectories for multiple articulated arms moving near deformable anatomy. GPU parallel motion planning (RRT*, PRM) checks thousands of configuration-space samples for collision against a GPU-resident signed-distance-field (SDF) of the patient anatomy simultaneously, achieving path generation in under 100 ms — 50–100× faster than CPU planners. Deep-learning collision detectors trained in simulation (Learning-from-Simulation, 2025) replace explicit geometric checks with GPU neural networks, handling soft-tissue deformation that classical rigid-body checkers cannot. The GPU also runs online model-predictive controllers that re-plan at 50 Hz as tissue moves during respiration.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

GPU-parallel RRT*/PRM with SDF collision query, signed-distance-field generation via GPU ray marching, neural collision detector (implicit neural representation), MPC for force-controlled insertion, generalized momentum observer for external-force estimation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/surgical-robot-path-planning-collision-detection.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/surgical-robot-path-planning-collision-detection.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\surgical-robot-path-planning-collision-detection.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: SurgRobotics Dataset — da Vinci tool tracking + anatomy meshes (verify URL via MICCAI); SCARED Dataset — stereo depth reconstruction in laparoscopy (https://endovissub2019-scared.grand-challenge.org/); MICCAI 2024 Surgical Scene Segmentation Challenge (verify URL via Grand Challenge); CholecT50 (https://github.com/CAMMA-public/cholect50) — tool-tissue interaction labels.

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

GPU-based Parallel Collision Detection (UNC Gamma group, http://gamma.cs.unc.edu/gplanner/) — GPU PRM reference; cuRobo (https://github.com/NVlabs/curobo) — NVIDIA CUDA-accelerated robot motion generation; SOFA Framework (https://github.com/sofa-framework/sofa) — deformable anatomy + robot coupling; IsaacGym (https://developer.nvidia.com/isaac-gym) — GPU parallel surgical-robot RL training.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA kernels for SDF generation (parallel ray marching), cuDNN for neural collision network inference, Thrust for parallel RRT sample feasibility checks; pattern: GPU SDF updated from tissue deformation → 4096 configuration samples checked in parallel → feasible path selected → MPC re-plan at 50 Hz → torque commands dispatched. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
