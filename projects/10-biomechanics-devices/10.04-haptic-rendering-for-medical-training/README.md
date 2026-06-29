# 10.4 — Haptic Rendering for Medical Training

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biomechanics%2C%20Biomedical%20Devices%20%26%20Surgery-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 10: Biomechanics, Biomedical Devices & Surgery · Catalog ID `10.4`
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

Haptic devices require force updates at 1 kHz or faster; the GPU must solve deformation and contact in under 1 ms per cycle. Energy-based haptic rendering computes virtual coupling forces from the difference between haptic device position and simulated tissue surface, requiring rapid contact detection and signed-distance-field (SDF) queries. GPU-accelerated SDFs pre-computed on volumetric grids enable sub-millisecond closest-point queries. Arterial catheter simulators, endoscopy trainers, and bone-drilling trainers all demand layered material models (mucosa, submucosa, muscle) with distinct stiffness, requiring per-layer GPU FE subsolvers. The bottleneck is contact resolution at the tool-tissue interface, parallelized over candidate contact pairs.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Energy-based haptic rendering with virtual coupling, signed-distance-field (SDF) contact detection, XPBD constraint projection, layered viscoelastic material models (Kelvin-Voigt), penumbra-based friction, god-object method for haptic proxy.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/haptic-rendering-for-medical-training.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/haptic-rendering-for-medical-training.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\haptic-rendering-for-medical-training.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: SOFA haptic benchmark scenes (liver puncture, needle insertion) (https://www.sofa-framework.org/); CholecT50 — laparoscopic cholecystectomy video for ground-truth tissue interaction reference (https://github.com/CAMMA-public/cholect50); Hamlyn Centre Laparoscopic / Robotic Video Dataset (http://hamlyn.doc.ic.ac.uk/vision/); Human Tissue Mechanical Properties Database (Picinbono et al., verify via SpringerLink).

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

SOFA Framework (https://github.com/sofa-framework/sofa) — modular GPU haptic-enabled simulator with OpenHaptics integration; Haptics-Medical-Simulation (https://github.com/HarrisKomn/Haptics-Medical-Simulation) — SOFA-based lung/bronchus haptic trainer with Geomagic Touch; Open-Source Visuo-Haptic Simulator (https://github.com/ChiaraSapo/Open-Source-Visuo-Haptic-Simulator-for-Surgical-Training) — SOFA-based multi-task haptic trainer; CHAI3D (https://www.chai3d.org) — haptic rendering framework with GPU geometry kernel support.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA kernels for SDF ray marching and contact pair query, cuSPARSE for tissue stiffness subsolve, Thrust for collision broadphase; pattern: GPU-resident SDF updated each deformation step → parallel contact pair generation → energy-gradient force computation → CPU haptic device readout at 1 kHz via shared ring buffer. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
