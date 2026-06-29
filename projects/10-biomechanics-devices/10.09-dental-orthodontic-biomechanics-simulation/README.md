# 10.9 — Dental & Orthodontic Biomechanics Simulation

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biomechanics%2C%20Biomedical%20Devices%20%26%20Surgery-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 10: Biomechanics, Biomedical Devices & Surgery · Catalog ID `10.9`
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

Orthodontic tooth movement depends on PDL (periodontal ligament) stress distribution, alveolar bone remodeling, and contact forces between brackets, wires, and clear aligners — all requiring nonlinear FEA on individually segmented CBCT geometries. GPU acceleration allows the dense contact constraint systems (dozens of tooth-aligner contact pairs per timestep) to be assembled and solved in parallel, enabling treatment planning that runs in minutes rather than hours. Dental implant osseointegration modeling couples elastic bone FEM with a poroelastic fluid-in-pore submodel at the implant interface. Population-scale virtual clinical trials — thousands of patient-specific models run simultaneously — become feasible on GPU clusters.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Hyperelastic PDL material models (Mooney-Rivlin, Ogden), bone-remodeling (Frost mechanostat), penalty-based contact, mortar contact formulation, thermo-mechanical coupling for composite restorations, coupled poroelastic FEM.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/dental-orthodontic-biomechanics-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/dental-orthodontic-biomechanics-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\dental-orthodontic-biomechanics-simulation.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: CBCT Tooth Segmentation Challenge (ToothFairy, MICCAI 2023) — annotated dental CBCT (https://toothfairy.grand-challenge.org/); 3D Dental Mesh Dataset (Teeth3DS) — 1800 intraoral scans (https://github.com/abenhamadou/3DTeethSeg22_challenge); NIH NIDCR FaceBase craniofacial CT atlas (https://www.facebase.org/); Open Dental Science datasets — clinical records + x-rays (verify URL via opendentalsoftware.com).

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

FEBio (https://github.com/febiosoftware/FEBio) — handles PDL and bone-remodeling constitutive models; CGAL (https://github.com/CGAL/cgal) — mesh generation from CBCT segmentations; ITK-SNAP (https://www.itksnap.org/) — CBCT segmentation to mesh pipeline; 3DTeethSeg (https://github.com/abenhamadou/3DTeethSeg22_challenge) — tooth segmentation model for mesh generation.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuSPARSE/cuSolver for contact-augmented stiffness matrix, CUDA kernels for per-element PDL stress update, Thrust for mortar contact pair enumeration; pattern: element-parallel stiffness assembly → penalty contact augmentation → PCG solve on GPU → bone-density update → geometry export for aligner CAD. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
