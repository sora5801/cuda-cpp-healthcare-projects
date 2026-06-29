# 10.14 — LVAD / Rotary Blood Pump CFD & Hemolysis Prediction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biomechanics%2C%20Biomedical%20Devices%20%26%20Surgery-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 10: Biomechanics, Biomedical Devices & Surgery · Catalog ID `10.14`
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

Left ventricular assist devices (LVADs) expose blood to high shear stress at impeller blades, triggering hemolysis and thrombus formation. Patient-specific CFD with a rotating reference frame and moving mesh requires GPU-accelerated Navier-Stokes solutions on unstructured grids with ~5 M cells. The hemolysis index (power-law Giersiepen-Wurzinger model) is integrated along particle pathlines, computed by GPU-resident Lagrangian particle tracking. The 2024 simulation study demonstrated that hyperadhesion of activated platelets plays a dominant role in LVAD thrombosis at high rotor speeds. Design variants (impeller blade count, tip clearance) are evaluated in batches on GPU to build surrogate response surfaces for optimization.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Rotating reference frame Navier-Stokes (MRF/sliding mesh), Lagrangian particle tracking for hemolysis integration, platelet activation and thrombosis model (7-agonist biochemical cascade), power-law hemolysis (GKM), Euler-Euler two-phase (plasma + RBC) formulation, immersed boundary for rotor blades.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/lvad-rotary-blood-pump-cfd-hemolysis-prediction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/lvad-rotary-blood-pump-cfd-hemolysis-prediction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\lvad-rotary-blood-pump-cfd-hemolysis-prediction.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: FDA Benchmark Pump Dataset — PIV-measured flow in centrifugal/axial blood pumps (https://www.fda.gov/science-research/about-science-research-fda/computational-modeling-biomedical-devices); Multi-GPU IB Hemodynamics Benchmark (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7402620/); LVAD Thrombosis Simulation Archive (see https://arxiv.org/abs/2312.04761); HeartMate 3 geometry (anonymized, verify via Frontiers Cardiovasc Med).

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

OpenFOAM (https://github.com/OpenFOAM) — rotating machinery solvers (MRFSimpleFoam) with GPU linear-algebra backends; HemeLB (https://github.com/UCL/hemelb) — GPU LBM for cardiovascular flows; IBM at Extreme Scale (https://arxiv.org/html/2605.04335) — OpenACC+CUDA+NCCL IBM solver; CUDA particle tracking kernel templates (https://github.com/NVIDIA/CUDALibrarySamples).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA rotating-frame velocity interpolation kernels, cuSPARSE for pressure-velocity coupling, Thrust for particle trajectory integration; pattern: GPU unstructured CFD mesh → MRF velocity correction per cell → Lagrangian particle release → CUDA pathline integration → per-particle hemolysis accumulation → thrombosis probability map. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
