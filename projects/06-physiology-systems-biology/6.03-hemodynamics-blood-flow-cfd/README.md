# 6.3 — Hemodynamics / Blood-Flow CFD

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.3`
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

Solves the incompressible Navier-Stokes equations on patient-specific vascular geometries (aorta, coronary arteries, cerebral vasculature) reconstructed from CT/MRI angiography. Non-Newtonian blood rheology (Carreau-Yasuda model) and fluid-structure interaction (FSI) with compliant vessel walls add computational stiffness. Wall shear stress (WSS) and oscillatory shear index (OSI) fields—risk factors for atherosclerosis—require temporally resolved solutions across the cardiac cycle (~1000 time steps). GPU parallelism maps naturally onto the unstructured mesh cell updates.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Incompressible Navier-Stokes (fractional-step / SIMPLE / PISO), ALE formulation for FSI, non-Newtonian viscosity (Carreau-Yasuda), arbitrary Lagrangian-Eulerian mesh motion, finite volume method on unstructured polyhedral meshes, multigrid pressure solver, RBF mesh morphing.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/hemodynamics-blood-flow-cfd.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/hemodynamics-blood-flow-cfd.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\hemodynamics-blood-flow-cfd.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: PhysioNet MIMIC-III waveforms — invasive pressure/flow recordings (https://physionet.org/content/mimiciii/1.4/); Vascular Model Repository — patient-specific vascular geometries (http://www.vascularmodel.com); Zenodo Cardiac Mechanics Emulation dataset (https://zenodo.org/records/7075055); UK Biobank aortic flow (4D flow MRI subset) (https://www.ukbiobank.ac.uk).

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

SimVascular/svFSI (https://github.com/SimVascular/svFSI) — open-source image-to-simulation pipeline with GPU-capable parallel solver; OpenFOAM-dev (https://github.com/OpenFOAM/OpenFOAM-dev) — general CFD with biomedical application via custom boundary conditions; Chaste (https://github.com/Chaste/Chaste) — includes vascular network flow module; HemeLB (https://github.com/hemelb-codes/hemelb) — sparse vascular lattice-Boltzmann alternative.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

AmgX (GPU multigrid pressure solver), cuSPARSE (SpMV for flux assembly), NVIDIA RAPIDS for mesh preprocessing; pattern: domain decomposition with MPI+CUDA, halo-exchange via NCCL, time-stepping loop with async memory transfers. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
