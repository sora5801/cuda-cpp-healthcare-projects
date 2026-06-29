# 10.15 — Cochlear Implant Computational Modeling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biomechanics%2C%20Biomedical%20Devices%20%26%20Surgery-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 10: Biomechanics, Biomedical Devices & Surgery · Catalog ID `10.15`
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

Cochlear implant (CI) electrodes stimulate spiral ganglion neurons via current fields that spread through complex fluid-filled scala tympani geometries. GPU-accelerated FEM on micro-CT-derived cochlear geometries computes the full 3D voltage distribution across the spiral ganglion fiber population in under a second, enabling real-time comparison of electrode array designs. Multi-compartment auditory nerve fiber (ANF) cable models are integrated in parallel on GPU — one thread per fiber per timestep — to predict neural firing patterns from arbitrary stimulation waveforms. Population-model simulations over thousands of virtual patients with varying cochlear anatomy quantify inter-subject variability in electrode coupling.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Volume-conductor FEM (bidomain), multi-compartment Hodgkin-Huxley cable models for ANF, psychoacoustic loudness growth modeling, Green's function electrode-impedance computation, Monte Carlo sampling over cochlear geometry populations.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/cochlear-implant-computational-modeling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/cochlear-implant-computational-modeling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\cochlear-implant-computational-modeling.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Cochlear Micro-CT Atlas (25 ANF traced geometries, see https://www.frontiersin.org/articles/10.3389/fnins.2025.1639092); Electrical Stimulation Human Cochlea Dataset (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6915103/); SIMBIOsys Cochlear Models (https://www.upf.edu/web/simbiosys/cochlear-implants); PhysioNet auditory nerve response databases (verify via physionet.org).

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

FEBio (https://github.com/febiosoftware/FEBio) — bidomain volume conductor FEM; NEURON simulator GPU branch (https://github.com/neuronsimulator/nrn) — parallel ANF cable integration; SimNIBS (https://github.com/simnibs/simnibs) — FEM for electrostimulation (adaptable to cochlear geometry); Cochlear FEM pipeline (SIMBIOsys UPF, verify URL at UPF site) — CI-specific meshing and solving workflow.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuSPARSE/cuSolver for bidomain FEM voltage solve, CUDA kernels for per-fiber HH cable ODE integration (embarrassingly parallel over ANFs), cuRAND for stochastic threshold variability; pattern: GPU FEM voltage field → per-fiber interpolation of extracellular potential → parallel ODE integration of HH equations → spike-time extraction → population audiogram prediction. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
