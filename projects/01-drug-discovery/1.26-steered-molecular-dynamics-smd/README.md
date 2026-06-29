# 1.26 — Steered Molecular Dynamics (SMD)

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.26`
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

SMD applies external forces or velocity constraints to pull a molecule along a predefined coordinate (e.g., unbinding a ligand from a pocket), enabling calculation of work profiles and estimation of free energies via Jarzynski's equality. GPU MD allows many independent SMD trajectories to be run simultaneously, improving statistical convergence of Jarzynski estimates. Applications include estimation of drug residence time, rupture force of protein-ligand bonds, and domain opening mechanisms. NAMD pioneered GPU SMD; OpenMM provides Python-scriptable SMD via external forces.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Constant-velocity SMD (harmonic spring), constant-force SMD, Jarzynski equality for ΔG, fluctuation theorems, non-equilibrium work analysis, umbrella integration (follow-up).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/steered-molecular-dynamics-smd.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/steered-molecular-dynamics-smd.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\steered-molecular-dynamics-smd.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: NAMD SMD tutorials (https://www.ks.uiuc.edu/Training/Tutorials/); BindingDB residence time data (https://www.bindingdb.org); PDB force-probe simulation benchmark cases; published SMD studies on ion channels and motor proteins.

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

NAMD (https://www.ks.uiuc.edu/Research/namd/) — production GPU SMD; GROMACS pull code (https://github.com/gromacs/gromacs) — GPU SMD via pull-coord; OpenMM CustomExternalForce (https://github.com/openmm/openmm) — Python SMD; alchemlyb (https://github.com/alchemistry/alchemlyb) — Jarzynski post-processing.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Full GPU MD; custom CUDA force kernel for harmonic spring SMD; CUDA streams for multiple independent pulling trajectories; GPU memory for storing work accumulated along path. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
