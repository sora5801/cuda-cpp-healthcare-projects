# 2.34 — Biophysical Simulation of Biomolecular Condensates (Active Learning Loop)

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.34`
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

Understanding the sequence determinants of biomolecular condensate properties (surface tension, viscosity, partition coefficients of client molecules) requires an active learning loop: GPU CG-MD generates condensate properties, a surrogate model (GNN on sequence) learns the property landscape, and Bayesian optimization proposes new sequences. This closes the loop between sequence, structure, and function for disordered proteins. GPU acceleration enables the necessary throughput (hundreds of condensate simulations per iteration). Applications include designing condensate-targeting therapeutics and understanding IDR evolution.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Bayesian active learning on sequence space, GNN surrogate for condensate properties, GPU CG-MD with IDP force fields, coexistence concentration estimation, diffusion coefficient estimation from MSD, transfer matrix for condensate-client partition.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/biophysical-simulation-of-biomolecular-condensates-active-learning-loop.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/biophysical-simulation-of-biomolecular-condensates-active-learning-loop.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\biophysical-simulation-of-biomolecular-condensates-active-learning-loop.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: PhaSePro (https://phasepro.elte.hu); DisProt (https://disprot.org); experimental LLPS partition coefficient datasets (verify URL); published condensate MD trajectory datasets (FUS, TDP-43, hnRNPA1).

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

CALVADOS 2 (https://github.com/KULL-Centre/CALVADOS) — GPU-compatible residue-level IDP model; OpenMM + GNN surrogate (https://github.com/openmm/openmm) — active learning condensate loop; LAMMPS GPU (https://github.com/lammps/lammps) — large-scale CG condensate simulation; BoTorch (https://github.com/pytorch/botorch) — GPU Bayesian optimization for sequence design.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

GPU CG-MD for condensate equilibration; PyTorch GNN surrogate on sequence features; BoTorch GPU Bayesian optimization; multi-GPU ensemble of condensate simulation replicas; GPU MSD calculation for diffusion coefficient. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
