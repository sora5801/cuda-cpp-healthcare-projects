# 1.6 — Enhanced Sampling — Metadynamics & Replica Exchange

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.6`
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

Standard MD cannot cross large free energy barriers on accessible timescales. Enhanced sampling methods accelerate conformational exploration by adding history-dependent bias potentials (metadynamics) or by running multiple copies at different temperatures/Hamiltonians (REMD/HREX). PLUMED plugs into GROMACS, NAMD, OpenMM, and LAMMPS to implement CVs and bias on the fly. GPU MD trajectories feed the bias engine with minimal overhead. Well-tempered metadynamics ensures convergence of the free energy surface (FES) and is widely used for drug binding pathway elucidation. GPU-MetaD (2025) achieves full-lifecycle GPU acceleration for ML potential metadynamics with systems up to 1.3M atoms.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Well-tempered metadynamics, funnel metadynamics, Hamiltonian replica exchange (HREX), temperature REMD (T-REMD), replica exchange with solute tempering (REST2), collective variable (CV) on-the-fly evaluation, free energy surface estimation via reweighting.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/enhanced-sampling-metadynamics-replica-exchange.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/enhanced-sampling-metadynamics-replica-exchange.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\enhanced-sampling-metadynamics-replica-exchange.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: PLUMED-NEST — repository of published metadynamics/enhanced sampling input files (https://www.plumed-nest.org); GPCRmd trajectory archive (https://gpcrmd.org); D. E. Shaw millisecond MD datasets (available via RCSB); benchmark FES for alanine dipeptide / chignolin (commonly used test systems).

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

PLUMED (https://github.com/plumed/plumed2) — plugin for collective variables and enhanced sampling, GPU-compatible via host MD engine; GROMACS + PLUMED (https://github.com/gromacs/gromacs) — standard GPU metadynamics stack; OpenPathSampling (https://github.com/openpathsampling/openpathsampling) — transition path sampling framework; HTMD (https://github.com/Acellera/htmd) — high-throughput MD with adaptive sampling on GPU clusters.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Bias potential evaluation on CPU via PLUMED with negligible overhead; full MD force/integration on GPU; multi-walker metadynamics uses MPI + NCCL across GPUs; GPU kernels for on-the-fly CV computation in GPU-MetaD. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
