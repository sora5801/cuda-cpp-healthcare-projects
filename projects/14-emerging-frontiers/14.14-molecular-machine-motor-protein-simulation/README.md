# 14.14 — Molecular Machine & Motor Protein Simulation

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Emerging%2C%20Theoretical%20%26%20Grand--Challenge%20Frontiers-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 14: Emerging, Theoretical & Grand-Challenge Frontiers · Catalog ID `14.14`
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

Molecular machines — kinesin walking on microtubules, ATP synthase rotating, ribosome translating — operate at nanoscale over microsecond-to-millisecond timescales that are far beyond conventional all-atom MD. GPU-accelerated enhanced sampling methods (metadynamics with PLUMED-CUDA, replica-exchange MD, HTMD adaptive sampling) extend the timescale window by orders of magnitude. Coarse-grained (MARTINI, CGMD) simulations on GPU model the full kinesin power stroke in minutes. The cryo-EM structural database provides high-resolution snapshots of machine conformations that seed GPU MD simulations of the mechanical cycle. Understanding motor protein dysfunction underpins treatments for neurodegeneration, cancer, and rare genetic diseases.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

All-atom MD (GROMACS GPU, OpenMM), coarse-grained MD (MARTINI CGMD), metadynamics / funnel metadynamics with PLUMED-CUDA, replica-exchange MD (REMD), accelerated MD (aMD), elastic network model (ENM) for collective modes, Brownian ratchet mechanochemical models.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/molecular-machine-motor-protein-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/molecular-machine-motor-protein-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\molecular-machine-motor-protein-simulation.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: RCSB PDB motor protein structures — kinesin, dynein, myosin, ATP synthase (https://www.rcsb.org/); CHARMM-GUI membrane builder inputs (https://www.charmm-gui.org/); EMDB cryo-EM maps of conformational states (https://www.ebi.ac.uk/emdb/); GPCRdb for GPCR molecular machine models (https://gpcrdb.org/).

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

GROMACS (https://github.com/gromacs/gromacs) — GPU MD with CUDA/HIP, fastest production MD engine; OpenMM (https://github.com/openmm/openmm) — Python GPU MD with custom force plugins; PLUMED (https://github.com/plumed/plumed2) — GPU-compatible enhanced sampling (metadynamics) CV library; HTMD (https://github.com/Acellera/htmd) — GPU adaptive sampling for protein conformational exploration.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA bonded/non-bonded force kernels (GROMACS native CUDA), cuFFT for PME long-range electrostatics, GPU neighbor-list Verlet scheme; pattern: cryo-EM structure → CHARMM-GUI parameterization → GPU REMD ensemble (N replicas × GPU) → PLUMED metadynamics bias application → free-energy surface reconstruction via WHAM. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
