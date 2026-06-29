# 2.5 — Coarse-Grained / MARTINI Simulation

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟢 Beginner · Established** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.5`
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

Coarse-grained (CG) force fields like MARTINI map ~4 heavy atoms to a single interaction site, enabling microsecond-to-millisecond simulations of large membrane systems (entire plasma membranes with 63 lipid species, viral capsids, ribosomes). MARTINI3 CG-MD runs in GROMACS with full GPU acceleration, gaining ~100-fold timescale extension over all-atom MD. Membrane protein insertion, lipid scrambling, and vesicle formation are accessible only at CG resolution. The GPU bottleneck is non-bonded CG pair interactions; the coarser grid makes PME and neighbor lists faster than all-atom.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

MARTINI3 force field, Lennard-Jones + shifted electrostatics for CG beads, elastic network overlay (Gō-MARTINI) for protein secondary structure, CG-to-AA backmapping, PME for long-range CG electrostatics.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/coarse-grained-martini-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/coarse-grained-martini-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\coarse-grained-martini-simulation.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: CHARMM-GUI MARTINI membrane builder outputs (https://charmm-gui.org); lipid parameter database (https://cgmartini.nl); membrane-active peptide aggregation benchmarks; EMDB viral capsid reference maps for validation.

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

GROMACS+MARTINI3 (https://github.com/gromacs/gromacs) — production GPU CG-MD; MARTINI force field files (https://cgmartini.nl) — official parameter repository; TS2CG (https://github.com/weria-pezeshkian/TS2CG) — triangulated surface to CG membrane builder; insane.py (https://github.com/Tsjerk/Insane) — membrane assembly tool for MARTINI.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA kernels for CG non-bonded pair evaluation; cuFFT for CG PME; neighbor list construction with larger cutoffs (1.1–1.2 nm vs 0.9 nm AA); GPU memory efficiency improved by reduced atom count (~4× vs AA). --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
