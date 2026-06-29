# 2.19 — Membrane Protein Simulation

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟢 Beginner · Established** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.19`
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

Membrane proteins (GPCRs, ion channels, transporters, integrins) are embedded in lipid bilayers and represent >50% of current drug targets. Explicit membrane MD requires building asymmetric bilayers with physiological lipid compositions and running microsecond simulations to sample conformational changes. CHARMM-GUI automates system building; GPU GROMACS/NAMD runs production simulations. Key challenges include equilibrating the membrane (~100 ns), maintaining bilayer asymmetry, and capturing slow conformational transitions. GPU-accelerated CG-MARTINI pre-equilibration (1–10 μs) followed by backmapping to all-atom provides a common pipeline.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

CHARMM36 lipid force field, POPE/POPC/cholesterol bilayer assembly, semi-isotropic barostat (NPT-xy coupling), PME for charged bilayer system, CG-to-AA backmapping, k-means clustering of ion channel gate states.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/membrane-protein-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/membrane-protein-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\membrane-protein-simulation.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: MemProtMD — 3133 membrane proteins in lipid bilayers (https://memprotmd.bioch.ox.ac.uk); GPCRdb — GPCR structures and MD data (https://gpcrdb.org); CGMD Platform benchmark systems (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7765266/); OPM — orientations of proteins in membranes (https://opm.phar.umich.edu).

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

CHARMM-GUI Membrane Builder (https://charmm-gui.org) — automated bilayer + protein setup; GROMACS (https://github.com/gromacs/gromacs) — GPU membrane protein MD; HTMD (https://github.com/Acellera/htmd) — GPU-accelerated membrane protein pipeline; packmol-memgen (https://github.com/memembranes) — AMBER membrane system builder.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

GPU semi-isotropic barostat coupling; cuFFT for PME with charged bilayer; custom CUDA PME corrections for 2D slab geometry; multi-GPU domain decomposition along z-axis; GPU neighbor list for heterogeneous lipid-protein system. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
