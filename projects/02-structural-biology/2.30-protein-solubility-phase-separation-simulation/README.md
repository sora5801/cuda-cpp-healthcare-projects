# 2.30 — Protein Solubility & Phase Separation Simulation

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.30`
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

Liquid-liquid phase separation (LLPS) of intrinsically disordered proteins (IDPs) and RNA-binding proteins underlies formation of biomolecular condensates (stress granules, P-bodies, nucleolus). Simulating LLPS requires system sizes of millions of CG atoms over millisecond timescales — only accessible with GPU CG-MD. FUS, TDP-43, and hnRNPA1 condensate-forming domains have been simulated with MARTINI or HPS (hydrophobicity scale) CG models on GPU. Phase diagrams are computed by running multiple concentration conditions simultaneously. Applications include predicting condensate-forming mutations and designing condensate-disrupting drugs.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Coarse-grained HPS/Kim-Hummer IDP model, MARTINI IDR parameters, Gibbs ensemble MC for phase coexistence, density functional theory for phase diagram, metadynamics order parameter for condensate formation, finite-size scaling for phase boundary.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/protein-solubility-phase-separation-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/protein-solubility-phase-separation-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\protein-solubility-phase-separation-simulation.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: FuzDB — fuzzy protein complex database (https://fuzdb.org); PhaSePro — proteins undergoing LLPS (https://phasepro.elte.hu); DisProt — intrinsically disordered proteins (https://disprot.org); human proteome LLPS predictor datasets (catGRANULE, PScore).

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

LAMMPS + HPS model (https://github.com/lammps/lammps) — GPU IDP LLPS simulation; OpenMM HPS (https://github.com/openmm/openmm) — Python IDP CG MD; CALVADOS 2 (https://github.com/KULL-Centre/CALVADOS) — residue-level IDP model for LLPS; GROMACS MARTINI IDR (https://github.com/gromacs/gromacs) — GPU CG LLPS.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

GPU CG-MD for multi-million-bead IDP system; CUDA kernel for simplified HPS non-bonded interactions; GPU-parallel concentration ensemble (multiple boxes); GPU-accelerated order parameter clustering for phase detection. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
