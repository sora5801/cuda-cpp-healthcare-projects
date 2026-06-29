# 1.22 — Constant-pH Molecular Dynamics

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.22`
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

Biomolecular simulations normally fix protonation states, ignoring pH-dependent conformational changes critical for drug design (e.g., histidine flips, aspartate protonation near binding sites). Continuous constant-pH MD (CpHMD) in AMBER22 pmemd.cuda couples proton titration MC moves to GPU MD, sampling both conformation and protonation simultaneously. A 400-residue protein at single-pH takes ~1 hour on an RTX 2080 — >1000× faster than CPU. Applications include pKa prediction, pH-dependent drug binding, and ion channel gating.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Continuous CpH titration (GB or PME-explicit solvent), Metropolis MC protonation moves, replica exchange CpHMD (REX-CpHMD), free energy estimation of pKa shifts, AMBER ff14SB/ff19SB titration parameters.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/constant-ph-molecular-dynamics.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/constant-ph-molecular-dynamics.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\constant-ph-molecular-dynamics.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: pKa databases: PKAD (https://compbio.clemson.edu/pkad/), PHMD reference pKa values; Benchmark pKa sets for Asp/Glu/His/Cys/Lys residues; DrugBank compounds with ionizable groups (https://go.drugbank.com).

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

AMBER pmemd.cuda CpHMD (https://ambermd.org/GPUSupport.php) — GPU constant-pH MD; CHARMM CpHMD (https://www.charmm.org) — GBSW implicit solvent titration; OpenMM constant-pH (https://github.com/openmm/openmm) — Python CpH framework; PropKa (https://github.com/jensengroup/propka) — fast pKa prediction for system setup.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Full MD on GPU (pmemd.cuda); MC protonation moves evaluated via energy difference on GPU; replica exchange across pH replicas using NCCL/MPI; trajectory analysis on GPU via cuML. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
