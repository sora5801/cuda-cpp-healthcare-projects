# 2.23 — Protein-Ligand Interaction Energy Decomposition

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.23`
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

Per-residue energy decomposition (MM-GBSA per-residue, FEP energy components) identifies which protein residues contribute most to ligand binding, guiding lead optimization and resistance mutation analysis. GPU MD trajectories provide snapshots; GPU-parallel per-residue energy evaluation attributes contributions from each residue. This reveals hot-spot residues for mutational scanning, identifies water-mediated interactions, and explains selectivity across protein family members. Kinase resistance mutation mapping in oncology is a prime application.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

MM-GBSA per-residue energy decomposition, pairwise interaction energy, electrostatic + VDW component separation, water bridge detection, solvent contribution per residue, FEP component analysis.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/protein-ligand-interaction-energy-decomposition.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/protein-ligand-interaction-energy-decomposition.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\protein-ligand-interaction-energy-decomposition.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: PDB-bind (http://www.pdbbind.org.cn); resistance mutation datasets (ClinVar, https://www.ncbi.nlm.nih.gov/clinvar/); KLIFS kinase binding data (https://klifs.net); ChEMBL activity data for target families (https://www.ebi.ac.uk/chembl/).

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

AMBER MMPBSA.py decomp (https://ambermd.org/AmberTools.php) — per-residue energy decomposition; gmx_MMPBSA (https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA) — GROMACS MM-GBSA decomposition; MDAnalysis (https://github.com/MDAnalysis/mdanalysis) — pairwise residue-ligand contact analysis; ProLIF (https://github.com/chemosim-lab/ProLIF) — IFP for binding mode decomposition.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

GPU MD trajectory generation; CUDA parallel per-residue GB energy evaluation; GPU-batched snapshot processing (N frames × M residues); cuBLAS for energy matrix accumulation. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
