# 1.23 — QM/MM Molecular Dynamics

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.23`
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

Hybrid quantum mechanics/molecular mechanics (QM/MM) partitions a system into a reactive QM region (drug + key residues, 50–200 atoms) treated at DFT/semi-empirical level and a larger MM region. GPU acceleration applies to both the QM Hamiltonian (via TeraChem/GPU-DFT) and the MM dynamics (via AMBER/GROMACS). The critical bottleneck is the QM/MM electrostatic coupling and QM Hamiltonian evaluation at every MD step. Open-source GPU QM/MM is available via AMBER+QUICK (GPU-accelerated DFT engine). Applications include enzyme catalysis mechanism, covalent drug reactivity, and proton transfer pathways.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

ONIOM/link-atom QM/MM coupling, electrostatic embedding, DFT-based QM region (B3LYP/PBE), GFN2-xTB semi-empirical QM, AIMD in QM region with Verlet MM, adaptive QM/MM for large reactive systems.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/qm-mm-molecular-dynamics.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/qm-mm-molecular-dynamics.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\qm-mm-molecular-dynamics.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: QM/MM benchmark from SAMPL challenges (verify URL); enzyme reaction databases (BRENDA, https://www.brenda-enzymes.org); crystal structures of enzyme-drug complexes from PDB (https://www.rcsb.org); RCSB ligand validation data (https://www.rcsb.org).

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

AMBER+QUICK (https://github.com/merzlab/QUICK) — GPU-accelerated DFT for QM/MM with AMBER; TeraChem-TCPB (https://www.petachem.com) — GPU DFT server for QM/MM with NAMD/AMBER; OpenMM+PySCF QM/MM (https://github.com/openmm/openmm) — Python QM/MM interface; cp2k (https://github.com/cp2k/cp2k) — GPU-accelerated QM/MM for periodic systems.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

GPU ERI computation for QM Hamiltonian via TeraChem/QUICK CUDA kernels; MM region on GPU (pmemd.cuda); asynchronous GPU-CPU communication for QM/MM coupling; CUDA streams for overlapping QM and MM compute. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
