# 1.28 — Covalent Docking

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.28`
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

Covalent inhibitors form a permanent or semi-permanent bond with a nucleophilic residue (usually Cys, Ser, Lys, Tyr). Docking them requires two-stage sampling: (1) non-covalent pre-reaction pose generation (as in standard docking) and (2) covalent bond geometry enforcement with post-reaction scoring. GPU acceleration helps explore the expanded conformational space after covalent bond formation. Methods include CovDock (Schrodinger), AutoDock-GPU covalent option, and emerging DL methods (CovDocker, 2025). EGFR/BTK/KRAS(G12C) covalent drug programs drive industrial interest.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Two-stage covalent docking protocol, warhead reactive group enumeration, covalent bond geometry constraint, MM-GBSA rescoring of covalent complexes, covalent pharmacophore matching.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/covalent-docking.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/covalent-docking.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\covalent-docking.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: CovDocker benchmark (2025, verify URL); ChEMBL covalent inhibitor set (https://www.ebi.ac.uk/chembl/); PDB covalent complex structures (https://www.rcsb.org); BindingDB covalent entries (https://www.bindingdb.org).

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

AutoDock-GPU (https://github.com/ccsb-scripps/AutoDock-GPU) — supports covalent docking mode; GNINA (https://github.com/gnina/gnina) — CNN-scored docking with covalent options; Uni-Dock (https://github.com/dptech-corp/Uni-Dock) — GPU docking extendable to covalent; CovDocker (arxiv 2506.21085, verify GitHub URL) — DL covalent docking benchmark.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Same as standard docking GPU pattern; additional CUDA kernel for covalent bond constraint penalty; GPU-parallel conformational sampling of warhead + linker degrees of freedom. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
