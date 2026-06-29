# 2.2 — Protein-Protein Docking

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟢 Beginner · Established** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.2`
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

Predicting protein-protein complex structures is critical for understanding signaling pathways, antibody-antigen recognition, and designing PPI inhibitors. Classical docking (ClusPro, ZDOCK) uses FFT-based rigid-body search over rotational/translational degrees of freedom — a 6D correlation function evaluable via GPU FFT on spherical harmonic expansions. DL methods (DiffDock-PP, HelixDock, RoseTTAFold2NA) use equivariant diffusion or MSA-based co-evolution to predict complex structures. GPU enables both the FFT rigid-body search and deep learning inference.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

FFT-based rigid-body docking (spherical harmonics), residue-level contact prediction (coevolution), equivariant diffusion docking, half-sphere exposure (HSE) surface scoring, electrostatic/shape complementarity scoring.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/protein-protein-docking.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/protein-protein-docking.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\protein-protein-docking.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Docking Benchmark 5.5 — 230 non-redundant protein complexes (https://zlab.umassmed.edu/benchmark/); SAbDab — structural antibody database (https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/); PPI4DOCK benchmark (verify URL); PDB protein complexes (https://www.rcsb.org).

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

ClusPro (https://cluspro.bu.edu) — FFT docking server (GPU-accelerated back end); DiffDock-PP (https://github.com/ketatam/DiffDock-PP) — rigid protein-protein diffusion docking; HADDOCK (https://wenmr.science.uu.nl/haddock2.4/) — data-driven docking with GPU MD refinement; RoseTTAFold (https://github.com/RosettaCommons/RoseTTAFold) — two-track network for complex prediction.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuFFT for rigid-body FFT correlation in 3D; GPU-parallel spherical harmonic expansion; PyTorch CUDA equivariant GNN layers for DL docking; GPU-batched energy evaluation for docking pose refinement. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
