# 2.16 — ΔΔG Stability Prediction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.16`
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

Predicting the thermodynamic stability change upon single amino acid mutation (ΔΔG) is critical for protein engineering, antibody optimization, and understanding disease variants. ML approaches train on experimental ΔΔG datasets (Protherm, Megascale) using structural features (ProteinMPNN ddG, ThermoMPNN), sequence language models (ESM-1v, EVmutation), or structure-sequence joint models. GPU training on millions of mutation datapoints and GPU inference for saturation mutagenesis scanning (all 20 AA × every position) makes library-scale ΔΔG feasible.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

ProteinMPNN fixed-backbone energy decomposition, ESM-1v zero-shot log-likelihood scoring, Rosetta ddG monomer protocol (FoldX, Cartesian ddG), GNN per-residue embedding, saturation mutagenesis scanning.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/g-stability-prediction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/g-stability-prediction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\g-stability-prediction.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Protherm database — >25k experimental ΔΔG values (https://www.abren.net/protherm/); Megascale dataset — 2.5M thermodynamic stability measurements (https://github.com/Rocklin-Lab/cdna-display-proteolysis-datasets); ProteinGym substitutions benchmark (https://github.com/OATML-Markslab/ProteinGym); S669 curated stability benchmark (verify URL).

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

ThermoMPNN (https://github.com/Kuhlman-Lab/ThermoMPNN) — GPU ΔΔG prediction from ProteinMPNN; ProteinMPNN-ddG (https://github.com/PeptoneLtd/proteinmpnn_ddg) — saturation mutagenesis ΔΔG; ESM-1v (https://github.com/facebookresearch/esm) — zero-shot stability from language model; FoldX (https://foldxsuite.crg.eu) — fast empirical ΔΔG.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

GPU GNN inference for per-residue stability; batched language model forward passes (cuDNN attention); GPU saturation mutagenesis via batched masked prediction; PyTorch Distributed for large-scale training. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
