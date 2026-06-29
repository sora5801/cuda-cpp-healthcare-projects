# 13.4 — Drug-Drug Interaction Prediction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Pharmacology%20%26%20Clinical%20Quantitative%20Modeling-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 13: Pharmacology & Clinical Quantitative Modeling · Catalog ID `13.4`
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

Predicts pharmacokinetic drug-drug interactions (PK-DDI) caused by CYP enzyme inhibition/induction, transporter competition, and protein binding displacement; also predicts pharmacodynamic DDI from synergistic/antagonistic receptor effects. Graph neural networks encode drug molecular structure; bipartite interaction graphs model shared enzyme substrates. GPU parallelism across large drug-pair combination spaces is essential — the DrugBank DDI graph has ~250k interaction edges from 2.4k drugs, but virtual screening explores millions of hypothetical pairs. Static mechanistic models (R-value, AUC ratio prediction) are solved in batched parallel ODEs on GPU for all pairs simultaneously.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

GNN on drug molecular graphs with edge-level DDI prediction, DeepDDI (sequence-based DDI), TransE/RotatE knowledge graph embedding for DDI, R-value static mechanistic model, AUC ratio DDI prediction, CYP inhibition ODE models, PBPK-embedded DDI simulation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/drug-drug-interaction-prediction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/drug-drug-interaction-prediction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\drug-drug-interaction-prediction.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: DrugBank DDI — 250k+ drug interaction records with mechanism (https://www.drugbank.com/) TWOSIDES — 3.7M adverse event pairs from spontaneous reports (verify URL; originally published by Tatonetti lab) OFFSIDES — off-label adverse effects dataset (verify URL; Tatonetti lab) FDA Adverse Event Reporting System (FAERS) (https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers)

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

DeepDDI (https://github.com/NCIBI/DeepDDI) — deep learning DDI prediction from drug SMILES SkipGNN (verify URL) — graph neural network for DDI on drug interaction graphs TorchDrug (https://github.com/DeepGraphLearning/torchdrug) — GPU molecular GNN framework applicable to DDI STITCH — chemical-protein interactions database (http://stitch.embl.de/) with downloadable interaction files

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

DGL/PyG sparse message passing on drug interaction graphs, cuBLAS for PBPK ODE Jacobians, custom CUDA DDI scoring kernels; pattern: batch-parallel DDI pair scoring over millions of drug combinations on GPU. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
