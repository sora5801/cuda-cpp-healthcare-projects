# 1.11 — QSAR / Property Prediction

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.11`
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

Quantitative structure-activity relationship (QSAR) models predict biological activity from molecular descriptors or learned representations. Modern approaches use message-passing neural networks (MPNNs) over molecular graphs, enabling GPU-batched training on millions of labeled datapoints. The bottleneck shifts from feature computation to batch normalization and message aggregation over irregular graph structures — handled by PyTorch Geometric or DGL with CUDA backends. GPU-accelerated QSAR models at pharmaceutical companies screen hundreds of millions of virtual compounds per hour for ADMET and activity filters.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Directed message-passing (D-MPNN / Chemprop), graph convolutional networks (GCN), graph attention networks (GAT), transformer on molecular graphs (Uni-Mol), random forest / XGBoost on Morgan fingerprints, uncertainty quantification (ensemble, MCDropout).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/qsar-property-prediction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/qsar-property-prediction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\qsar-property-prediction.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: MoleculeNet — curated ML benchmark for 17+ molecular datasets (https://moleculenet.org); ChEMBL bioactivity data (https://www.ebi.ac.uk/chembl/); TDC (Therapeutics Data Commons) — 66 tasks for drug discovery ML (https://tdcommons.ai); PCBA (PubChem BioAssay) — 128 bioassays on 440k compounds (https://moleculenet.org).

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

Chemprop (https://github.com/chemprop/chemprop) — D-MPNN for molecular property prediction, GPU training; Uni-Mol (https://github.com/deepmodeling/Uni-Mol) — 3D molecular transformer pre-trained on 209M conformers; DeepChem (https://github.com/deepchem/deepchem) — broad GPU-accelerated ML chemistry toolkit; DGL-LifeSci (https://github.com/awslabs/dgl-lifesci) — graph neural networks for life science on GPU.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

PyTorch Geometric CUDA sparse tensor ops for graph batching; cuDNN for feedforward layers; FP16 mixed precision; GPU-accelerated descriptor generation via RDKit CUDA extensions (verify URL). --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
