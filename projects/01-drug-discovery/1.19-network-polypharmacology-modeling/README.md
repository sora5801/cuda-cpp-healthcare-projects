# 1.19 — Network / Polypharmacology Modeling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.19`
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

Polypharmacology recognizes that drugs interact with multiple targets, creating complex biological networks. GPU-accelerated graph neural networks on drug-target interaction (DTI) networks, protein-protein interaction (PPI) networks, and disease-gene networks enable systems-level prediction of off-target effects, drug combinations, and drug repurposing. Large-scale heterogeneous graph training (heterogeneous GNN, knowledge graph embeddings) with millions of nodes requires GPU memory and compute. GPU-parallel network perturbation simulations assist target identification.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Heterogeneous graph neural networks, knowledge graph embedding (TransE, RotatE), drug-target interaction prediction (DeepDTA, GraphDTA), network diffusion, community detection, drug combination synergy prediction (DeepSynergy).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/network-polypharmacology-modeling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/network-polypharmacology-modeling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\network-polypharmacology-modeling.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: STRING PPI network — 11.8B protein interactions (https://string-db.org); DrugBank — FDA-approved drugs and targets (https://go.drugbank.com); STITCH — drug-protein interactions (http://stitch.embl.de); DrugComb — drug combination synergy data (https://drugcomb.fimm.fi).

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

PyTorch Geometric (https://github.com/pyg-team/pytorch_geometric) — GPU heterogeneous graph learning; DGL (https://github.com/dmlc/dgl) — GPU graph learning for DTI networks; DeepPurpose (https://github.com/kexinhuang12345/DeepPurpose) — drug-target interaction prediction toolkit; HeteroMed (verify URL) — heterogeneous medical knowledge graphs.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

PyTorch Geometric CUDA sparse tensor operations for heterogeneous GNN; cuSPARSE for adjacency matrix products; GPU-batched negative sampling; FP16 embedding tables for large entity sets. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
