# 7.2 — Drug-Target Interaction Prediction (GNN)

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20AI%20%26%20Clinical%20Deep%20Learning-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 7: Medical AI & Clinical Deep Learning · Catalog ID `7.2`
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

Predicts whether a small molecule (drug) will bind to a protein target and estimates binding affinity (Kd/Ki) or binary interaction labels. Molecular graphs have irregular topology, so graph neural message-passing aggregates neighbour features in parallel across thousands of candidate pairs simultaneously on GPU. Protein sequences can be encoded via transformer attention (ESM-2, ProtTrans) whose quadratic attention is accelerated by Flash Attention on CUDA. The bottleneck is the cross-attention between drug graph embeddings and protein sequence embeddings over large virtual screening libraries (millions of compounds), which maps to batched sparse matrix operations. GPU throughput determines how many candidates can be scored per day in drug discovery pipelines.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Message Passing Neural Networks (MPNN), Graph Attention Networks (GAT), Directed Message Passing (DMPNN), transformer cross-attention, contrastive DTI objectives, Graph Isomorphism Networks (GIN), graph-level pooling, Bayesian hyperparameter optimisation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/drug-target-interaction-prediction-gnn.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/drug-target-interaction-prediction-gnn.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\drug-target-interaction-prediction-gnn.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: BindingDB — ~2.9 million measured binding affinities for drug-target pairs (https://www.bindingdb.org/) ChEMBL — curated bioactivity database with >20M activity records (https://www.ebi.ac.uk/chembl/) Davis Kinase Dataset — kinase inhibitor affinities for 442 kinases × 68 drugs (verify URL) KIBA — integrated kinase inhibitor bioactivity benchmark (verify URL)

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

DeepPurpose (https://github.com/kexinhuang12345/DeepPurpose) — 15 drug/protein encoders, 50+ architectures for DTI TorchDrug (https://github.com/DeepGraphLearning/torchdrug) — GPU-accelerated graph learning library for drug discovery DGL-LifeSci (https://github.com/awslabs/dgl-lifesci) — DGL-based molecular GNN toolkit with CUDA-backed sparse ops DTA-GNN (https://github.com/lennylv/DTA-GNN) — toolkit for target-specific DTA dataset construction and GNN training

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

DGL/PyG sparse adjacency ops on GPU, Flash Attention 2 for protein encoders, cuDNN for MLP heads; pattern: heterogeneous data parallelism (drug batch × protein batch), optional multi-GPU model parallelism for large protein encoders. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
