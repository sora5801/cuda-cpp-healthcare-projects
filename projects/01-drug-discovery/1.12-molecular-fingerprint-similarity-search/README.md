# 1.12 — Molecular Fingerprint Similarity Search

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.12`
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

Tanimoto similarity between Morgan/ECFP bit-vectors is the standard metric for chemical similarity searching. Brute-force comparison of a query against a library of 100M compounds requires 10^10 bit-AND/popcount operations — ideally suited for GPU SIMD. Each 2048-bit fingerprint fits in 32 uint64 words; a GPU thread evaluates one query-vs-library pair in ~5 ns. Schrodinger's gpusimilarity loads an entire library into GPU memory and achieves sub-second retrieval on billion-compound libraries. The GPU pattern is embarrassingly data-parallel with a final topK reduction.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Tanimoto coefficient (Jaccard on bit-vectors), Morgan/ECFP fingerprints (radius 2–3), TopK reduction, LSH-based approximate search, Faiss IVF for high-dimensional vectors.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/molecular-fingerprint-similarity-search.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/molecular-fingerprint-similarity-search.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\molecular-fingerprint-similarity-search.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: ChEMBL (https://www.ebi.ac.uk/chembl/); ZINC20 (https://zinc20.docking.org); PubChem Compound — 115M+ compounds (https://pubchem.ncbi.nlm.nih.gov); Enamine REAL (https://enamine.net).

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

gpusimilarity (https://github.com/schrodinger/gpusimilarity) — CUDA/Thrust brute-force fingerprint search; FPSim2 (https://github.com/chembl/FPSim2) — fast similarity search using PyTables and GPU-accelerated popcount; RDKit (https://github.com/rdkit/rdkit) — cheminformatics toolkit with Morgan fingerprint generation; Faiss (https://github.com/facebookresearch/faiss) — GPU-accelerated ANN search applicable to molecular embeddings.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Thrust device_vector for library storage; custom CUDA kernels with __popcll() for bit-count; warp-shuffle reduction for partial Tanimoto sums; GPU topK using cub::DeviceRadixSort; texture memory for fingerprint cache. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
