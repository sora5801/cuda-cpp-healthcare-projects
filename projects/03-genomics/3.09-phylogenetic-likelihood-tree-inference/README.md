# 3.9 — Phylogenetic Likelihood / Tree Inference

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.9`
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

Maximum-likelihood phylogenetic inference evaluates the Felsenstein pruning recursion—computing site likelihood at each internal node by multiplying branch transition probability matrices (4×4 or 20×20 per site, per node) up the tree—for millions of alignment columns and hundreds of tree search moves (NNI, SPR). For large trees (thousands of taxa, genome-scale alignments), the log-likelihood computation is the bottleneck and is embarrassingly parallel across alignment sites. Bayesian phylogenetics (MrBayes) runs thousands of MCMC steps each requiring full-tree likelihood evaluation; GPU acceleration reported 63× speedup vs. serial CPU by assigning each site to a thread. RAxML-NG and IQ-TREE GPU are active development targets.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Felsenstein pruning / Felsinstein's pruning recursion; substitution model matrix exponentiation (GTR, WAG, LG); nearest-neighbor interchange (NNI) and subtree pruning/regrafting (SPR) tree search; Metropolis-Hastings MCMC (Bayesian); bootstrap resampling.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/phylogenetic-likelihood-tree-inference.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/phylogenetic-likelihood-tree-inference.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\phylogenetic-likelihood-tree-inference.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: TreeBASE — curated phylogenetic alignments and trees (https://www.treebase.org/); SILVA rRNA database — large rRNA alignment for phylogenetics (https://www.arb-silva.de/); NCBI CDD — conserved domain alignments (https://www.ncbi.nlm.nih.gov/Structure/cdd/cdd.shtml); OpenTreeOfLife — aggregated phylogenetic data (https://opentreeoflife.github.io/).

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

IQ-TREE 2 (https://iqtree.github.io/) — state-of-the-art ML tree inference (GPU extension in development); RAxML-NG (https://github.com/amkozlov/raxml-ng) — fast ML inference with GPU acceleration hooks; MrBayes (https://github.com/NBISweden/MrBayes) — Bayesian inference with CUDA-accelerated site likelihood; BeagleLib (https://github.com/beagle-dev/beagle-lib) — GPU-accelerated phylogenetic likelihood library used by MrBayes/BEAST.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

BeagleLib uses custom CUDA kernels for 4×4/20×20 matrix-vector products per site per node; one CUDA thread per alignment site within a likelihood pass; cuBLAS for transition matrix exponentiation; multi-GPU over tree partitions. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
