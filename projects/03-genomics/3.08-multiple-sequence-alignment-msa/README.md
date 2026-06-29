# 3.8 — Multiple Sequence Alignment (MSA)

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.8`
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

MSA aligns N sequences simultaneously, core to phylogenetics, variant analysis, and as input to protein structure prediction. Progressive MSA (ClustalW, MAFFT PartTree) first computes an N×N pairwise distance matrix (O(N²) SW comparisons), then builds a guide tree and folds sequences in. On GPU, the distance matrix computation is embarrassingly parallel—each thread block computes one pair—yielding reported 6× speedup for the MAFFT-PartTree distance phase on GPU. CUK-Band (2024) implements center-star MSA on GPU using banded DP. For protein MSA in AlphaFold2 pipelines, MMseqs2-GPU now accelerates the iterative search that builds deep MSAs, the most time-consuming preprocessing step.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Progressive alignment via guide tree (Neighbor-Joining); center-star alignment reduction; banded Smith-Waterman pairwise DP; profile-profile alignment; Sum-of-Pairs scoring; MAFFT Parttree distance matrix; iterative MSA refinement.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/multiple-sequence-alignment-msa.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/multiple-sequence-alignment-msa.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\multiple-sequence-alignment-msa.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: BAliBASE — benchmark MSA reference set (https://www.lbgi.fr/balibase/); HomFam — large homologous family MSA benchmark (verify URL); OXFam benchmark (verify URL); Pfam seed alignments (https://www.ebi.ac.uk/interpro/download/).

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

MAFFT (https://mafft.cbrc.jp/alignment/software/) — fastest large-scale CPU MSA with GPU-accelerated distance phase prototype; CUDA-ClustalW — parallel GPU progressive MSA (https://github.com/topics/multiple-sequence-alignment); CUK-Band (https://link.springer.com/chapter/10.1007/978-981-97-5692-6_8) — 2024 CUDA center-star MSA; MMseqs2 GPU (https://github.com/soedinglab/MMseqs2) — GPU-accelerated MSA search for structure prediction pipelines.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

One CUDA thread block per pairwise alignment (distance matrix phase); shared-memory banded DP; thrust for distance matrix sort; cuBLAS GEMM for profile-profile scoring; CUDA streams for guide-tree-ordered batch alignments. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
