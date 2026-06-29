# 3.1 — Smith-Waterman / Needleman-Wunsch Alignment

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟢 Beginner · Established** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.1`
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

Smith-Waterman (SW) computes the optimal local alignment between two sequences via a dynamic-programming (DP) score matrix filled cell-by-cell; at protein-database scale this means quadratic work per query against millions of targets. GPUs collapse this into anti-diagonal wavefront parallelism: all cells on the same anti-diagonal are independent and can be computed simultaneously across thousands of CUDA threads, eliminating the serial dependency that cripples CPUs. CUDASW++4.0 (2024) achieves up to 5.71 TCUPS on an H100 by exploiting Hopper's DPX integer-DP instructions, hardware-native to the architecture, alongside tile-based matrix partitioning and sequence-database chunking for maximal occupancy. The specific bottleneck parallelised is the per-cell recurrence max(H[i-1,j-1]+s, H[i,j-1]-g, H[i-1,j]-g) across the anti-diagonal frontier.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Smith-Waterman anti-diagonal DP wavefront; Needleman-Wunsch global DP; striped SIMD inter-sequence parallelism; affine gap scoring; DPX hardware DP instructions (Hopper); sequence-database tiling and batched kernel launch.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/smith-waterman-needleman-wunsch-alignment.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/smith-waterman-needleman-wunsch-alignment.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\smith-waterman-needleman-wunsch-alignment.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: UniProtKB/Swiss-Prot — curated protein sequence database, ~570 k entries (https://www.uniprot.org/downloads); NCBI nr (non-redundant protein) — comprehensive protein database, 100 M+ sequences (https://ftp.ncbi.nlm.nih.gov/blast/db/); PDB sequences — structural protein sequences for benchmarking alignments (https://www.rcsb.org/downloads); NCBI RefSeq — reference nucleotide and protein sequences (https://ftp.ncbi.nlm.nih.gov/refseq/).

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

CUDASW4 (https://github.com/asbschmidt/CUDASW4) — CUDASW++4.0, H100/A100/L40S optimised, DPX, up to 5.71 TCUPS; GenomeWorks / ClaraGenomics SDK (https://github.com/NVIDIA-Genomics-Research/GenomeWorks) — NVIDIA CUDA pairwise alignment primitives for both protein and nucleotide; WFA-GPU (verify URL: github.com/quim0/WFA-GPU) — wavefront alignment algorithm on GPU, gap-affine, ultra-fast for long DNA; Parasail (https://github.com/jeffdaily/parasail) — SIMD/CUDA pairwise alignment library used as reference.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuBLAS (score accumulation); thrust (sort, scan); CUB (warp-level reduction); custom anti-diagonal kernels with shared memory tiling; inter-sequence batching (one CUDA block per query–target pair or striped across warps); DPX integer instructions on Hopper SM90. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
