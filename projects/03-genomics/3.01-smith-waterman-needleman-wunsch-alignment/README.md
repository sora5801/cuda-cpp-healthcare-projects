# 3.01 — Smith-Waterman / Needleman-Wunsch Alignment

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟢 Beginner · Established** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.01`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Find the **best local alignment** between two DNA sequences with the
**Smith-Waterman** algorithm: fill a dynamic-programming score matrix, then read
off the highest-scoring path. The recurrence looks hopelessly serial — each cell
needs its top, left, and diagonal neighbours — yet the GPU extracts massive
parallelism by computing whole **anti-diagonals** at once. This project is the
deliberate *contrast* to the "independent jobs" pattern in `1.12`: same goal
(use every thread), completely different technique (a dependency wavefront).

## What this computes & why the GPU helps

Smith-Waterman computes the optimal local alignment via a quadratic DP score
matrix. At database scale (one query vs. millions of targets) that is enormous,
but the per-cell recurrence
`H[i][j] = max(0, H[i-1][j-1]+s, H[i-1][j]+gap, H[i][j-1]+gap)`
has a special structure: all cells on the same **anti-diagonal** (`i+j` constant)
are independent. GPUs collapse the serial dependency into **anti-diagonal
wavefront parallelism** — thousands of threads advance the frontier together.
Production tools (CUDASW++4.0) reach multiple TCUPS on data-center GPUs using
this idea plus hardware DP instructions and database batching.

**The parallel bottleneck** is filling the `O(M·N)` matrix; we parallelize each
anti-diagonal across threads (one cell per thread), sweeping `M+N-1` diagonals.

## The algorithm in brief

- **DP recurrence** (local, linear gap): `H[i][j] = max(0, diag+s, up+gap, left+gap)`.
- **Wavefront:** process anti-diagonals `d = i+j` in order; cells of one diagonal
  are independent.
- **Score** = the maximum cell; **traceback** from there to the first 0 recovers
  the aligned substrings.

See [THEORY.md](THEORY.md) for the full derivation (incl. how Needleman-Wunsch global
alignment differs).

## Build

Requires **Visual Studio 2026** (v145) + **CUDA 13.3** ([docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/smith-waterman-needleman-wunsch-alignment.sln`.
2. Select **`Release|x64`** → **Build Solution** →
   `build/x64/Release/smith-waterman-needleman-wunsch-alignment.exe`.

CLI: `msbuild build\smith-waterman-needleman-wunsch-alignment.sln /p:Configuration=Release /p:Platform=x64`

## Run the demo

```powershell
./demo/run_demo.ps1
```

Builds if needed, aligns the committed sequences, prints the score + alignment,
and verifies the GPU matrix equals the CPU matrix.

## Data

- **Sample (committed):** `data/sample/sequences_sample.txt` — two **synthetic**
  DNA sequences sharing a mutated motif (a clear local alignment).
- **Full data:** two FASTA records from UniProt/NCBI — see
  `scripts/download_data.ps1` and [data/README.md](data/README.md).
- Larger synthetic problem: `python scripts/make_synthetic.py --motif 400 --mut 0.2`.

## Expected output

`demo/expected_output.txt` holds the deterministic stdout (score, endpoint,
percent identity, aligned columns). The GPU wavefront (`src/kernels.cu`) and the
serial CPU DP (`src/reference_cpu.cpp`) fill the **same integer matrix**, so they
agree exactly (`matrix mismatches = 0`).

## Code tour

1. [`src/main.cu`](src/main.cu) — load, run CPU + GPU, compare matrices, traceback, print.
2. [`src/reference_cpu.h`](src/reference_cpu.h) — scoring constants, `SeqPair`, the DP & traceback prototypes + the wavefront idea.
3. [`src/kernels.cuh`](src/kernels.cuh) — the per-diagonal kernel interface + the anti-diagonal diagram.
4. [`src/kernels.cu`](src/kernels.cu) — the wavefront kernel + the host diagonal sweep.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial DP + traceback.

## Prior art & further reading

- **CUDASW++4.0** (<https://github.com/asbschmidt/CUDASW4>) — state-of-the-art GPU SW (DPX, database tiling).
- **NVIDIA GenomeWorks** (<https://github.com/NVIDIA-Genomics-Research/GenomeWorks>) — CUDA pairwise-alignment primitives.
- **WFA-GPU** (<https://github.com/quim0/WFA-GPU>) — wavefront *alignment* algorithm (gap-affine) on GPU.
- **Parasail** (<https://github.com/jeffdaily/parasail>) — SIMD reference library; good for cross-checking scores.

Study these for the production approach; reimplement didactically (CLAUDE.md §2).

## CUDA pattern used here

Anti-diagonal **wavefront** (extracting parallelism from a dependency structure) ·
one kernel launch per diagonal · integer DP (exact, deterministic) · serial
host-side traceback. Contrast with the embarrassingly-parallel `1.12`.

## Exercises

1. **Needleman-Wunsch.** Switch to global alignment: initialize row/col 0 to
   `i*GAP` / `j*GAP`, drop the `max(0, …)`, and read the score from `H[M][N]`.
2. **Affine gaps.** Add separate gap-open/gap-extend penalties (the Gotoh
   recurrence with three matrices H/E/F). How does the kernel change?
3. **Single-launch wavefront.** Replace the `M+N-1` launches with ONE kernel that
   loops over diagonals using a cooperative-groups grid barrier. Measure the win.
4. **Batched alignment.** Align one query against 10,000 targets, one CUDA block
   per pair. This is the genuinely GPU-favorable workload — compare its timing.
5. **Shared-memory tiling.** Cache the active diagonals in shared memory to cut
   global-memory traffic (the production optimization).

## Limitations & honesty

- **DNA + linear gaps only.** Real protein alignment uses a substitution matrix
  (BLOSUM/PAM) and affine gaps; described in THEORY.
- **Per-diagonal launches** make the GPU *slower* than the CPU on a single small
  pair (launch overhead). This is an honest teaching artifact — see THEORY.
- Traceback is serial on the host (the GPU teaching point is the parallel fill).
- The full `(M+1)·(N+1)` matrix is materialized; score-only variants need just
  two diagonals of memory.
