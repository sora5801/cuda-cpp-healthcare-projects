# 3.10 — RNA Secondary-Structure Prediction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.10`
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

RNA folds into hairpins and stems governed by free-energy minimisation via the Zuker algorithm (O(n³) time, O(n²) space). For sequences >10 kb (rRNA, lncRNA), the cubic cost is prohibitive on CPU. GPU parallelism exploits the diagonal wavefront of the DP table: all cells (i,j) on the same diagonal d=j-i are independent and can be updated simultaneously by CUDA threads, similar to SW alignment. CUDA RNAfold achieves 14× speedup for sequences up to 30 kb. LinearFold reduces the complexity to O(n) using a beam-search approximation and lends itself to GPU batch processing of thousands of short RNAs in parallel.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Zuker free-energy minimisation (partition function DP); McCaskill partition function (base-pair probabilities); anti-diagonal wavefront parallelism; LinearFold beam-search O(n); Vienna RNA thermodynamic model; stochastic context-free grammar (SCFG) parsing.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/rna-secondary-structure-prediction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/rna-secondary-structure-prediction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\rna-secondary-structure-prediction.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Rfam — RNA family alignments and secondary structures (https://rfam.org/); RNAcentral — comprehensive RNA sequence database (https://rnacentral.org/); PDB RNA structures — known 3D-validated secondary structures (https://www.rcsb.org/); ArchiveII benchmark — curated RNA secondary structure data (verify URL).

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

CUDA RNAfold (https://www.biorxiv.org/content/10.1101/298885v1.full) — GPU-parallelised Vienna RNAfold, 14× speedup; LinearFold (https://github.com/LinearFold/LinearFold) — O(n) RNA folding with GPU batch variant; LinearAlifold (https://github.com/LinearFold/LinearAlifold) — consensus structure prediction; EternaFold (https://github.com/eternagame/EternaFold) — ML-trained folding model for GPU inference.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Anti-diagonal wavefront kernel (custom CUDA, shared-memory tiling of DP triangle); one warp per diagonal cell group; thrust for energy table initialization; cuFFT (not standard here, but used in some spectral RNA analyses); batch RNA folding with one CTA per sequence. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
