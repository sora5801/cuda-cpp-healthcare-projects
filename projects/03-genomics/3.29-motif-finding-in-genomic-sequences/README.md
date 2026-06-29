# 3.29 — Motif Finding in Genomic Sequences

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.29`
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

Transcription factor motif discovery from ChIP-seq peaks searches for over-represented sequence patterns (IUPAC or position weight matrices) against a background model. Expectation-Maximisation over all N×W sequence windows (N peaks × W-k+1 positions per peak) is O(N×W×4^k) for exhaustive search; GPU parallelism assigns one thread to each window position, computing the PWM score via a parallel dot product. mCUDA-MEME achieves orders-of-magnitude speedup by distributing MEME's EM steps across GPU cores and GPU clusters. For genome-scale ChIP-seq (millions of peaks), this turns multi-day CPU runs into hours.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

MEME expectation-maximisation over sequence windows; position weight matrix (PWM) scoring; ZOOPS/OOPS/TCM motif occurrence models; FIMO discrete log-sum-over-PWM scoring; Gibbs sampling for motif discovery; JASPAR database PWM matching.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/motif-finding-in-genomic-sequences.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/motif-finding-in-genomic-sequences.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\motif-finding-in-genomic-sequences.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: ENCODE ChIP-seq peak BED files — thousands of TF experiments (https://www.encodeproject.org/); JASPAR 2024 — curated PWM database (https://jaspar.elixir.no/); ReMap 2022 — regulatory elements from 5 k ChIP-seq experiments (https://remap.univ-amu.fr/); GEO ChIP-seq datasets (https://www.ncbi.nlm.nih.gov/geo/).

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

CUDA-MEME / mCUDA-MEME (https://cuda-meme.sourceforge.io/homepage.htm) — GPU cluster MEME, ultrafast motif discovery; Argo_CUDA (https://pubmed.ncbi.nlm.nih.gov/29281953/) — exhaustive GPU motif discovery for large datasets; MEME Suite (https://meme-suite.org/) — reference CPU motif toolkit; HOMER (https://github.com/samtools/homer — verify URL, originally http://homer.ucsd.edu/) — CPU ChIP-seq motif enrichment tool.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

One CUDA thread per sequence window for PWM scoring; shared-memory PWM matrix loaded once per kernel; warp-level sum for log-probability accumulation; thrust for top-k motif score extraction; batched EM outer loops with inter-GPU synchronisation. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
