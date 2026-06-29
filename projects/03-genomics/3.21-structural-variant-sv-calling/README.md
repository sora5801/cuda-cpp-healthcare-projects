# 3.21 — Structural Variant (SV) Calling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.21`
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

Structural variants (deletions, insertions, inversions, translocations ≥50 bp) are detected by read-support signatures: split reads, discordant pairs, and assembly-based breakpoint realignment. GPU acceleration applies at two points: (1) rapid re-alignment of split-read candidates using banded SW to pinpoint breakpoints precisely, and (2) batched deep learning inference (convolutional models on pileup images) to genotype and filter SVs. Sniffles2 uses a fast clustering algorithm for ONT/HiFi; pbsv uses local realignment. GPU-accelerated genotyping (similar to DeepVariant's image-based approach) is an emerging direction for SV filtering at population scale.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Split-read alignment and breakpoint clustering; discordant pair signature scoring; local assembly with miniasm/hifiasm at breakpoints; convolutional image-based genotyping (DeepSV style); SV merging across samples (SURVIVOR); genotype likelihood calculation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/structural-variant-sv-calling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/structural-variant-sv-calling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\structural-variant-sv-calling.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: GiaB SV benchmark (HG002) — gold-standard deletion/insertion/inversion calls (https://www.nist.gov/programs-projects/genome-bottle); PacBio SV benchmark (https://github.com/PacificBiosciences/sv-benchmark); 1000 Genomes SV catalog (https://www.internationalgenome.org/data); ENCODE long-read SV studies (https://www.encodeproject.org/).

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

Sniffles2 (https://github.com/fritzsedlazeck/Sniffles) — fast ONT/HiFi SV caller; PBSV (https://github.com/PacificBiosciences/pbsv) — PacBio SV caller; cuteSV (https://github.com/tjiangHIT/cuteSV) — clustering-based SV caller; NGSEP (https://github.com/NGSEP/NGSEPcore) — variant calling suite with GPU-amenable CNN scoring.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Banded SW CUDA kernels for breakpoint realignment; cuDNN CNN for SV image genotyping; batched pileup image inference; thrust for read cluster sorting; multi-GPU for population-scale SV genotyping. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
