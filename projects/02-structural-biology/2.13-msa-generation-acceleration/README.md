# 2.13 — MSA Generation Acceleration

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.13`
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

Multiple sequence alignment (MSA) construction for AlphaFold2 is a major bottleneck: HHblits and Jackhmmer search the UniRef90 database (210GB) requiring hours of CPU time. GPU acceleration of profile hidden Markov model (HMM) search is an active area: GPU-HMMER uses CUDA to parallelize the Viterbi/Forward-Backward dynamic programming recursion over thousands of sequence targets simultaneously. Accelerating MSA generation could remove one of the last CPU-bound steps in the AF2 prediction pipeline, enabling rapid large-scale proteome annotation.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Profile HMM Viterbi algorithm, Forward-Backward DP, Smith-Waterman alignment, position-specific scoring matrix (PSSM) search, k-mer seed hashing, HHblits iterated profile-profile alignment.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/msa-generation-acceleration.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/msa-generation-acceleration.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\msa-generation-acceleration.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: UniRef90 — 210GB protein sequence database (https://www.uniprot.org/help/uniref); UniClust30 (https://uniclust.mmseqs.com); MGnify metagenomics sequences (https://www.ebi.ac.uk/metagenomics/); BFD — Big Fantastic Database (https://bfd.mmseqs.com).

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

MMseqs2 (https://github.com/soedinglab/MMseqs2) — ultra-fast protein search and clustering (GPU-capable via SIMD/GPU versions); ColabFold MSA server (https://github.com/sokrypton/ColabFold) — GPU-accelerated MSA for AlphaFold2; GPU-HMMER (verify URL) — CUDA Viterbi HMM search; Linclust (https://github.com/soedinglab/MMseqs2) — GPU-accelerated sequence clustering.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA DP recursion for HMM Viterbi (row-parallel); GPU parallel Smith-Waterman via CUDASW++; warp-parallel query-vs-target scoring; GPU hash tables for k-mer seed lookup. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
