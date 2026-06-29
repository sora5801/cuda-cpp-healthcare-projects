# 3.28 — Profile HMM (Viterbi / Forward)

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.28`
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

Profile HMMs (pHMMs) model protein families as position-specific probability distributions; HMMER3 searches databases by applying a cascade: MSV/SSV (Multi-Segment Viterbi) filter, P7Viterbi, and Forward-Backward scoring. MSV/SSV alone consumes 72% of runtime. CUDAMPF parallelises the MSV/Viterbi recurrence across database sequences: each CUDA thread block processes one query-profile versus one database sequence, computing the N×M score matrix in shared memory. For very deep database scans (>10⁹ sequences in metagenomics), GPU pHMM search reduces days to hours.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

MSV/SSV Multi-Segment Viterbi; P7Viterbi DP over profile-sequence grid; Forward-Backward algorithm (sum-product); Viterbi traceback; plan-7 profile HMM architecture; hit reporting with E-value calculation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/profile-hmm-viterbi-forward.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/profile-hmm-viterbi-forward.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\profile-hmm-viterbi-forward.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Pfam-A — 20 k protein family profiles (https://www.ebi.ac.uk/interpro/download/); UniRef50 — protein sequences for database search (https://www.uniprot.org/help/uniref); Rfam — RNA family profiles (https://rfam.org/); JGI metagenome proteins — environmental pHMM targets (https://genome.jgi.doe.gov/).

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

CUDAMPF (https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-016-0946-4) — multi-tiered CUDA HMMER acceleration; HMMER3 (https://github.com/EddyLab/hmmer) — CPU reference, CUDA port target; MMseqs2 profile search (https://github.com/soedinglab/MMseqs2) — faster alternative using k-mer prefilter; GPU-HMMER speculative search (verify URL) — speculative HMMER implementation on GPU.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom shared-memory MSV/Viterbi kernel (one block per sequence); vectorised score matrix with CUDA float4; CUB warp-level max for Viterbi path; multi-GPU sequence database partitioning; CUDA streams for I/O and compute overlap. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
