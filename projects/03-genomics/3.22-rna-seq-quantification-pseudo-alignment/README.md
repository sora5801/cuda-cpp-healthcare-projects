# 3.22 — RNA-seq Quantification / Pseudo-alignment

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟢 Beginner · Established** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.22`
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

Pseudo-alignment (kallisto, Salmon) bypasses full read alignment by mapping k-mers directly to equivalence classes of transcripts, then running the EM algorithm to estimate transcript abundances. GPU acceleration of kallisto redesigns the k-mer compatibility look-up and EM optimisation for GPU throughput: k-mer hash table queries map naturally to parallel GPU hash probes, and the EM update over millions of reads is a dense GEMV. A 2026 study ("RNA-seq analysis in seconds using GPUs," Melsted et al.) demonstrates GPU kallisto completing quantification in seconds vs. minutes on CPU. Salmon's variational Bayes EM is similarly GPU-amenable.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

K-mer de Bruijn graph construction for transcriptome index; pseudoalignment compatibility class assignment; expectation-maximisation (EM) for abundance estimation; variational Bayes EM (Salmon); bootstrap resampling for uncertainty; quasi-mapping hash-based alignment.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/rna-seq-quantification-pseudo-alignment.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/rna-seq-quantification-pseudo-alignment.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\rna-seq-quantification-pseudo-alignment.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: GENCODE human transcriptome — reference transcript index (https://www.gencodegenes.org/); ENCODE RNA-seq FASTQs — diverse cell-type transcriptomes (https://www.encodeproject.org/); GTEx v9 — tissue RNA-seq compendium (https://gtexportal.org/); SRA RNA-seq studies (https://www.ncbi.nlm.nih.gov/sra).

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

kallisto GPU branch (https://github.com/pachterlab/kallisto) — GPU branch for pseudo-alignment; Salmon (https://github.com/COMBINE-lab/salmon) — quasi-mapping quantification (GPU EM target); bustools (https://github.com/BUStools/bustools) — BUS file manipulation for scRNA-seq downstream; alevin-fry (https://github.com/COMBINE-lab/alevin-fry) — fast single-cell quantification, GPU-amenable.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

GPU hash table for k-mer to equivalence class look-up; custom EM kernel (sparse GEMV per read per EM iteration); warp-level reduction for abundance accumulation; cuSPARSE for sparse equivalence class matrices; CUDA streams for I/O and compute overlap. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
