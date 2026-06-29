# 11.8 — CRISPR System Design & Modeling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biotechnology%2C%20Bioprocess%20%26%20Synthetic%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 11: Biotechnology, Bioprocess & Synthetic Biology · Catalog ID `11.8`
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

CRISPR guide RNA (gRNA) design requires genome-wide off-target site enumeration (all 20-mer matches with ≤4 mismatches in 3 billion base pairs), scoring each off-target's likelihood based on mismatch position and type. GPU-accelerated exact string matching (GPU BWT/FM-index) reduces the genome scanning from hours to minutes. Deep learning off-target predictors (CNN, BiGRU, BERT-based LLMs) run on GPU over millions of candidate gRNAs in parallel. The CRISOT tool suite derives RNA-DNA molecular interaction fingerprints from GPU-accelerated MD simulations of Cas9-gRNA-DNA ternary complexes to compute structural off-target scores.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

FM-index / BWT genome search on GPU, CNN/BiGRU/Transformer off-target classifiers, molecular dynamics of Cas9 R-loop formation, energy minimization for gRNA thermodynamic stability, seqmap-style GPU hash table for rapid k-mer matching, CRISOT molecular fingerprinting.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/crispr-system-design-modeling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/crispr-system-design-modeling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\crispr-system-design-modeling.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: CRISPOR Guide RNA Dataset — experimentally validated on/off-target activities (https://crispor.tefor.net/); CIRCLE-seq Off-Target Dataset (Tsai et al., Nature Methods) — unbiased off-target identification; Genome-wide CRISPR off-target benchmark (https://www.nature.com/articles/s41467-023-42695-4); ClinVar — disease-relevant on-target loci for therapeutic gRNA selection (https://www.ncbi.nlm.nih.gov/clinvar/).

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

CRISPOR (https://github.com/maximilianh/crisporWebsite) — GPU-accelerated guide design pipeline; CRISPRscan (https://www.crisprscan.org/) — on/off-target prediction (verify GitHub URL); DeepCRISPR (https://github.com/jieccccc/DeepCRISPR) — CNN off-target prediction with GPU inference; GROMACS (https://github.com/gromacs/gromacs) — GPU MD of Cas9 R-loop for CRISOT-style fingerprinting.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA BWT string index for GPU genome scanning, cuDNN for CNN/Transformer off-target scoring over batches of gRNAs, cuRAND for MD trajectory generation; pattern: 20-mer gRNA → GPU BWT scan of genome → candidate off-target list → batch GPU DL scoring → filter by specificity score → MD fingerprint for top candidates. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
