# 11.7 — mRNA / Vaccine Sequence Design

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biotechnology%2C%20Bioprocess%20%26%20Synthetic%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 11: Biotechnology, Bioprocess & Synthetic Biology · Catalog ID `11.7`
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

mRNA vaccine efficacy depends on optimal codon usage (for high ribosome translation), minimum free-energy (MFE) secondary structure (for stability), and 5'-UTR/3'-UTR element design. LinearDesign finds near-optimal MFE + CAI jointly in 11 minutes via dynamic programming on a lattice (analogous to CYK parsing), and GPU parallelization of the lattice can further accelerate multi-target vaccine design. VaxPress (2024) runs iterative codon optimization with customizable scoring functions including codon adaptation index, GC content, repeat minimization, and vaccine-specific immune stimulation features. Deep generative models (Nature Communications 2025) optimize codon sequences via GPU-trained VAEs, improving translation efficiency measurably in cell-free expression.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Minimum-free-energy (MFE) RNA folding (Zuker dynamic programming), codon adaptation index (CAI) optimization, LinearDesign lattice-DP algorithm, epitope prediction (MHC-I/II binding), RNA-structure gradient optimization, deep generative codon design (VAE/flow matching).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/mrna-vaccine-sequence-design.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/mrna-vaccine-sequence-design.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\mrna-vaccine-sequence-design.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: NCBI RefSeq CDS — validated coding sequences for codon usage tables (https://www.ncbi.nlm.nih.gov/refseq/); RNAcentral — non-coding RNA + UTR sequences (https://rnacentral.org/); VaxPress Test Suite — 100 vaccine antigens for benchmarking (https://github.com/ChangLabSNU/VaxPress); IEDB — immune epitope database for T/B cell responses (https://www.iedb.org/).

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

LinearDesign (https://github.com/LinearDesignSoftware/LinearDesign) — fast MFE+CAI co-optimization; VaxPress (https://github.com/ChangLabSNU/VaxPress) — codon optimizer with LinearDesign integration; VaxLab (https://github.com/ChangLabSNU/VaxLab) — integrated design platform; CodonBERT (verify URL, search "CodonBERT GitHub") — BERT-based codon optimization model (GPU inference).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN for Transformer-based codon sequence scoring (CodonBERT), CUDA dynamic-programming kernels for parallel MFE computation across sequence windows, Flash Attention for long mRNA sequence context; pattern: target antigen CDS → GPU LinearDesign DP → VaxPress iterative refinement on GPU → GPU epitope scoring → ranked candidates. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
