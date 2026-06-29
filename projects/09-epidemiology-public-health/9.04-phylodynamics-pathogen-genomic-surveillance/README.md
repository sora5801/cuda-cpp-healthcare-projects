# 9.4 — Phylodynamics & Pathogen Genomic Surveillance

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Epidemiology%20%26%20Public%20Health-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 9: Epidemiology & Public Health · Catalog ID `9.4`
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

Infers the evolutionary and epidemiological history of pathogens from genomic sequences using Bayesian phylodynamic models (BEAST2, TreeTime). The computational bottleneck is evaluating the phylogenetic likelihood across millions of trees sampled by MCMC — each likelihood evaluation requires computing evolutionary substitution probabilities across thousands of sequence sites and tree branches. BEAGLE (Broad-platform Evolutionary Analysis General Likelihood Evaluator) provides a GPU-accelerated library for this core computation, delivering 20–50× speedup over CPU BEAST. GPU-accelerated variant calling pipelines (DNAnexus, NVIDIA Parabricks) feed surveillance outputs into phylodynamic pipelines.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Bayesian phylogenetic MCMC (Metropolis-Hastings), HKY/GTR nucleotide substitution models, Kingman's coalescent, birth-death diversification models, skyline population size estimation, ancestral state reconstruction, phylogeographic diffusion, TreeTime maximum-likelihood dating.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/phylodynamics-pathogen-genomic-surveillance.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/phylodynamics-pathogen-genomic-surveillance.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\phylodynamics-pathogen-genomic-surveillance.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: GISAID — 15M+ SARS-CoV-2 and influenza sequences with metadata (https://www.gisaid.org/) NCBI Pathogen Detection Database — real-time foodborne pathogen genomics (https://www.ncbi.nlm.nih.gov/pathogens/) GenBank — nucleotide sequence archive for all pathogens (https://www.ncbi.nlm.nih.gov/genbank/) Nextstrain data pipelines — curated SARS-CoV-2, influenza, mpox builds (https://nextstrain.org/)

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

BEAST 2 (https://www.beast2.org/) — Bayesian phylogenetic inference; GPU via BEAGLE library BEAGLE (https://github.com/beagle-dev/beagle-lib) — GPU-accelerated phylogenetic likelihood library (CUDA/OpenCL) Nextstrain (https://github.com/nextstrain/augur) — real-time pathogen genomic surveillance pipeline NVIDIA Parabricks (https://github.com/clara-parabricks) — GPU-accelerated variant calling and genome analysis

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

BEAGLE CUDA kernels for transition probability matrix exponentiation across tree branches, cuBLAS for substitution rate matrix multiplies; pattern: embarrassingly parallel site-likelihood computation across sequence columns, aggregated with parallel prefix products across tree branches. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
