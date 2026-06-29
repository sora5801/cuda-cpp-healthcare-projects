# 11.10 — Antibody Developability Prediction & Optimization

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biotechnology%2C%20Bioprocess%20%26%20Synthetic%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 11: Biotechnology, Bioprocess & Synthetic Biology · Catalog ID `11.10`
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

Even potent antibodies fail if they aggregate, have high viscosity, polyreact with off-targets, or are immunogenic — properties collectively called developability. Predicting all six key developability flags (pI, hydrophobicity, aggregation propensity, poly-specificity, expression level, immunogenicity) from sequence alone via GPU-trained BERT-style models enables early-stage winnowing of design libraries with millions of variants. Multi-property Pareto optimization across affinity and developability runs on GPU via multi-objective Bayesian optimization over learned surrogate surfaces.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Protein LLM fine-tuning for developability regression, multi-objective Bayesian optimization (qParEGO), aggregation prediction (camsol/spatial aggregation propensity), immunogenicity prediction (T-cell epitope presentation MHC-II), expression-level prediction from sequence.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/antibody-developability-prediction-optimization.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/antibody-developability-prediction-optimization.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\antibody-developability-prediction-optimization.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: SAFit dataset — self-association from AstraZeneca (verify URL via Bioinformatics journal); TAP dataset — Therapeutic Antibody Profiler developability (https://opig.stats.ox.ac.uk/webapps/oas/tap); OAS (https://opig.stats.ox.ac.uk/webapps/oas/oas) — natural antibody sequence space for pre-training; CoV-AbDab (https://opig.stats.ox.ac.uk/webapps/covabdab/) — experimental affinity + neutralization data.

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

Therapeutic Antibody Profiler (TAP) (https://opig.stats.ox.ac.uk/webapps/oas/tap) — web server + scoring functions; AbLang (https://github.com/oxpig/AbLang) — antibody language model pre-training; AntiFold (https://github.com/oxpig/AntiFold) — GPU antibody inverse folding for sequence redesign; ANARCI (https://github.com/oxpig/ANARCI) — antibody numbering for feature alignment.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN for Transformer LLM inference over antibody sequence batches, Flash Attention for variable-length CDR context, CUDA kernels for parallel developability feature computation; pattern: million-variant library → batch GPU LLM embedding → multi-property regression → GPU Pareto front computation → top candidates advance to wet-lab synthesis. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
