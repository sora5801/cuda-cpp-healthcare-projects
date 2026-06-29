# 13.7 — Adverse-Event & Pharmacovigilance Signal Detection

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Pharmacology%20%26%20Clinical%20Quantitative%20Modeling-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 13: Pharmacology & Clinical Quantitative Modeling · Catalog ID `13.7`
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

Detects unexpected drug safety signals from spontaneous reporting systems (FAERS, EudraVigilance) by applying disproportionality analysis and machine learning over millions of case reports. Reporting Odds Ratio (ROR) and Information Component (IC) calculations across all drug-AE pairs are parallelisable on GPU as batched sparse contingency table computations. Deep learning NLP models (BioBERT, ClinicalBERT) applied to FAERS narrative free-text are GPU-bound transformer inference. Longitudinal signal monitoring with Bayesian information component (multi-item gamma Poisson shrinker, MGPS) across a drug×AE matrix of 10⁶+ pairs requires GPU-resident sparse tensor operations.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Reporting Odds Ratio (ROR), Proportional Reporting Ratio (PRR), Multi-item Gamma Poisson Shrinker (MGPS), Bayesian Confidence Propagation Neural Network (BCPNN), NLP-based signal extraction (BERT NER on adverse event text), longitudinal CUSUM signal monitoring, graph-based drug-AE network analysis.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/adverse-event-pharmacovigilance-signal-detection.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/adverse-event-pharmacovigilance-signal-detection.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\adverse-event-pharmacovigilance-signal-detection.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: FDA FAERS (Adverse Event Reporting System) — 25M+ individual case safety reports (https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers) EudraVigilance — EMA adverse event reporting database (https://www.adrreports.eu/) WHO VigiAccess — global drug adverse reaction database (https://www.vigiaccess.org/) SIDER — side-effect data from drug package inserts (http://sideeffects.embl.de/)

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

PhViD (https://cran.r-project.org/web/packages/PhViD/) — R pharmacovigilance disproportionality package pyVigilance (verify URL) — Python FDA FAERS signal detection package BioBERT (https://github.com/dmis-lab/biobert) — GPU-pretrained biomedical BERT for FAERS NLP OpenVigil 2.1 (http://openvigil.pharmacology.uni-kiel.de/) — web-based pharmacovigilance signal detection tool

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuSPARSE for sparse drug-AE contingency matrix, cuBLAS for MGPS matrix operations, cuDNN for BERT-based NLP inference; pattern: batch-parallel disproportionality computation across all drug-AE pairs on GPU. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
