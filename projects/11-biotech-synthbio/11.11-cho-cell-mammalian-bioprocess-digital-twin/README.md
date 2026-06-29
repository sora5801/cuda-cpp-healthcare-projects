# 11.11 — CHO Cell & Mammalian Bioprocess Digital Twin

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biotechnology%2C%20Bioprocess%20%26%20Synthetic%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 11: Biotechnology, Bioprocess & Synthetic Biology · Catalog ID `11.11`
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

Chinese Hamster Ovary (CHO) cell fed-batch cultures for monoclonal antibody production exhibit complex interplay of metabolism, glycosylation, dissolved oxygen, and pH dynamics that are expensive to characterize experimentally. GPU-accelerated hybrid digital twins (Nature npj 2026) couple ODE kinetic models with genome-scale FBA on GPU, with LSTM networks trained on GPU correcting model-plant mismatch online. Bayesian parameter estimation with HMC (GPU-accelerated via NumPyro/JAX) fits hundreds of kinetic parameters to multi-omics fed-batch data in hours. Real-time digital twins receive PAT (process analytical technology) sensor streams and predict glycoform distributions ahead of time for automated feeding control.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Hybrid mechanistic-ML (ODE + LSTM), genome-scale metabolic modeling (FBA, GEM reduction), Bayesian HMC parameter estimation, Gaussian process regression for process uncertainty, PLS/PCA for spectroscopic soft sensing, dynamic FBA (dFBA).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/cho-cell-mammalian-bioprocess-digital-twin.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/cho-cell-mammalian-bioprocess-digital-twin.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\cho-cell-mammalian-bioprocess-digital-twin.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: CHO Fed-Batch Time-Course Metabolomics (BioRxiv 2025, Zenodo) — 12 cultures with 80+ metabolite time profiles; BioNumbers Database — CHO-specific growth/uptake rates (https://bionumbers.hms.harvard.edu/); BioModels Database — published CHO kinetic models (https://www.ebi.ac.uk/biomodels/); JGI/DBTBS gene expression compendium for CHO pathway analysis (verify URL).

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

COBRApy (https://github.com/opencobra/cobrapy) — GEM FBA for CHO; NumPyro (https://github.com/pyro-ppl/numpyro) — GPU Bayesian HMC for kinetic parameter estimation; PyTorch LSTM (https://pytorch.org/) — hybrid ODE-LSTM digital twin training; Pyomo (https://github.com/Pyomo/pyomo) — algebraic modeling for dynamic FBA optimization.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN for LSTM training/inference, JAX GPU backend for HMC MCMC, CUDA batch LP for parallel FBA across time points; pattern: online PAT sensor feed → GPU LSTM state update → GPU GEM FBA at current metabolite concentrations → kinetic ODE integration → feeding strategy MPC → compare to lab measurements → Bayesian posterior update. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
