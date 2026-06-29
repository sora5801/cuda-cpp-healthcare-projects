# 11.6 — Metabolic Engineering & Strain Design

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biotechnology%2C%20Bioprocess%20%26%20Synthetic%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 11: Biotechnology, Bioprocess & Synthetic Biology · Catalog ID `11.6`
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

Metabolic engineering seeks genetic modifications (gene knockouts, overexpression, heterologous pathway insertion) that maximize desired metabolite production. GPU acceleration enables genome-scale flux-balance analysis (FBA) to be solved for millions of genetic perturbation combinations in parallel — each FBA is an independent LP problem — dramatically outpacing CPU batch FBA. Constraint-based strain design algorithms (OptKnock, MOMA) search exponentially large combinatorial spaces, tractable only with GPU parallelism. Kinetic whole-pathway models (ODEs with hundreds of reactions) can be fitted to multi-omics data using GPU-accelerated Bayesian MCMC (NUTS/HMC).

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Flux Balance Analysis (LP, GPU batch), Dynamic FBA, OptKnock / RobustKnock strain design, ensemble kinetic modeling (EKM), Bayesian MCMC parameter estimation (NUTS/HMC), genome-scale metabolic network reduction (data-driven, 2025).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/metabolic-engineering-strain-design.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/metabolic-engineering-strain-design.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\metabolic-engineering-strain-design.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: BiGG Models — 108 genome-scale metabolic models (https://bigg.ucsd.edu/); KEGG Metabolic Pathways (https://www.kegg.jp/kegg/pathway.html); MetaboLights — metabolomics raw data (https://www.ebi.ac.uk/metabolights/); CHO-GEM Genome-Scale Model — CHO cell metabolic network (verify URL via Zenodo/BioModels).

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

COBRApy (https://github.com/opencobra/cobrapy) — FBA/FVA in Python; cameo (https://github.com/biosustain/cameo) — strain design algorithms including OptKnock; MICOM (https://github.com/micom-dev/micom) — microbiome community FBA; GPU-FBA (verify URL, search "GPU flux balance analysis CUDA") — CUDA batch LP solver for parallel strain enumeration.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA LP solver (per-combination parallel simplex/interior-point), cuBLAS for stoichiometric matrix operations, Thrust for parallel combinatorial enumeration; pattern: stoichiometric matrix resident on GPU → one thread block per genetic perturbation combination → parallel FBA solve → objective value reduction → top strains ranked. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
