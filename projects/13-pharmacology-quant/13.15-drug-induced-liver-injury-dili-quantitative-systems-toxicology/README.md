# 13.15 — Drug-Induced Liver Injury (DILI) & Quantitative Systems Toxicology

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Pharmacology%20%26%20Clinical%20Quantitative%20Modeling-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 13: Pharmacology & Clinical Quantitative Modeling · Catalog ID `13.15`
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

Predicts and mechanistically explains drug-induced liver injury using multi-scale QST models (DILIsym) that integrate intracellular mitochondrial function, bile acid synthesis/transport, oxidative stress, and innate immune response with drug concentration-dependent perturbations. The stiff ODE system (300+ equations for intracellular biochemistry × hepatocyte populations × liver zonation) requires GPU-parallel stiff integration for virtual patient simulations. Graph convolutional networks on drug molecular graphs (BioGL-GCN) trained on hepatotoxicity labels enable rapid screening of new compounds. Combining GCN screening with mechanistic QST validation on GPU covers both speed and interpretability.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

QST ODE integration (CVODE, RODAS4), mitochondrial membrane potential dynamics ODEs, bile acid transport ODE system, NF-κB signalling cascade, GCN/GNN on molecular graphs for hepatotoxicity classification, random forest + physicochemical feature DILI prediction, multiscale coupling of PBPK with intracellular QST.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/drug-induced-liver-injury-dili-quantitative-systems-toxicology.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/drug-induced-liver-injury-dili-quantitative-systems-toxicology.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\drug-induced-liver-injury-dili-quantitative-systems-toxicology.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: DILIst — curated DILI positive/negative drug list (verify URL; NCATS) LiverTox — NIH database of drug-induced liver disease (https://www.ncbi.nlm.nih.gov/books/NBK547852/) Tox21 — 12,000+ compounds with hepatotoxicity assay data (https://tox21.gov/) DILIsym virtual patient database (Simulations Plus) — calibrated virtual liver population (verify URL)

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

DILIsym (https://www.simulations-plus.com/software/dilisym/) — commercial QST DILI platform (Simulations Plus) BioGL-GCN (verify URL) — graph convolutional network for DILI prediction from drug structures DeepTox (https://github.com/bioinf-jku/tox21_networks) — deep learning Tox21 prediction baseline nvQSP (https://github.com/NVIDIA-Digital-Bio/nvQSP) — GPU stiff ODE solver for QST models

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA CVODE/RODAS4 stiff ODE kernels for QST integration, DGL for hepatotoxicity GCN, cuBLAS for bile acid flux Jacobians; pattern: virtual patient batch — one CUDA block per patient, intracellular biochemistry compartments in shared memory. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
