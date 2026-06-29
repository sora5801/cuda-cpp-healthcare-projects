# 13.13 — QT-Prolongation & Cardiac Safety Risk Assessment

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Pharmacology%20%26%20Clinical%20Quantitative%20Modeling-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 13: Pharmacology & Clinical Quantitative Modeling · Catalog ID `13.13`
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

Predicts drug-induced QT interval prolongation — a surrogate for fatal arrhythmia (Torsade de Pointes) — from drug structure, hERG channel IC50 measurements, and clinical ECG data. The CardioGenAI framework uses GPU-accelerated molecular graph neural networks to predict hERG block and re-engineer drug structures for reduced liability. Clinical ECG-based deep learning (3DRECON-QT) reconstructs 3D spatial QTc from single-lead recordings using CNN on GPU. Mechanistic cardiac action potential models (O'Hara-Rudy, Paci human iPSC-CM) simulate drug effects on ion channels at thousands of drug concentrations simultaneously on GPU — each simulation is an ODE stiff system on the 40+ state Hodgkin-Huxley-type action potential model.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

GNN-based hERG IC50 prediction from SMILES, 3DRECON-QT spatial reconstruction, O'Hara-Rudy action potential ODE, voltage-clamp state machine (Markov model for hERG), torsade de pointes risk classification (TdP risk categories), dynamic clamp simulation on GPU, QTc Fridericia/Bazett correction.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/qt-prolongation-cardiac-safety-risk-assessment.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/qt-prolongation-cardiac-safety-risk-assessment.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\qt-prolongation-cardiac-safety-risk-assessment.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: CiPA (Comprehensive in vitro Pro-arrhythmia Assay) ion channel datasets — multi-channel IC50 for 28 reference drugs (verify URL via FDA) hERGCentral database — hERG patch-clamp measurements (verify URL) MIMIC-IV-ECG — clinical QTc measurements linked to medication data (https://physionet.org/content/mimic-iv-ecg/) CardioNet ECG database (verify URL) — large annotated ECG dataset for QT analysis

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

CardioGenAI (https://github.com/mgreenig/CardioGenAI) — ML framework for re-engineering drugs for reduced hERG liability myokit (https://github.com/myokit/myokit) — cardiac action potential ODE modelling; GPU via CUDA backend OpenCARP (https://opencarp.org/) — cardiac electrophysiology simulator with GPU support DeepHERG (verify URL) — deep learning hERG inhibition prediction

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

DGL for hERG GNN, custom CUDA Hodgkin-Huxley ODE kernels for action potential batch simulation, cuRAND for Monte Carlo drug concentration sweeps; pattern: one CUDA thread per drug concentration × cell simulation, with shared memory for ion channel state variables. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
