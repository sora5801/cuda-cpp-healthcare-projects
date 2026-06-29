# 1.16 — ADMET / Toxicity Prediction

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.16`
>
> _Educational only — not for clinical use (see CLAUDE.md §8). This is a
> **reduced-scope teaching version** (CLAUDE.md §13): a linear multi-task model
> stands in for the production graph neural network, which is described but not
> trained here._

## Summary

Before a drug candidate ever reaches a patient it has to survive a gauntlet of
**ADMET** screens — Absorption, Distribution, Metabolism, Excretion, and
Toxicity. Predicting those properties *computationally*, early, lets chemists
throw out doomed molecules before spending years and millions on them. This
project screens a batch of candidate molecules against many toxicity endpoints
at once: each molecule is a fixed-length numeric **descriptor**, each endpoint
is a small trained classifier, and we predict the whole **molecules × endpoints**
matrix of toxicity probabilities on the GPU — one independent prediction per
thread. It then reduces that matrix to the triage a chemist actually wants: how
many molecules each endpoint flags, and which single molecule looks worst.

## What this computes & why the GPU helps

Absorption, Distribution, Metabolism, Excretion, and Toxicity (ADMET) properties
gate entry into clinical trials; predicting them computationally early in
discovery eliminates costly failures. GPU-trained GNN/MPNN models (Chemprop-based)
can screen 100M virtual compounds for ADMET in hours; the ADMET-AI platform (2024)
uses a Chemprop-RDKit ensemble for best-in-class speed. **Multi-task** learning on
heterogeneous assays (LogP, hERG, Caco-2, microsomal clearance, Ames mutagenicity)
benefits from GPU parallelism across tasks **and** molecules simultaneously.

**The parallel bottleneck:** the **N×M prediction matrix**. Screening libraries
are huge (10⁶–10⁹ molecules) and each must be scored against every endpoint
(M ≈ 12). Every cell `p_{i,t}` depends only on molecule *i*'s descriptor and
endpoint *t*'s model — there are **no dependencies between cells**, so the whole
matrix is embarrassingly parallel: we give each `(molecule, endpoint)` cell its
own GPU thread. The endpoint models are tiny and read by every thread, so they
live in **constant memory** (broadcast warp-wide), exactly the pattern used by the
1.12 Tanimoto flagship.

## The algorithm in brief

- **Descriptor → logit:** for molecule *i*, endpoint *t*, compute the linear score
  `z = b_t + Σ_d w_{t,d}·x_{i,d}` (a dot product + bias).
- **Logit → probability:** `p = sigmoid(z) = 1/(1+e^{-z})` (numerically stable
  two-branch form).
- **Multi-task:** repeat for all M endpoints — that is the "multi-task" head, M
  logistic regressions sharing the same input descriptor.
- **Deterministic reduction:** threshold each `p` at 0.5 into a 0/1 flag; count
  flags per endpoint with **integer atomics** (reproducible); pick the worst
  molecule by a deterministic argmax.
- The catalog's full method (D-MPNN message passing, conformal/evidential
  uncertainty, Tox21 endpoint models) is explained in `THEORY.md` under
  "Where this sits in the real world".

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/admet-toxicity-prediction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/admet-toxicity-prediction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\admet-toxicity-prediction.sln /p:Configuration=Release /p:Platform=x64
```

This project links only the CUDA runtime (`cudart_static.lib`) — no extra CUDA
library is needed (the kernel is a hand-written dot-product, deliberately, so
nothing is a black box).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if the CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the per-endpoint flag
counts and the worst molecule, shows the GPU-vs-CPU agreement check, and prints a
timing line.

## Data

- **Sample (committed):** `data/sample/admet_sample.txt` — 24 synthetic molecules
  × 12 endpoints (64-dim descriptors), so the demo runs with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print where to get the
  real public ADMET sets and how to featurize them.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: Tox21 — 12 toxicity endpoints, 8k compounds
(https://tripod.nih.gov/tox21/); TDC ADMET benchmark group
(https://tdcommons.ai/benchmark/admet_group/overview/); ClinTox — FDA-approved and
failed drugs (https://moleculenet.org); DILI databases.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program computes the toxicity-probability matrix on the **GPU**
(`src/kernels.cu`) and on a **CPU reference** (`src/reference_cpu.cpp`) and
asserts they agree: the probabilities within `1e-9` (they actually match to
~`5e-16`, machine precision, because both call the same `__host__ __device__`
math in `src/admet_core.h`) and the integer per-endpoint flag counts **exactly**.
That agreement is the correctness guarantee. In the sample, per-endpoint flag
rates span 5/24–16/24 and the planted toxic molecule `MOL_0000` tops the ranking.

## Code tour

Read in this order:

1. [`src/admet_core.h`](src/admet_core.h) — the shared `__host__ __device__` math
   (dot product, sigmoid, threshold). The heart of CPU/GPU parity.
2. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the two kernels (predict, flag-count) and
   the host wrapper; constant memory + integer atomics.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline
   and the text loader.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **Chemprop** (<https://github.com/chemprop/chemprop>) — the D-MPNN backbone used
  by most modern ADMET models; study how directed message passing builds a learned
  molecular vector that *replaces* the fixed descriptor we use here.
- **ADMET-AI** (<https://github.com/swansonk14/admet_ai>) — a GPU-accelerated ADMET
  platform (Chemprop-RDKit ensemble); study its multi-endpoint output and speed.
- **DeepChem** (<https://github.com/deepchem/deepchem>) — includes Tox21 models and
  many featurizers; a good place to see descriptors and splits done properly.
- **pkCSM** (<https://biosig.lab.uq.edu.au/pkcsm/>) — graph-signature ADMET
  predictor (web server); study the "graph signature" featurization idea.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent jobs + constant-memory models** (PATTERNS.md §1, exemplar 1.12):
one thread per `(molecule, endpoint)` cell over a flat 1-D grid with a
grid-stride loop; the M endpoint models live in `__constant__` memory so the
constant cache broadcasts them warp-wide; the reduction to per-endpoint flag
counts uses **integer `atomicAdd`** for determinism (PATTERNS.md §3). The
production stack the catalog names (PyTorch Geometric sparse ops, cuDNN, FP16,
multi-task loss aggregation) is discussed in `THEORY.md` "real world".

## Exercises

1. **Scale it.** Run `python scripts/make_synthetic.py --n 1000000` and watch the
   stderr timing — where does the GPU start to clearly beat the CPU, and why is it
   launch/copy-bound on the tiny sample?
2. **Top-K worst molecules.** Extend `main.cu` to report the *K* most toxic
   molecules (like 1.12's top-K) instead of just the single worst.
3. **Tile the descriptor in shared memory.** For larger D, cooperatively load
   molecule *i*'s descriptor into shared memory once per block instead of each
   thread re-reading it from global memory; measure the effect.
4. **Add an uncertainty column.** Implement a simple conformal-prediction style
   margin (`|p − 0.5|`) and flag "abstain" cells where the model is unsure — a
   first taste of the uncertainty quantification the catalog mentions.
5. **FP32 vs FP64.** Switch the shared math to `float` and observe how the
   CPU-vs-GPU `max_abs_err` grows (and why the tolerance would have to loosen).

## Limitations & honesty

- **Reduced scope.** The real method is a **trained graph neural network** (D-MPNN)
  over molecular graphs with learned features and uncertainty estimates. Here the
  model is a **linear logistic regression per endpoint over a fixed descriptor** —
  the classical baseline GNNs are benchmarked against. We teach the *GPU mapping*
  (the N×M independent-prediction screen), not the learning.
- **Synthetic everything.** The descriptors, weights, biases, and endpoint names
  are randomly generated and labeled synthetic. The probabilities and "toxicity"
  flags are **chemically meaningless**.
- **No training.** We do not fit the models; weights come pre-set from the data
  file. Real ADMET needs careful train/validation/test splits (scaffold splits!)
  and assay-specific calibration.
- **Not a tool.** Nothing here may inform any real chemical, safety, or clinical
  decision. It is study material for the CUDA pattern only.
