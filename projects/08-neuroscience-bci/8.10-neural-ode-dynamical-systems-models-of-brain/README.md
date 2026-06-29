# 8.10 — Neural ODE / Dynamical Systems Models of Brain

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Neuroscience%20%26%20Brain--Computer%20Interfaces-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 8: Neuroscience & Brain-Computer Interfaces · Catalog ID `8.10`
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

Neural ODEs parameterize the time derivative of hidden neural state as a neural network, enabling continuous-time models of brain dynamics that can be fit to irregular-interval neural recordings and extrapolated to unseen time points. Applied to whole-brain fMRI or calcium imaging, they learn latent dynamical manifolds underlying cognition. Adjoint sensitivity (checkpointed backpropagation through the ODE solver) is memory-intensive and GPU-critical; the adjoint method requires storing only a constant number of activations regardless of integration depth.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Neural ODE (Runge-Kutta adjoint), augmented neural ODE (ANODE), latent ODE / SDE (VAE + neural ODE), flow matching, score-based generative modeling for neural trajectories, continuous normalizing flow (CNF), Gaussian process ODE, reservoir computing (echo state networks).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/neural-ode-dynamical-systems-models-of-brain.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/neural-ode-dynamical-systems-models-of-brain.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\neural-ode-dynamical-systems-models-of-brain.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Human Connectome Project resting-state fMRI (https://db.humanconnectome.org); DANDI electrophysiology (https://dandiarchive.org); Allen Brain Observatory calcium imaging (https://portal.brain-map.org); NLB Neural Latents Benchmark (https://neurallatents.github.io).

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

torchdiffeq (https://github.com/rtqichen/torchdiffeq) — GPU neural ODE with adjoint backpropagation; torchsde (https://github.com/google-research/torchsde) — stochastic differential equation neural models on GPU; LFADS (https://github.com/google-research/google-research/tree/master/lfads) — RNN-based latent factor analysis; Diffrax (https://github.com/patrick-kidger/diffrax) — JAX-based GPU ODE/SDE solver suite.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN for neural network RHS evaluation; checkpointed adjoint via custom CUDA memory management; cuRAND for SDE noise sampling; pattern: time-reversed adjoint integration with activations recomputed on-the-fly, CUDA graph for repeated-pattern ODE step. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
