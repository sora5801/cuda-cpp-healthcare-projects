# 13.17 — Neural-ODE & Physics-Informed Neural Networks for PK/PD

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Pharmacology%20%26%20Clinical%20Quantitative%20Modeling-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 13: Pharmacology & Clinical Quantitative Modeling · Catalog ID `13.17`
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

Replaces explicit pharmacokinetic ODEs with neural networks embedded within differential equations (Neural ODEs) or constrains neural architectures to satisfy ODE physics (Physics-Informed Neural Networks, PINNs). This allows learning latent pharmacokinetic dynamics from sparse clinical observations without specifying a mechanistic compartmental model. The GPU bottleneck is differentiating through the ODE solver (adjoint sensitivity method) for backpropagation, implemented in torchdiffeq. For PINNs, the collocation loss (residual of the ODE at sample points) is evaluated in batches on GPU. Recent Latent Neural-ODE approaches (arXiv:2602.03215) model-informed precision dosing with 15% fewer AEs than standard dosing.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Neural ODE (Chen et al. 2018), adjoint sensitivity for backprop through ODE, Physics-Informed Neural Networks (PINNs), Universal Differential Equations (UDEs), Latent ODE with VAE encoder, Gaussian process ODE priors, Fourier Neural Operators for PDE-based dosing, symbolic regression to recover interpretable ODE from data.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/neural-ode-physics-informed-neural-networks-for-pk-pd.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/neural-ode-physics-informed-neural-networks-for-pk-pd.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\neural-ode-physics-informed-neural-networks-for-pk-pd.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Latent Neural-ODE precision dosing dataset (https://arxiv.org/abs/2602.03215) — model-informed dosing with neural ODE MIMIC-IV ICU PK data — vancomycin/aminoglycoside time series (https://physionet.org/content/mimiciv/) Published population PK datasets (vancomycin, busulfan) from PharmPK listserv (verify URL) Synthetic NLME benchmark datasets from Monolix/NONMEM validation suites (verify URL)

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

torchdiffeq (https://github.com/rtqichen/torchdiffeq) — Neural ODE with GPU-accelerated adjoint sensitivity DiffEqFlux.jl (https://github.com/SciML/DiffEqFlux.jl) — Universal Differential Equations in Julia with GPU DeepXDE (https://github.com/lululxvi/deepxde) — GPU PINN framework for PDE/ODE-constrained learning SciMLBenchmarks (https://github.com/SciML/SciMLBenchmarks.jl) — benchmarks for neural ODE solvers

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

torchdiffeq adjoint ODE solver on GPU via PyTorch CUDA, cuBLAS for neural ODE network forward pass, JAX XLA for JIT-compiled PINN training; pattern: batched neural ODE integration with GPU-resident adjoint sensitivity gradients. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
