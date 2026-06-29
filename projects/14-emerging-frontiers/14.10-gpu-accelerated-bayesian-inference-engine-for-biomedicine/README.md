# 14.10 — GPU-Accelerated Bayesian Inference Engine for Biomedicine

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Emerging%2C%20Theoretical%20%26%20Grand--Challenge%20Frontiers-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 14: Emerging, Theoretical & Grand-Challenge Frontiers · Catalog ID `14.10`
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

Bayesian inference over high-dimensional biomedical models (pharmacokinetic, genetic, epidemiological) requires Markov chain Monte Carlo (MCMC) or variational inference (VI) that is historically slow. GPU-accelerated Hamiltonian Monte Carlo (HMC/NUTS) in NumPyro or PyMC-JAX achieves 10–100× speedup over CPU Stan, enabling inference in population PKPD models with 10⁴ parameters and >10⁶ observations. GPU batch parallelism runs independent MCMC chains simultaneously, and GPU-accelerated gradients via JAX/autograd make HMC feasible for complex ODEs. Clinical trial simulation (tens of thousands of virtual patients) is a key use case.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Hamiltonian Monte Carlo (HMC) + No-U-Turn Sampler (NUTS), variational inference (ADVI, normalizing flows), sequential Monte Carlo (SMC), population PKPD (NONMEM-equivalent), Gaussian process inference, integrated nested Laplace approximation (INLA).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/gpu-accelerated-bayesian-inference-engine-for-biomedicine.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/gpu-accelerated-bayesian-inference-engine-for-biomedicine.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\gpu-accelerated-bayesian-inference-engine-for-biomedicine.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: NONMEM Pharmacokinetic Reference Dataset (Holford NHG, verify URL); UK Biobank phenome-wide association studies (https://www.ukbiobank.ac.uk/); OpenFDA Drug Adverse Event database (https://open.fda.gov/apis/drug/event/); CDISC SDTM clinical trial datasets (verify URL via cdisc.org).

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

NumPyro (https://github.com/pyro-ppl/numpyro) — GPU HMC/NUTS via JAX; PyMC (https://github.com/pymc-devs/pymc) — probabilistic programming with JAX/GPU backend; BlackJAX (https://github.com/blackjax-devs/blackjax) — GPU MCMC kernels in JAX; Stan (https://github.com/stan-dev/stan) — reference Bayesian inference (CPU; GPU via GPU-compatible backend research).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

JAX XLA GPU compilation for HMC gradient computation, cuBLAS for covariance matrix operations in GP inference, cuFFT for spectral MCMC methods; pattern: prior + likelihood specification in NumPyro → GPU JIT-compiled HMC kernel → parallel chains on GPU → posterior diagnostics (R-hat, ESS) → posterior predictive check. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
