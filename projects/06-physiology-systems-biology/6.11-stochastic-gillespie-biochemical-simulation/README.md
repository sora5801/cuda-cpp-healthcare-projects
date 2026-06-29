# 6.11 — Stochastic (Gillespie) Biochemical Simulation

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟢 Beginner · Established** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.11`
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

The Gillespie Stochastic Simulation Algorithm (SSA) exactly samples the master equation for discrete molecular counts in a well-mixed chemical reaction network—critical when molecule numbers are small (transcription factors, signaling molecules). Each stochastic trajectory is independent, so GPU parallelism maps one trajectory per thread. With 1 000–10 000 trajectories needed for statistics, GPU batch SSA achieves orders-of-magnitude speedup. Tau-leaping approximations (binomial/Poisson) trade exactness for speed at higher copy numbers.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Gillespie SSA (direct method), Gibson-Bruck next-reaction method, tau-leaping (explicit/implicit/binomial), R-leaping, chemical Langevin equation (CLE), reaction-diffusion master equation (RDME) for spatial stochastic simulation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/stochastic-gillespie-biochemical-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/stochastic-gillespie-biochemical-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\stochastic-gillespie-biochemical-simulation.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: BioModels Database — curated stochastic SBML models (https://www.ebi.ac.uk/biomodels); NIST Chemical Kinetics Database (https://kinetics.nist.gov); single-molecule tracking datasets on DANDI (https://dandiarchive.org); smFISH gene expression data (various GEO deposits at https://www.ncbi.nlm.nih.gov/geo/).

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

GillesPy2 (https://github.com/GillesPy2/GillesPy2) — Python SSA + tau-leaping + CLE, GPU backend in progress; StochPy (https://github.com/SystemsBioinformatics/stochpy) — Python stochastic simulation with SSA and tau-leaping; cuTauLeaping (verify URL — CUDA tau-leaping reference implementations in CUDA samples literature); MOOSE (https://github.com/BhallaLab/moose-core) — compartmental stochastic kinetic simulations.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuRAND for per-trajectory random exponential/uniform variates (one cuRAND stream per thread); CUDA Thrust for propensity prefix-sum (direct-method reaction selection); pattern: one CUDA thread per trajectory, independent RNG state in registers; atomic operations avoided by design (each thread is fully independent). --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
