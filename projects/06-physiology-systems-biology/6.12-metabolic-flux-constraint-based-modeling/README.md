# 6.12 — Metabolic Flux / Constraint-Based Modeling

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟢 Beginner · Established** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.12`
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

Flux balance analysis (FBA) finds optimal metabolic fluxes by solving a linear program (LP) constrained by stoichiometry, thermodynamics, and enzyme capacity on genome-scale metabolic models (GEMs) with 3 000–8 000 reactions. GPU parallelism enters through solving thousands of LP instances in parallel (e.g., for all conditions in a drug screen, or all single-gene knockouts in an essentiality screen). Mixed-integer programming (MILP) variants for gap-filling and thermodynamic FBA benefit from GPU-accelerated interior-point methods.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Flux balance analysis (FBA), flux variability analysis (FVA), parsimonious FBA (pFBA), thermodynamic FBA (tFBA), MILP gap-filling, minimal cut sets, COBRA toolbox algorithms, interior-point LP (revised simplex), shadow price / sensitivity analysis.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/metabolic-flux-constraint-based-modeling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/metabolic-flux-constraint-based-modeling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\metabolic-flux-constraint-based-modeling.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Recon3D — human genome-scale metabolic model (https://github.com/SBRG/Recon3D); HMDB — Human Metabolome Database (https://hmdb.ca); Reactome (https://reactome.org); BiGG Models Database — curated GEMs (http://bigg.ucsd.edu).

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

COBRApy (https://github.com/opencobra/cobrapy) — Python FBA/FVA with multiple LP/MILP solver backends; Recon3D model files (https://github.com/SBRG/Recon3D); Virtual Metabolic Human (https://vmh.life) — interactive Recon3D portal; SUNDIALS (https://github.com/LLNL/sundials) — for dynamic FBA ODE integration.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuSOLVER dense LP factor (batch small LP); custom CUDA interior-point primal-dual kernel for LP batches; ArrayFire (https://github.com/arrayfire/arrayfire) for dense matrix batches; pattern: one LP per CUDA block, shared memory for constraint matrix, warp-level reduction for objective gradient. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
