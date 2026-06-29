# 5.13 — BNCT Dose Calculation & Optimization

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.13`
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

Boron Neutron Capture Therapy (BNCT) delivers therapeutic dose by targeting tumor cells loaded with ¹⁰B, which captures thermal neutrons to release high-LET alpha particles and lithium recoils. Dose calculation involves: (1) neutron transport (diffusion or discrete ordinates / Monte Carlo) to compute thermal neutron flux maps, (2) boron dose from ¹⁰B(n,α)⁷Li reaction rates, (3) high-LET photon dose, and (4) fast neutron dose — each requiring separate cross-section libraries and requiring GPU-parallel transport. The compound biological effectiveness (CBE) factor and boron uptake heterogeneity add biological modeling complexity. Treatment planning must jointly optimize beam direction and boron carrier dosing.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Monte Carlo neutron transport (OpenMC, MCNP, GATE), discrete ordinates neutron transport (Sₙ), multi-group cross-section library (ENDF/B-VIII), boron dose kernel convolution, CBE-weighted biological dose, neutron activation analysis on GPU, joint boron+neutron beam optimization.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/bnct-dose-calculation-optimization.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/bnct-dose-calculation-optimization.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\bnct-dose-calculation-optimization.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: IAEA BNCT benchmark cases (verify URL at iaea.org); BNCT clinical trial CT data from Finnish accelerator BNCT program; OpenMC validation datasets (https://github.com/openmc-dev/openmc/tree/develop/tests); NIST neutron cross-section data (verify URL).

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

OpenMC (https://github.com/openmc-dev/openmc) — open-source GPU-capable neutron MC (OpenMP/GPU via OpenMP target offload); GATE 10 (https://github.com/OpenGATE/opengate) — neutron transport for BNCT; COMPASS BNCT MC (verified in Nature Scientific Reports, https://pmc.ncbi.nlm.nih.gov/articles/PMC10366114/); OpenMC MeVisLab BNCT pipeline (https://www.hplpb.com.cn/en/article/doi/10.11884/HPLPB202537.250246).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

GPU neutron transport via OpenMP offload or custom CUDA kernel (one thread per neutron history); material cross-section tables in texture memory; boron concentration map in 3D GPU array; cuBLAS for multi-group matrix-vector flux equations; warp-divergence mitigation by material-sorted particle batches. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
