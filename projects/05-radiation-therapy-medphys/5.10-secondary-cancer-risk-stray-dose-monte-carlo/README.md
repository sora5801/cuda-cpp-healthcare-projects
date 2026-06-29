# 5.10 — Secondary Cancer Risk & Stray-Dose Monte Carlo

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.10`
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

Radiotherapy delivers dose not only to the target but also to distant organs via stray radiation (leakage, scatter, neutrons from proton therapy nuclear interactions), creating secondary cancer risk. Stray-dose is ~3–4 orders of magnitude lower than target dose, requiring 10¹¹–10¹²+ particle histories per calculation for statistical precision — intractable even on GPU without variance reduction (splitting, forced detection, geometry importance). GPU-based stray-dose MC requires importance sampling and photon-electron transport over the full body habitus beyond the treated field, rarely implemented in commercial systems. Secondary neutron fluence from proton therapy high-Z nozzle elements requires hadronic physics in Geant4/TOPAS, adding GPU parallelization complexity.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Forced detection variance reduction, splitting/Russian roulette, photonuclear interaction cross-sections, hadronic interaction model (INCL, BERT) for secondary neutrons, whole-body geometric phantom integration (ICRP110 voxel phantoms), Lifetime Risk Model (BEIR VII) convolution with dose distribution.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/secondary-cancer-risk-stray-dose-monte-carlo.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/secondary-cancer-risk-stray-dose-monte-carlo.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\secondary-cancer-risk-stray-dose-monte-carlo.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: ICRP 110 voxel phantoms (adult male/female, https://www.icrp.org/publication.asp?id=ICRP%20Publication%20110); NIST photon cross-section databases (https://www.nist.gov/pml/xcom-photon-cross-sections); secondary dose measurements from literature; TCIA proton therapy planning CTs.

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

TOPAS (https://github.com/OpenTOPAS/OpenTOPAS) — full hadronic transport, stray-dose extensions; GATE 10 (https://github.com/OpenGATE/opengate) — neutron transport, out-of-field dose scoring; EGSnrc (https://github.com/nrc-cnrc/EGSnrc) — photon/electron with advanced variance reduction; PHITS (https://phits.jaea.go.jp/ — verify URL) — hadronic + neutron transport for radiation protection.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA hadronic transport kernel (one thread per particle, nested interaction sampling loop); constant memory for cross-section tables; variance reduction handled per-thread (splitting → thread forking via particle stack on GPU); cuRAND for correlated sampling sequences. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
