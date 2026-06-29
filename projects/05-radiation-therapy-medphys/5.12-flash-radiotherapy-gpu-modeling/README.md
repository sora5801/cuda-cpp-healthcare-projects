# 5.12 — FLASH Radiotherapy GPU Modeling

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.12`
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

FLASH-RT delivers doses at ultra-high dose rates (>40 Gy/s, typically >10⁴ Gy/s for electrons, >100 Gy/s for protons) in millisecond pulses, sparing normal tissue while maintaining tumor control. Modeling the FLASH effect requires coupled radiation-chemistry simulation: (1) GPU MC particle transport to compute local dose deposition patterns, (2) GPU track-structure to generate initial radical (OH•, H₂O₂, e⁻ₐq) distributions, and (3) GPU diffusion-reaction kinetics to simulate oxygen depletion and radical recombination in tissue. The MPEXS2.1-DNA code implements GPU water radiolysis under UHDR. Biological effect modeling requires stochastic ODE integration over microscopic reaction networks — a GPU-parallel task across millions of spatial positions.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

GPU MC particle transport at UHDR pulse structure, water radiolysis reaction-diffusion (Gillespie SSA on GPU), oxygen depletion kinetics, stochastic diffusion-reaction (MPEXS2.1-DNA), LET-dependent radical yield models, oxygen enhancement ratio (OER) map computation, pulse-by-pulse dose accumulation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/flash-radiotherapy-gpu-modeling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/flash-radiotherapy-gpu-modeling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\flash-radiotherapy-gpu-modeling.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: FLASH-RT experimental dosimetry from CERN/CLEAR, UCLouvain, Stanford FLASH programs (verify access); AAPM FLASH-RT working group benchmark datasets (verify URL); published oxygen tension measurements in tumors; GEANT4-DNA radiolysis validation datasets.

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

MPEXS2.1-DNA (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12551771/ — verify GitHub URL from paper) — GPU water radiolysis for UHDR; GATE 10 (https://github.com/OpenGATE/opengate) — FLASH macro-dose MC; TOPAS (https://github.com/OpenTOPAS/OpenTOPAS) — FLASH dosimetry extensions; Geant4-DNA (https://github.com/Geant4/geant4) — micro-kinetics for FLASH effect modeling.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA diffusion-reaction kernel (per-spatial-voxel Gillespie SSA, one thread block per µm³ tissue voxel); cuRAND for stochastic reaction channel selection; shared memory for local species concentration array; CUDA streams for pipelining pulse-by-pulse dose transport and chemistry; atomic ops for species count updates. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
