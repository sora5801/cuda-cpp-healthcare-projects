# 5.11 — Microdosimetry & Track-Structure Simulation

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.11`
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

Microdosimetry and nanodosimetry characterize the stochastic distribution of energy deposition events in microscopic volumes (µm–nm scale), relevant for predicting DNA damage and biological effectiveness. Track-structure codes (Geant4-DNA, MPEXS-DNA) simulate every electron interaction step-by-step, requiring liquid water cross-sections down to sub-eV energies; a single proton track produces ~10⁵ secondary interactions. GPU parallelization across simultaneous primary particle tracks (one thread per track) achieves 50–70× speedup. Applications include carbon-ion RBE calculation, targeted radionuclide dosimetry (alpha emitters), and predicting clustered DNA damage yields from mixed radiation fields.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Event-by-event track structure (Geant4-DNA cross-sections), step-by-step condensed random walk, DNA damage scoring (DSB, SSB, base damage), diffusion-reaction chemistry simulation (radiolysis), nanodosimeter simulation, LET spectrum calculation, biological effectiveness prediction.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/microdosimetry-track-structure-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/microdosimetry-track-structure-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\microdosimetry-track-structure-simulation.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Geant4-DNA physics validation data (https://geant4-dna.in2p3.fr/); NIST electron stopping powers (https://www.nist.gov/pml/estar); AAPM/NCRP microdosimetry benchmark datasets; published DNA damage yield datasets from radiobiology experiments.

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

Geant4-DNA (https://geant4-dna.in2p3.fr/ — part of Geant4, https://github.com/Geant4/geant4) — standard track-structure code; MPEXS-DNA (CUDA GPU version, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6850505/ — verify GitHub) — GPU microdosimetry and radiolysis; TOPAS-nBio (https://github.com/topas-nbio/TOPAS-nBio) — nano-biological extension of TOPAS; PARTRAC (verify URL) — track structure specialized for DNA damage.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA per-track simulation (one warp per track, reaction lookup in constant memory); divergence minimized by sorting tracks by interaction type before step; cuRAND Philox generator for per-track random sequences; atomic adds to DNA damage histogram; shared memory for cross-section table of current material step. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
