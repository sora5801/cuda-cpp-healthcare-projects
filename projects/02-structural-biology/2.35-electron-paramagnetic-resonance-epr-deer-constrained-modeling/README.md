# 2.35 — Electron Paramagnetic Resonance (EPR/DEER) Constrained Modeling

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.35`
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

DEER (Double Electron-Electron Resonance) distance measurements between spin labels constrain the conformational ensemble of flexible proteins and membrane proteins in their native membrane environment. GPU-accelerated MD restrained by DEER distance distributions enables ensemble refinement of proteins that cannot be crystallized. The GPU compute pattern parallelize over hundreds of independent MD replicas, each evaluated against DEER restraints (population-weighted distance distribution comparison). Applications include ABC transporter gating, GPCR dynamics, and IDR backbone sampling.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

DEER distance distribution back-calculation from MD ensemble, maximum entropy ensemble reweighting (EROS/BioEn), rotamer library convolution for spin-label placement (MTSSL), GPU MD with soft DEER restraints, population re-weighting.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/electron-paramagnetic-resonance-epr-deer-constrained-modeling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/electron-paramagnetic-resonance-epr-deer-constrained-modeling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\electron-paramagnetic-resonance-epr-deer-constrained-modeling.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: SASBDB EPR-constrained structures (verify URL); published DEER datasets for membrane transporters; EPR.cxls community datasets (verify URL); PDB structures refined with EPR data.

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

MMM (Multiscale Modeling of Macromolecules, https://www.epr.ethz.ch/software/mmm.html) — EPR-driven ensemble modeling; DEER-PREdict (verify URL) — DEER distance prediction from MD; EnsembleFit/BioEn (https://github.com/bio-phys/BioEN) — GPU Bayesian ensemble reweighting; OpenMM DEER restraints (https://github.com/openmm/openmm) — soft distance restraints from DEER.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

GPU MD array for ensemble members; CUDA DEER back-calculation kernel (rotamer convolution over N spin-label positions); GPU population reweighting via maximum entropy; multi-GPU replica ensemble with shared experimental target. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
