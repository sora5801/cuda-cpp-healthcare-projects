# 5.3 — Proton & Heavy-Ion Therapy Dose

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.3`
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

Proton and carbon-ion beams deposit dose with a sharp Bragg peak distal to the target, enabling sparing of surrounding normal tissue. Analytical dose engines (pencil-beam algorithm, PBA) convolve pencil-beam kernels with CT stopping-power maps; GPU parallelizes the per-spot convolution across the ~10⁴ spots in a plan, reducing a full plan from minutes to seconds. Full Monte Carlo (FRED, TOPAS, GATE) simulates hadronic physics including nuclear fragmentation (dominant for carbon ions), requiring GPU for clinical throughput. Range uncertainty (due to CT Hounsfield-unit–to–stopping-power conversion) is managed by robust optimization over 3 mm / 3.5% scenarios, multiplying GPU compute requirements.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Pencil-beam algorithm (PBA), analytical Bragg-peak model, GPU MC (FRED, MOQUI, gPMC), nuclear fragmentation transport (Geant4-TOPAS), LET (linear energy transfer) calculation, RBE (relative biological effectiveness) weighting, multi-field optimization, robust proton optimization.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/proton-heavy-ion-therapy-dose.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/proton-heavy-ion-therapy-dose.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\proton-heavy-ion-therapy-dose.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: TOPAS/GATE benchmark proton beam data; clinical proton CT datasets (develop via institution); TCIA proton treatment response datasets; POPI model for proton treatment planning (https://www.creatis.insa-lyon.fr/rio/popi-model).

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

FRED (https://www.fredonline.eu/) — GPU fast MC for ions, clinical-grade, DICOM-RT input; MOQUI (https://github.com/mghro/moquimc) — GPU proton MC for quick dose recalculation (MGH, open source); OpenTOPAS (https://github.com/OpenTOPAS/OpenTOPAS) — open fork of TOPAS, Geant4-based proton MC; matRad (https://github.com/e0404/matRad) — analytic proton dose engine with GPU-parallel spot convolution.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA for per-spot pencil-beam convolution (one thread per spot × voxel pair); cuFFT for convolution in k-space; texture memory for CT stopping-power map; cuRAND for MC sampling; CUDA atomic adds for dose histogram accumulation. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
