# 5.7 — Brachytherapy Dose & Source Modeling

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🟢 Beginner · Established** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.7`
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

Brachytherapy (BT) delivers dose from radioactive sources (Ir-192 HDR, Pd-103, I-125) implanted inside or adjacent to the tumor. TG-43 formalism computes dose analytically from tabulated radial and anisotropy functions per source dwell position; for an HDR plan with 50 dwell positions in a prostate implant, GPU parallelization across (source, voxel) pairs reduces plan calculation from seconds to milliseconds. Beyond TG-43, model-based dose algorithms (MBDCA) — Acuros BT, Monte Carlo — account for tissue heterogeneity and inter-source shielding, requiring the same GPU particle-transport infrastructure as external-beam MC. Real-time BT dose visualization on TRUS/fluoroscopy feed requires GPU latency <100 ms.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

TG-43 dose formalism (radial dose function, anisotropy function), superposition of point-source kernels, MBDCA (model-based dose calculation algorithm), MC for BT (Geant4-TOPAS, EGSnrc BrachyDose), shielding correction for multi-source, real-time dose overlay on TRUS imaging.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/brachytherapy-dose-source-modeling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/brachytherapy-dose-source-modeling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\brachytherapy-dose-source-modeling.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: AAPM TG-43 consensus datasets (radial/anisotropy tables — https://www.aapm.org/pubs/reports/); TCIA prostate BT CT datasets; ESTRO ACROP BT guideline test cases; BrachyView QA data (verify URL).

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

BrachyDose (via EGSnrc, https://github.com/nrc-cnrc/EGSnrc) — EGSnrc BT MC user code; TOPAS-BrachyDose (https://github.com/topasmc) — Geant4-based BT MC; PyTG43 (https://github.com/GregSal/PyTG43 — verify URL) — Python TG-43 dose calculator; matRad BT module (https://github.com/e0404/matRad) — MATLAB BT dose and optimization.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA kernel for TG-43 dose (grid of threads covering output voxels; inner loop over source dwell positions; tables in constant memory); cuRAND for MC BT photon sampling; texture memory for 2D anisotropy function tables; warp-level reduction for summing source contributions. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
