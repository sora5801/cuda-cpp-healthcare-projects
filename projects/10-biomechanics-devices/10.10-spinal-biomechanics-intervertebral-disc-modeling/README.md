# 10.10 — Spinal Biomechanics & Intervertebral Disc Modeling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biomechanics%2C%20Biomedical%20Devices%20%26%20Surgery-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 10: Biomechanics, Biomedical Devices & Surgery · Catalog ID `10.10`
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

The lumbar spine involves poroelastic disc mechanics, facet-joint contact, and large deformation under combined flexion-compression loads, requiring multi-physics FEA with >500 K DOF per motion segment. GPU parallelism compresses the 97.9% time-reduction already demonstrated in automated MRI-to-FEM pipelines (Frontiers 2024) by further accelerating the PCG solver for the full lumbar assembly. Population virtual trials — evaluating thousands of patient-specific spinal constructs after fusion surgery — run overnight on GPU clusters, replacing months of cadaveric testing. GPU-resident bone-density maps updated with DXA-calibrated HU values enable patient-specific fracture risk prediction on clinical timescales.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Biphasic/poroelastic FEM (Mow-Holmes disc model), hyperelastic anulus fibrosus (fiber-reinforced), penalty facet-joint contact, bone-remodeling, automated mesh generation (Laplacian smoothing + decimation), shape correspondence via non-rigid ICP.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/spinal-biomechanics-intervertebral-disc-modeling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/spinal-biomechanics-intervertebral-disc-modeling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\spinal-biomechanics-intervertebral-disc-modeling.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: VerSe Challenge — 374 CT scans with vertebral shape annotation (https://verse-challenge.github.io/); MICCAI SpineSeg — lumbar MRI segmentation (verify URL via Grand Challenge); CT Spine Dataset (verse2020, Zenodo) — 355 CTs with vertebral instance masks (https://doi.org/10.5281/zenodo.3755323); OrthoLoad Lumbar — in vivo spinal implant forces (https://orthoload.com/).

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

FEBio (https://github.com/febiosoftware/FEBio) — built-in biphasic and fiber-reinforced disc models; SpineWeb toolkit — vertebral mesh atlas (http://spineweb.digitalimaginggroup.ca/); TotalSegmentator (https://github.com/wasserth/TotalSegmentator) — fast CT organ+vertebra segmentation for mesh input; MRI-to-FEM pipeline (Frontiers 2024, verify Zenodo for code) — automated lumbar FE model generation.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuSPARSE PCG for multi-physics coupled system, CUDA kernels for fiber-reinforced anisotropic stress update, cuDNN for DXA HU calibration regression; pattern: GPU-resident CT density map → automatic mesh generation → fiber orientation interpolation on GPU → coupled solid-fluid PCG solve → fracture risk post-processing. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
