# 5.8 — Linac QA & Machine Performance Assessment

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🟢 Beginner · Established** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.8`
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

Linear accelerator (linac) quality assurance measures beam output, flatness, symmetry, and MLC leaf positions from portal dosimetry images or log files. GPU acceleration is applied in three areas: (1) rapid gamma-index computation comparing measured vs. planned dose distributions (3D gamma on a 200³ dose grid requires ~10⁹ distance searches), (2) EPID (electronic portal imaging device) image-based dose reconstruction converting 2D portal images to 3D dose via a GPU MC kernel, and (3) machine learning prediction of machine failures from large log-file datasets (training on GPU). Automated daily QA with immediate GPU-based analysis enables real-time feedback before the treatment session.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Gamma-index dose comparison (3D, distance-to-agreement + dose-difference), EPID portal dose reconstruction (MC kernel convolution on GPU), MLC leaf-gap analysis, Winston-Lutz test automation, trajectory log analysis, ML anomaly detection on linac logs.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/linac-qa-machine-performance-assessment.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/linac-qa-machine-performance-assessment.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\linac-qa-machine-performance-assessment.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: AAPM TG-119 IMRT QA test cases; AAPM TG-218 tolerance criteria datasets; TCIA linac log datasets (verify URL); Varian/Elekta log file datasets from published QA studies; OpenMedPhys (https://github.com/jrkerns/awesome-medphys) reference datasets.

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

Pylinac (https://github.com/jrkerns/pylinac) — Python linac QA automation (image analysis, log files); PRIMO MC linac simulator (https://www.primoproject.net/ — verify URL); Plastimatch (https://plastimatch.org/) — GPU-accelerated gamma index; matRad (https://github.com/e0404/matRad) — plan-vs-measurement comparison.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA for 3D gamma index (each thread manages one reference-dose point, searches neighbor distance sphere in delivered dose volume); texture memory for delivered dose field; cuBLAS for log-file ML feature matrix; warp-level min-reduction for closest distance search. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
