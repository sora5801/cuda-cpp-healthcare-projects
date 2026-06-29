# 5.9 — Gamma-Index Dose Comparison

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🟢 Beginner · Established** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.9`
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

The gamma index (γ) at each reference point searches for the minimum normalized Euclidean distance in combined dose-difference / distance-to-agreement (DTA) space over all evaluated points: γ(r_ref) = min_r √[(Δd/Δd_crit)² + (Δr/DTA_crit)²]. For 3D clinical distributions at 2 mm DTA and 2% dose criterion, the exhaustive search over a 200³ evaluation grid from each of 200³ reference points is O(N⁶) naively, reduced to O(N³ × K) by limiting the search radius. GPU parallelizes this: one thread per reference point, searches a kernel of neighbor evaluated points; with shared-memory tiling this achieves 100–1,000× speedup over CPU, enabling sub-second 3D gamma on clinical GPUs. This is critical for patient-specific IMRT/VMAT pre-treatment verification.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

3D gamma index exhaustive search (distance-limited), fast gamma approximations (1D cross-plane), GPU kernel tiling for shared-memory neighbour caching, global gamma pass-rate statistics, normalized agreement testing (NAT), χ (chi) factor dose comparison.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/gamma-index-dose-comparison.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/gamma-index-dose-comparison.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\gamma-index-dose-comparison.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: AAPM TG-218 patient-specific IMRT QA reference data; plan+measurement DICOM pairs from departmental QA systems; IROC-Houston phantom dose datasets; linac EPID measurement datasets.

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

Pymedphys (https://github.com/pymedphys/pymedphys) — Python gamma index, DICOM dose tools; Plastimatch (https://plastimatch.org/) — GPU gamma-index C++ library; gamma-index GPU (https://pubmed.ncbi.nlm.nih.gov/21317484/ — verify GitHub from paper) — UCSD GPU gamma; OpenGATE (https://github.com/OpenGATE/opengate) — includes dose comparison utilities.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

One CUDA thread per reference point; shared-memory tile of evaluated dose grid (tiled by distance radius); minimum reduction in registers; atomic min for tie-breaking; cuBLAS for vectorized pass/fail statistics across patient cohort. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
