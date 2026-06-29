# 5.4 — Collapsed-Cone / Superposition-Convolution Dose

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🟢 Beginner · Established** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.4`
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

Superposition-convolution (SC) dose computation convolves Monte Carlo-derived photon energy-deposition kernels (polyenergetic dose-spread arrays, DSAs) with the TERMA (total energy released per unit mass) computed from CT. Collapsed-cone convolution (CCC) discretizes the kernel into angular cones and propagates dose along ray paths at each angle. For a 512³ CT volume and ~400 cone directions, each cone sweep is a 1D scan along the CT in that direction — embarrassingly parallel across cones and voxels. GPU parallelization across cone directions and voxel planes reduces a CCC plan from ~10 min to <10 s. This algorithm underlies most commercial photon dose engines (Eclipse AXB, RayStation).

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Superposition/convolution with poly-energetic DSA kernels, collapsed-cone convolution (CCC), anisotropic analytical algorithm (AAA), Acuros XB (linear Boltzmann transport), TERMA ray-tracing (Siddon/ray-voxel), heterogeneity correction via density scaling.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/collapsed-cone-superposition-convolution-dose.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/collapsed-cone-superposition-convolution-dose.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\collapsed-cone-superposition-convolution-dose.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: AAPM TG-105 test cases (heterogeneous media dose benchmarks); IROC lung phantom CT + dosimetry data; TCIA clinical photon planning datasets; CIRS IMRT verification phantom data.

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

matRad (https://github.com/e0404/matRad) — photon pencil-beam + CC dose engine; Plastimatch (https://plastimatch.org/) — GPU-accelerated dose engine components; CERR (https://github.com/cerr/CERR) — dose calculation framework; open AAPM TG-105 reference datasets with comparison code (verify URL at aapm.org).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA kernel for TERMA ray-trace (Siddon's algorithm, one thread per ray); cone-direction parallel sweep in CCC (one CUDA block per cone direction); shared memory for density strip along current cone ray; reduction for energy normalization. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
