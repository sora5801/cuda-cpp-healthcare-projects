# 2.12 — Flexible Fitting / MDFF

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.12`
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

Molecular Dynamics Flexible Fitting (MDFF) fits an atomic model into a cryo-EM density map by adding density-derived forces to GPU MD, deforming the model to match the experimental map. This hybrid approach uses the GPU MD engine (NAMD or OpenMM) to handle sterics and covalent geometry while the density map acts as an external potential. GPU acceleration enables rapid convergence of the fitting for large complexes (ribosomes, viral capsids). Applications include fitting into sub-5 Å cryo-EM maps and interpreting conformational states.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

MDFF density-derived forces (cross-correlation gradient), EMFIT potential, real-space refinement, phenix.real_space_refine, morphing between states, flexible backbone MDFF.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/flexible-fitting-mdff.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/flexible-fitting-mdff.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\flexible-fitting-mdff.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: EMDB reference maps for MDFF (https://www.ebi.ac.uk/emdb/); EMPIAR raw particle data (https://www.ebi.ac.uk/empiar/); ribosome MDFF benchmarks (PDB 3J7Y, 4V6X); viral capsid fitting datasets.

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

NAMD MDFF (https://www.ks.uiuc.edu/Research/namd/) — production flexible fitting with CUDA; VMD MDFF plugin (https://www.ks.uiuc.edu/Research/vmd/) — MDFF setup and visualization; phenix.real_space_refine (https://phenix-online.org) — GPU-accelerated real-space refinement; Coot (https://www2.mrc-lmb.cam.ac.uk/personal/pemsley/coot/) — interactive model building into density.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Full GPU MD (NAMD CUDA) with additional density-gradient force kernel; cuFFT for cross-correlation computation in reciprocal space; GPU-parallel evaluation of density at atom positions via trilinear interpolation CUDA kernel. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
