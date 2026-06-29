# 2.24 — SAXS / SANS Data-Driven Structure Modeling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.24`
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

Small-angle X-ray/neutron scattering (SAXS/SANS) provides solution-phase structural information about proteins and complexes as a 1D intensity profile I(q). Fitting atomic or CG models to SAXS data requires rapid forward calculation of the scattering intensity from 3D coordinates via Debye formula or spherical harmonic expansion — a pairwise summation over all atoms that is GPU-parallelizable. GPU-MD + SAXS ensemble refinement (EROS, BioEn) samples thousands of conformers and reweights to match experimental SAXS. Applications include intrinsically disordered protein (IDP) ensemble characterization.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Debye scattering formula (O(N²) GPU-parallel), CRYSOL implicit solvent scattering model, spherical harmonic expansion for SAXS, SAXS-restrained MD ensemble refinement (EROS/BioEn), maximum entropy reweighting, atomistic vs CG SAXS prediction.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/saxs-sans-data-driven-structure-modeling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/saxs-sans-data-driven-structure-modeling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\saxs-sans-data-driven-structure-modeling.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: SASBDB — small-angle scattering biological data bank (https://www.sasbdb.org); PDB-SAXS depositions (https://www.rcsb.org); BIOISIS benchmark (verify URL); simulated SAXS from MD trajectories.

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

CRYSOL (https://www.embl-hamburg.de/biosaxs/crysol.html) — analytical SAXS computation; FOXS (https://modbase.compbio.ucsf.edu/foxs/) — fast SAXS fitting; WAXSiS (verify URL) — GPU-accelerated wide-angle scattering; MDAnalysis SAXS module (https://github.com/MDAnalysis/mdanalysis) — trajectory SAXS averaging.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA kernel for O(N²) Debye summation over atom pairs; GPU partial sum reduction for form factors; cuBLAS for spherical harmonic coefficients; GPU-parallel ensemble member scoring. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
