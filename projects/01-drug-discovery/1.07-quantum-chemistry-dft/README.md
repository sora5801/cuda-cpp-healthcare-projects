# 1.7 — Quantum Chemistry / DFT

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.7`
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

Density Functional Theory (DFT) calculates electronic structure by solving the Kohn-Sham equations self-consistently on a basis set (plane waves or Gaussians). The dominant cost is the construction of the Fock/Kohn-Sham matrix via electron repulsion integrals (ERIs) — an O(N^4) bottleneck that GPUs reduce substantially by computing integrals in batches. TeraChem pioneered GPU-accelerated DFT and can achieve 100× speedup over single-CPU codes. Applications in drug discovery include geometry optimization of drug fragments, calculation of electrostatic potential maps for pharmacophore generation, and QM-derived force field parameterization.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Kohn-Sham SCF, B3LYP/ωB97X-D exchange-correlation functionals, resolution-of-identity (RI) approximation for ERIs, DIIS convergence acceleration, plane-wave pseudopotential (PW-PP), linear-scaling DFT.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/quantum-chemistry-dft.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/quantum-chemistry-dft.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\quantum-chemistry-dft.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: QM9 — DFT-computed properties of 134k organic molecules (https://doi.org/10.6084/m9.figshare.978904); ANI-1ccx — CCSD(T)-level energies for diverse organic molecules (https://github.com/isayev/ANI1ccx_dataset); PubChemQC — DFT calculations for ~3M PubChem molecules (http://pubchemqc.riken.jp); CSD — Cambridge Structural Database for crystal structures (https://www.ccdc.cam.ac.uk).

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

TeraChem (https://www.petachem.com) — GPU-native DFT, commercial but widely cited; PySCF (https://github.com/pyscf/pyscf) — pure Python quantum chemistry with GPU4PySCF extension; CP2K (https://github.com/cp2k/cp2k) — GPU-accelerated mixed Gaussian/plane-wave DFT; NWChem (https://github.com/nwchemgit/nwchem) — parallel quantum chemistry with GPU-accelerated modules.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA kernels for ERI computation (two-electron integrals in shared memory); cuBLAS for matrix diagonalization; cuFFT for plane-wave FFT; warp-level parallelism over shell pairs. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
