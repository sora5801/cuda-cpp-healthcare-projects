# 2.06 — Normal Mode Analysis / Elastic Network Models

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟢 Beginner · Established** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.06`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Predict a protein's large-scale **collective motions** — domain hinges, breathing
modes — with **Normal Mode Analysis** on an **Elastic Network Model**. Represent
the protein as Cα atoms joined by springs within a cutoff, build the 3N×3N
**Hessian**, and **diagonalize** it: the eigenvectors are the modes, the
eigenvalues their frequencies, and the lowest non-zero modes are the functional
ones. This flagship is built around a **CUDA library** — **cuSOLVER** does the
eigendecomposition — a pattern distinct from every other flagship's hand-written
kernels.

## What this computes & why the GPU helps

NMA's bottleneck is diagonalizing the dense Hessian — an `O(N³)` symmetric
eigenvalue problem, intractable on CPU for large proteins. Elastic Network Models
(ANM/GNM) simplify the matrix; the eigendecomposition is handed to **cuSOLVER**, a
GPU library, with CUDA-accelerated matrix algebra. Here we build the ANM Hessian
on the host (clear and small) and let cuSOLVER do the heavy `O(N³)` solve.

**The parallelized work** is the eigendecomposition (the cuSOLVER call); the
Hessian build and the mobility analysis are cheap host steps.

## The algorithm in brief

- **Hessian:** for Cα pairs within cutoff, the 3×3 block is `-(γ/d²)·ΔΔᵀ`; the
  diagonal blocks accumulate the negatives (the elastic-energy second derivative).
- **Diagonalize:** symmetric eigensolver → eigenvalues (squared frequencies) +
  eigenvectors (modes). The 6 smallest are ~0 (rigid-body translation/rotation).
- **Mobility:** `Σ_{modes} (1/λ)·|v at residue|²` predicts per-residue flexibility.

See [THEORY.md](THEORY.md) for the elastic model, the eigenproblem, and what cuSOLVER does.

## Build

Requires **Visual Studio 2026** (v145) + **CUDA 13.3** ([docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).
This project **links cuSOLVER** (`cusolver.lib` + `cublas.lib` + `cusparse.lib`, already in the `.vcxproj`).

1. Open `build/normal-mode-analysis-elastic-network-models.sln`.
2. **`Release|x64`** → **Build** → the `.exe` under `build/x64/Release/`.

CLI: `msbuild build\normal-mode-analysis-elastic-network-models.sln /p:Configuration=Release /p:Platform=x64`

## Run the demo

```powershell
./demo/run_demo.ps1
```

Diagonalizes the Hessian on the GPU + CPU and verifies the eigenvalues match.

## Data

- **Sample (committed):** `data/sample/protein_ca.txt` — 60 synthetic Cα atoms.
- **Real structures:** Cα coordinates from RCSB PDB / AlphaFold (via ProDy) — see
  `scripts/download_data.ps1` and [data/README.md](data/README.md).
- Bigger structure: `python scripts/make_synthetic.py --N 120`.

## Expected output

`demo/expected_output.txt` holds the deterministic modes + mobility. The Hessian
(180×180) yields exactly **6 zero modes**; cuSOLVER (`src/kernels.cu`) and the CPU
Jacobi reference (`src/reference_cpu.cpp`) agree on the eigenvalues to ~`1e-12`.

## Code tour

1. [`src/main.cu`](src/main.cu) — load, build Hessian, CPU Jacobi + GPU cuSOLVER, verify, mobility, print.
2. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — **ANM Hessian + Jacobi eigensolver + mobility**.
3. [`src/kernels.cuh`](src/kernels.cuh) — the cuSOLVER interface + the "library, not black box" note.
4. [`src/kernels.cu`](src/kernels.cu) — **the cuSOLVER `Dsyevd` call (fully documented)**.

## Prior art & further reading

- **ProDy** (<https://github.com/prody/ProDy>) — Python ANM/GNM (the de-facto NMA tool).
- **Bio3D** (<https://thegrantlab.org/bio3d/>) — NMA in R.
- **ElNemo** (<https://www.sciences.univ-nantes.fr/elnemo/>) — elastic network modes server.
- Bahar, Atilgan & Erman (1997) — the Gaussian Network Model; Atilgan et al. (2001) — the ANM.

Study these for the production approach; reimplement didactically (CLAUDE.md §2).

## CUDA pattern used here

**Dense linear algebra via a CUDA library**: build a symmetric matrix, call
**cuSOLVER `Dsyevd`** (divide-and-conquer symmetric eigensolver) · verify against a
transparent CPU Jacobi eigensolver · interpret the eigenvectors (modes) physically.

## Exercises

1. **GNM.** Build the N×N Kirchhoff (connectivity Laplacian) instead and compare
   its mobility prediction to the ANM's.
2. **Real protein.** Feed a PDB structure's Cα atoms and compare the predicted
   mobility to the crystallographic **B-factors** (they correlate).
3. **Mode visualization.** Animate a low-frequency mode (displace Cα along an
   eigenvector) and watch the domain motion.
4. **Sparse / Lanczos.** For large N, only the lowest modes are needed — use a
   sparse Hessian + a Lanczos iteration instead of a full dense solve.
5. **cuSOLVER vs the build.** Profile where time goes (Hessian build vs solve) as N
   grows; the O(N³) solve dominates.

## Limitations & honesty

- **ANM with a uniform spring constant** and a full dense solver; production NMA
  uses sequence/distance-dependent springs, sparse Hessians + Lanczos for large
  proteins, and all-atom force fields for true vibrational analysis.
- The Hessian is built on the host (the parallelized O(N³) work is the eigensolver);
  the structure is synthetic, so the numbers are illustrative.
