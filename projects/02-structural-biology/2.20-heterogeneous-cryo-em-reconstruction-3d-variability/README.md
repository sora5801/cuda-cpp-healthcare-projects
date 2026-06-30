# 2.20 — Heterogeneous Cryo-EM Reconstruction (3D Variability)

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.20`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A protein imaged by cryo-EM is rarely frozen in one shape — it flexes, and each
of the (hundreds of thousands of) particle images captures a slightly different
conformation. **Heterogeneous reconstruction** tries to recover that motion, not
just one average structure. This project implements the *linear* flavor of it,
**3D Variability Analysis (3DVA)** — the method cryoSPARC uses — as **Principal
Component Analysis (PCA) on a set of 3-D density maps**, accelerated on the GPU.
We compute the covariance of the volumes, find its top eigenvectors with
**cuSOLVER**, and read off (a) how one-dimensional the flexibility is and (b)
each particle's coordinate along the dominant motion. On a synthetic dataset with
a known "sliding blob" motion, 3DVA recovers that motion with correlation ≈ 0.997.

> This is a deliberately **reduced-scope teaching version**. The full research
> tool (cryoDRGN) replaces the linear PCA with a deep variational autoencoder;
> see [THEORY.md](THEORY.md) → "Where this sits in the real world".

## What this computes & why the GPU helps

Real protein complexes adopt multiple conformational states simultaneously.
Heterogeneous reconstruction methods disentangle these states from particle
images. CryoDRGN uses a variational autoencoder (VAE) with an amortized encoder
that maps each particle image to a latent code representing its conformation, and
a decoder that generates the 3D density from the latent code via a coordinate
MLP. GPU training is essential: a cryoDRGN run on 100k particles requires hours on
A100. **3DVA (cryoSPARC) uses PCA-like linear subspace methods** — and that is the
version implemented here. Applications reveal continuous flexibility in ribosomes,
GPCR complexes, and viral assembly intermediates.

**The parallel bottleneck:** PCA on volumes is dominated by two steps, both
embarrassingly parallel. (1) Building the covariance: with `N` particle volumes
of `D = G³` voxels each, the `N × N` Gram matrix needs ~`N²/2` independent dot
products of length `D` — one GPU thread per matrix entry. (2) Lifting an
eigenvector back to a full volume-space principal component is `D` independent
voxel sums — one thread per voxel. The small `N × N` eigendecomposition in
between is handed to **cuSOLVER** (`Dsyevd`). For real data (`N` ~ 10⁵, `D` ~ 10⁶)
these steps are enormous, which is exactly why 3DVA runs on the GPU.

## The algorithm in brief

- **Mean volume** — the average density map; subtract it to center the data.
- **Covariance via the snapshot (eigenfaces) trick** — diagonalizing the huge
  `D × D` covariance is replaced by the small `N × N` Gram matrix `(1/N) XXᵀ`,
  whose nonzero spectrum is identical.
- **Eigendecomposition (PCA)** — eigenvalues = per-mode variances; eigenvectors =
  the conformational modes. Done with **cuSOLVER `Dsyevd`** (divide & conquer).
- **Lift to volume space** — turn the top Gram eigenvector `w` into the actual
  principal component `u = Xᵀw / ‖Xᵀw‖`, a density map you could open in ChimeraX.
- **Latent coordinates** — project each centered volume onto `u`: one number per
  particle, its position along the dominant motion.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation, including why the snapshot trick is exact and how we verify it.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)). This project links
**cuSOLVER** (`cusolver.lib` + its `cublas`/`cusparse` deps; already wired into
both the `.vcxproj` and `CMakeLists.txt`).

1. Open `build/heterogeneous-cryo-em-reconstruction-3d-variability.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/heterogeneous-cryo-em-reconstruction-3d-variability.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\heterogeneous-cryo-em-reconstruction-3d-variability.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (uses the optional CMake build)
```

The demo builds if needed, runs on `data/sample/volumes.txt`, prints the 3DVA
result, shows the GPU-vs-CPU agreement check, and prints a per-stage timing line.

## Data

- **Sample (committed):** `data/sample/volumes.txt` — 24 **synthetic** particle
  volumes (`G=6`, `D=216`) hiding one continuous "z-slide" motion, so the demo
  runs offline with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print pointers to the real
  EMPIAR / cryoDRGN datasets (they download nothing; the real sets are large and
  need preprocessing — this is a reduced-scope teaching project).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: EMPIAR-10180 (spliceosome), EMPIAR-10076 (80S ribosome),
EMPIAR-10028 (TRPV1) (all at https://www.ebi.ac.uk/empiar/); cryoDRGN benchmark
datasets (https://github.com/ml-struct-bio/cryodrgn); simulated heterogeneous
datasets from IgG/spike protein.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
2.20 -- Heterogeneous Cryo-EM Reconstruction (3D Variability)
3DVA (PCA on volumes): N=24 particles, G=6, D=216 voxels/volume
top-3 mode variances (eigenvalues, descending): 1.35169 0.14476 0.00472
PC1 variance explained: 0.8994
latent z along PC1 (8 sampled particles): +1.7639 +1.4561 +1.0194 +0.4876 -0.2887 -0.8507 -1.3289 -1.7556
PC1 vs ground-truth conformation |corr|: 0.9974
RESULT: PASS (GPU 3DVA matches CPU reference within tol)
```

The program computes 3DVA on both the **GPU** (`src/kernels.cu`, cuSOLVER) and a
**CPU reference** (`src/reference_cpu.cpp`, a transparent Jacobi eigensolver) and
asserts they agree to ~machine precision (eigenvalues, PC1, latent coordinates
all within `1e-9`). That agreement is the correctness guarantee. The `|corr|`
line is a second, *scientific* check: PC1 recovers the synthetic motion.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads volumes, runs CPU + GPU 3DVA, verifies, reports.
2. [`src/reference_cpu.h`](src/reference_cpu.h) — the data model **and** the shared
   `__host__ __device__` math (centering, Gram dot product, projection).
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the CPU 3DVA + Jacobi eigensolver.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the two-pattern idea.
5. [`src/kernels.cu`](src/kernels.cu) — the per-element kernels and the cuSOLVER call.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **cryoSPARC 3DVA** (https://cryosparc.com) — the production *linear* method this
  project mirrors: PCA over a volume subspace. Study its regularization choices.
- **cryoDRGN** (https://github.com/ml-struct-bio/cryodrgn) — the GPU VAE that
  replaces linear PCA with a nonlinear latent + coordinate-MLP decoder. The model
  to graduate to once the linear picture clicks.
- **Recovar** (https://github.com/ma-gilles/recovar) — GPU regularized-covariance
  heterogeneous reconstruction; the rigorous statistical take on the covariance
  used here.
- **DrgnAI / drgnai** (https://github.com/ml-struct-bio/drgnai) — neural
  reconstruction with joint pose optimization.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Two complementary GPU patterns, both in the [PATTERNS.md](../../../docs/PATTERNS.md)
cookbook:

- **Per-element kernels** (PATTERNS.md §1, "score one item vs many"): a 2-D grid,
  one thread per `(i,j)` Gram entry; one thread per voxel for the mean and the
  PC lift; one thread per particle for the projections. The per-element physics is
  shared with the CPU via a `__host__ __device__` core (PATTERNS.md §2).
- **A dense linear-algebra library** (PATTERNS.md §5): the `N × N` symmetric
  eigenproblem is delegated to **cuSOLVER `Dsyevd`**, documented (not a black
  box) exactly as flagship `2.06` does.

(The catalog's "PyTorch / FlashAttention / nufft / cuFFT / FP16" notes describe
the full *nonlinear* cryoDRGN pipeline — out of scope for this linear teaching
version, and discussed in THEORY.)

## Exercises

1. **Find PC2.** PC1 captures ~90% of the variance; extract the second
   eigenvector and project onto it. For this dataset PC2 should be mostly noise —
   confirm its variance is tiny and its correlation with the ground truth is ~0.
2. **Two motions.** Edit `make_synthetic.py` to add a *second* independent motion
   (e.g. the blob also brightens). Re-run: does the variance now split across two
   PCs? Does each PC recover one motion?
3. **Bigger problem.** Run `python scripts/make_synthetic.py --n 64 --g 8` and
   watch the GPU's `gram`/`lift` stage times grow relative to the (still tiny)
   eigensolve — the launch-bound regime giving way to compute-bound.
4. **Shared-memory Gram.** The `gram_kernel` re-reads both centered rows from
   global memory. Tile the dot product into shared memory (PATTERNS.md §1) and
   measure the speed-up at larger `D`.
5. **Reconstruct a state.** Pick a latent value `z*`; render `mean + z*·PC1` and
   compare it to the synthetic volume whose `t[p]` is closest. This is how 3DVA
   produces a movie of the motion.

## Limitations & honesty

- **Reduced scope.** This is the *linear* 3DVA, not cryoDRGN's nonlinear VAE. It
  cannot represent motions that are not a linear combination of a few modes
  (THEORY explains exactly where linearity breaks).
- **We start from volumes, not images.** Real heterogeneous reconstruction must
  also do CTF correction, pose estimation, and per-particle back-projection from
  2-D images (the Fourier-slice theorem). We assume those are already done and
  begin from `N` reconstructed volumes.
- **Synthetic data.** The committed sample is fabricated (a Gaussian blob sliding
  in z) and is labeled synthetic everywhere. It demonstrates the method; it is
  not a real protein and carries no scientific or clinical validity.
- **Tiny sizes.** `N=24`, `D=216` make the demo instant and the GPU timing
  launch-bound. The GPU advantage is real only at production scale (`N`~10⁵,
  `D`~10⁶); timings here are a teaching artifact, never a benchmark claim.
