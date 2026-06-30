# THEORY — 2.20 Heterogeneous Cryo-EM Reconstruction (3D Variability)

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

This project ships the **linear** flavor of heterogeneous reconstruction —
**3D Variability Analysis (3DVA)**, as in cryoSPARC — implemented as **PCA over a
set of 3-D density maps** on the GPU. It is a *reduced-scope teaching version* of
the catalog entry (the full tool, cryoDRGN, is a deep neural network; §7 explains
the difference). Everything below builds up to the five GPU stages in
`src/kernels.cu` and the verification in `src/main.cu`.

---

## 1. The science

A cryo-EM experiment flash-freezes many copies of a protein in a thin film of ice
and images them with an electron microscope. Each image is a noisy 2-D projection
of one molecule in **whatever shape it happened to be frozen in**. A rigid protein
gives many views of one structure; classical reconstruction averages them into a
single 3-D density map.

But most interesting molecules are **not** rigid. A ribosome ratchets, a GPCR
breathes, a spike protein opens. Freeze a million of them and you have frozen a
*continuum* of conformations. Averaging them all into one map smears the moving
parts into mush. **Heterogeneous reconstruction** asks the harder question:

> Given the particle images, can we recover not one structure but the *manifold of
> structures* — the molecule's motion?

3DVA answers the *linear* version: it models the conformational variability as a
small number of **principal motions** (linear deformation modes) about the mean
structure, and assigns every particle a coordinate along each motion. Plotting
those coordinates reveals the trajectory — e.g. "the ribosome's small subunit
rotates by X degrees as we move along principal component 1".

Our synthetic stand-in keeps the *structure* of the problem and strips the
imaging: one Gaussian density blob that **slides along the z-axis**. Particle `p`
has a hidden coordinate `t[p] ∈ [−1,+1]`; larger `t` ⇒ blob further along z. The
science question becomes: *from the volumes alone, can we rediscover the slide?*

---

## 2. The math

**Setup.** We are given `N` reconstructed volumes. Volume `p` is a `G×G×G` density
cube flattened to a vector **xₚ ∈ ℝᴰ**, with `D = G³`. Stack them as the rows of a
data matrix **X ∈ ℝ^{N×D}**.

**Centering.** Let the mean volume be
```
μ = (1/N) Σₚ xₚ          (μ ∈ ℝᴰ)
```
and the **centered** data matrix **Xc** with rows `xₚ − μ`. PCA explains how
volumes *deviate* from the mean, so all that follows uses Xc.

**Covariance.** The (population) covariance of the volumes is the `D×D` matrix
```
C = (1/N) Xcᵀ Xc          (C ∈ ℝ^{D×D}, symmetric, PSD)
```
Its eigenvectors **u₁, u₂, …** (unit, `D`-dimensional) are the **principal
components** — the dominant *shapes of variation*. Its eigenvalues `λ₁ ≥ λ₂ ≥ …`
are the **variances** explained along each. "PC1 explains fraction `f` of the
variance" means `f = λ₁ / Σ λ`.

**Latent coordinates.** Particle `p`'s position along mode `u` is the projection
```
zₚ = ⟨xₚ − μ, u⟩ = Σᵥ (xₚ,ᵥ − μᵥ) uᵥ
```
This single number per particle is the 3DVA output you plot.

**The snapshot / eigenfaces trick (the crux).** `D = G³` is huge, so forming and
diagonalizing the `D×D` matrix `C` is infeasible at scale. But there are only `N`
volumes, so `C` has rank ≤ `N−1`. Define the **small** `N×N` Gram matrix
```
G̃ = (1/N) Xc Xcᵀ          (G̃ ∈ ℝ^{N×N}, symmetric, PSD)
```
**Key identity:** `C = (1/N)Xcᵀ Xc` and `G̃ = (1/N)Xc Xcᵀ` share the same nonzero
eigenvalues, and their eigenvectors are linked. If `G̃ w = λ w` with `‖w‖ = 1` and
`λ > 0`, then
```
u = Xcᵀ w / ‖Xcᵀ w‖
```
is the unit eigenvector of `C` with the **same** eigenvalue `λ`. *Proof sketch:*
`C (Xcᵀw) = (1/N)Xcᵀ(Xc Xcᵀ)w = Xcᵀ G̃ w = λ (Xcᵀw)`, so `Xcᵀw` is an eigenvector
of `C` with eigenvalue `λ`; normalize it. So we diagonalize the **tiny `N×N`**
matrix and *lift* its eigenvectors back to volume space. That is the entire game.

Symbols: `N` particles, `G` grid edge, `D=G³` voxels, `X/Xc` raw/centered data
(`N×D`), `μ` mean volume (`ℝᴰ`), `C` covariance (`D×D`), `G̃` Gram (`N×N`),
`λₖ`/`uₖ` eigenvalue/PC, `wₖ` Gram eigenvector (`ℝᴺ`), `zₚ` latent coordinate.

---

## 3. The algorithm

```
Input : X (N volumes, D = G^3 voxels each)
1. μ        = (1/N) Σ_p x_p                         # mean volume      O(N·D)
2. G̃[i][j]  = (1/N) Σ_v (x_i,v - μ_v)(x_j,v - μ_v)   # N×N Gram matrix  O(N^2·D)
3. (λ, W)   = eig(G̃)                                # symmetric eig    O(N^3)
4. u_1      = Xc^T w_1 / ‖Xc^T w_1‖                  # lift top PC      O(N·D)
5. z_p      = ⟨x_p - μ, u_1⟩  for all p              # latent coords    O(N·D)
Output: λ (variances), u_1 (PC1 volume), z (per-particle latent), f = λ_1/Σλ
```

**Complexity.** The dominant term is **step 2**, the Gram matrix:
`O(N²·D)` — `~N²/2` dot products of length `D`. Step 3, the eigensolve, is only
`O(N³)` and `N ≪ D`, so it is cheap *as long as we use the snapshot trick*
(diagonalizing the `D×D` covariance directly would be `O(D³)` — catastrophic).
Steps 1, 4, 5 are each `O(N·D)`. Arithmetic intensity is high in step 2 (each
voxel pair feeds a multiply-add), which is why the GPU shines there.

**Serial vs parallel.** Serially this is five loops. In parallel, steps 1, 2, 4, 5
are *embarrassingly parallel* — every output (a voxel mean, a Gram entry, a PC
voxel, a latent) is independent, so the **parallel depth** is just the length of
the inner reduction (`O(N)` or `O(D)`), and the **work** matches the serial cost.
Step 3 is the only sequential-ish kernel; we delegate it to cuSOLVER.

---

## 4. The GPU mapping

Five stages, four are per-element kernels, one is a library call. The per-element
math lives in `reference_cpu.h` as `__host__ __device__` helpers so the CPU and
GPU compute identical values (PATTERNS.md §2).

| Stage | Kernel | Thread → data map | Reads / writes |
|---|---|---|---|
| 1 mean | `mean_kernel` | thread `v` ← voxel `v` | sums column `v` of X (N reads) → `μ[v]` |
| 2 Gram | `gram_kernel` | thread `(i,j)` ← entry `G̃[i][j]` | two centered rows (2·D reads) → `G̃[j*N+i]` |
| 3 eig | **cuSOLVER `Dsyevd`** | — (library) | `G̃` → eigenvalues + eigenvectors |
| 4 lift | `lift_kernel` | thread `v` ← voxel `v` of PC | N centered values · `w` → `u[v]` |
| 5 project | `project_kernel` | thread `p` ← particle `p` | row `p` · `u` (D reads) → `z[p]` |

**Launch configuration.** The 1-D kernels (mean, lift, project) use
`block = 256` threads (a warp multiple, enough warps to hide global-memory
latency on sm_75..sm_89) and `grid = ⌈len / 256⌉`, with the standard ragged-tail
guard `if (idx >= len) return;`. The Gram kernel uses a **2-D** grid:

```
            i (particle column) ──────────────►
        ┌───────────────────────────────────────┐
   j    │  block (16×16 threads)                 │   grid =
   (row)│   ┌────┬────┬────┐                     │   ( ⌈N/16⌉ , ⌈N/16⌉ )
        │   │    │    │    │   each thread (i,j)  │
        │   ├────┼────┼────┤   computes one       │   one thread per
        │   │    │    │    │   G̃[i][j] dot product │   matrix entry
        │   └────┴────┴────┘                      │
        └───────────────────────────────────────┘
```

**Memory hierarchy.** Everything lives in **global memory** (the volumes are the
big array). The mean and Gram kernels stream global memory; registers hold the
running accumulators. We deliberately keep the teaching version simple — *no*
shared memory — but the obvious optimization (Exercise 4) is to **tile** the Gram
dot product into shared memory so a block of particle rows is read from global
memory once and reused, which is exactly how a production GEMM-like covariance
build saves bandwidth. The eigenvectors come back in cuSOLVER's column-major
layout; since `G̃` is symmetric, its row-major and column-major forms are
identical, so we upload our row-major Gram with no transpose.

**The library call, not a black box (CLAUDE.md §6.1.6).** `cusolverDnDsyevd`
solves `A x = λ x` for a real symmetric `A` by **divide-and-conquer**:
tridiagonalize `A` with Householder reflectors, then recursively split the
tridiagonal eigenproblem and merge the pieces. With `jobz = VECTOR` it overwrites
`A` (our `d_gram`) with the orthonormal eigenvectors as **columns** and returns
the eigenvalues **ascending** in `d_W`. Hand-rolling this means coding
Householder tridiagonalization plus a stable QR or divide-and-conquer iteration
with deflation — hundreds of careful lines and a numerical-stability minefield —
which is precisely why we lean on cuSOLVER and merely *document* it. We pick PC1
as the **last** eigenvector (largest eigenvalue) and lift it.

---

## 5. Numerical considerations

- **Precision: FP64 throughout.** Covariance entries are sums of products of
  similar-magnitude densities; PCA on near-degenerate data is sensitive to
  rounding. Double precision keeps the Gram matrix accurate and lets CPU and GPU
  agree to ~1e-15. (Real 3DVA often runs FP32/FP16 for speed and accepts the
  noise; we choose clarity.)
- **Determinism — and why we avoid atomics.** Every kernel writes **distinct**
  outputs (voxel `v`, entry `(i,j)`, particle `p`), so there are **no atomics and
  no cross-thread reductions** — each accumulator is private to one thread and
  summed in a fixed order. Floating-point addition is not associative, so a
  reordered parallel sum would differ from the CPU in the last bits; by giving
  each output to exactly one thread we sidestep that entirely and get
  bit-reproducible results (PATTERNS.md §3).
- **The eigenvector sign ambiguity.** An eigenvector is defined only up to sign
  (`u` and `−u` are both valid PCs), and cuSOLVER vs Jacobi may return opposite
  signs. Left alone, that would make PC1, the latent coordinates, and the printed
  output non-deterministic. We impose a **canonical sign** (largest-magnitude
  voxel made positive) in *both* `lift_to_volume_pc` (CPU) and `run_3dva_gpu`
  (GPU), so the result is reproducible and the two paths match.
- **Stability of the snapshot trick.** It is exact for `λ > 0`. For a zero
  eigenvalue, `Xcᵀw = 0` and the lift would divide by zero; we guard the
  normalization (`if nrm > 0`). PC1 always has the largest, safely positive `λ`.

---

## 6. How we verify correctness

Three independent checks, two of them rigorous:

1. **GPU vs CPU agreement.** `reference_cpu.cpp` recomputes the *entire* 3DVA
   pipeline serially, using the **same** shared `__host__ __device__` helpers for
   the per-element math and a transparent **Jacobi** eigensolver for the
   eigenproblem. `main.cu` compares eigenvalues, PC1 voxels, and latent
   coordinates, requiring `max|Δ| ≤ 1e-9`. Why `1e-9` and not exact? The Gram
   build and projections are bit-identical (same math, same order), but
   **cuSOLVER's divide-and-conquer and our Jacobi sweep are different algorithms**
   — they reach the same eigenpairs to a few ulps, not to the bit (PATTERNS.md §4,
   "~machine precision for short double-precision work"). Observed worst case is
   ~7e-15, comfortably inside tolerance. Agreement between two *independent*
   implementations is strong evidence the result is right, not a shared bug.
2. **A known synthetic answer.** The data was generated with a hidden coordinate
   `t[p]` (the blob's z-slide). We report `|corr(z, t)|`, the correlation of the
   *recovered* latent with the *ground-truth* motion. It comes out ≈ **0.997** —
   3DVA rediscovered the motion it was never told about. This validates the
   *science*, not just CPU==GPU.
3. **Variance cross-check.** The CPU and GPU independently compute "PC1 variance
   explained" (`λ₁/Σλ`); the stderr line shows they match (0.8994 vs 0.8994), and
   the eigenvalue spectrum drops off a cliff (1.35 ≫ 0.14 ≫ 0.005) confirming the
   data really is ~rank-1, as designed.

---

## 7. Where this sits in the real world

This teaching version makes three big simplifications versus production
heterogeneous reconstruction:

- **Linear vs nonlinear.** 3DVA (and this project) models variability as a few
  **linear** modes about the mean — great for hinge/rotation motions that are
  approximately linear over their range, but it cannot represent a motion that
  curves through conformation space. **cryoDRGN** (the catalog's headline tool)
  replaces the linear PCA with a **variational autoencoder**: an image encoder maps
  each particle to a latent code, and a **coordinate-MLP decoder** (a NeRF-style
  implicit network) generates the density at any latent value. That learns a
  *nonlinear* manifold of structures. The catalog's "PyTorch / FlashAttention /
  differentiable nufft / cuFFT / FP16" notes all describe *that* pipeline.
- **Volumes vs images.** Real methods start from 2-D particle images, not 3-D
  volumes. They must jointly do **CTF correction**, **pose estimation** (each
  particle's unknown orientation, often by expectation-maximization), and relate
  image to volume through the **Fourier-slice theorem** (a 2-D projection equals a
  central slice of the 3-D Fourier transform — hence cuFFT/nufft in production).
  We assume reconstruction-to-volumes is already done and start from there.
- **Regularization & scale.** cryoSPARC 3DVA and **Recovar** add statistical
  regularization (the raw covariance is noisy at `N` ~ 10⁵ with `D` ~ 10⁶) and
  run on GPUs precisely because steps 2 and 4 are otherwise intractable. Our
  `N=24, D=216` demo is launch-bound; the GPU mapping here is the same one that,
  scaled up, makes the real computation feasible.

The throughline: **the linear-algebra core you can read in `kernels.cu` is the
conceptual skeleton inside the big tools** — center, covariance, eigen-decompose,
project. Master it here, then the VAE is "the same idea, but the modes are
learned and nonlinear".

---

## References

- **Punjani & Fleet, "3D Variability Analysis", J. Struct. Biol. 2021** — the
  cryoSPARC method this project mirrors; read for the regularized linear-subspace
  formulation.
- **Zhong et al., "CryoDRGN", Nature Methods 2021** (https://github.com/ml-struct-bio/cryodrgn)
  — the VAE + coordinate-MLP approach; the nonlinear successor.
- **Gilles & Singer, "Recovar"** (https://github.com/ma-gilles/recovar) — rigorous
  GPU regularized-covariance estimation; the statistics behind step 2.
- **Turk & Pentland, "Eigenfaces", 1991** — the original snapshot trick
  (`N×N` Gram instead of `D×D` covariance) that makes step 3 cheap; same math, a
  different field.
- **cuSOLVER docs — `Dsyevd`** — the divide-and-conquer symmetric eigensolver used
  in step 3; read the workspace-query + column-major eigenvector conventions.
- Flagship **2.06 (NMA / ENM)** in this repo — the sibling project that also leans
  on cuSOLVER for a symmetric eigenproblem; compare the two `Dsyevd` wrappers.
