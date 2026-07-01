# THEORY — 4.15 Diffusion MRI & Tractography

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

Water molecules in tissue are in constant thermal (Brownian) motion. In an open
glass of water that motion is **isotropic** — equally likely in every direction.
But inside brain **white matter**, water is hemmed in by tightly-packed, myelinated
axons (nerve fibers): it diffuses *freely along* the fiber bundles and is
*hindered across* them. Diffusion is therefore **anisotropic**, and — crucially —
the direction of easiest diffusion is the direction the fibers run.

**Diffusion MRI (dMRI)** measures this anisotropy non-invasively. By applying
magnetic-field *gradients* along many directions, an MRI scanner sensitizes the
signal to water motion along each direction: signal drops most where water moves
most. Fit a model to those directional signal drops and you recover, per voxel, a
picture of the local diffusion — and from it, the fiber orientation.

Two things follow, and they are the whole project:

1. **Diffusion Tensor Imaging (DTI):** model the local diffusion as a 3×3 tensor
   `D`. Its eigen-decomposition gives scalar maps used everywhere in
   neuroscience and neurology — **Fractional Anisotropy (FA)** and **Mean
   Diffusivity (MD)** — plus the principal fiber direction.
2. **Tractography:** follow the principal-direction field from voxel to voxel to
   reconstruct the white-matter **pathways** ("streamlines" / "tracts"), e.g. the
   corticospinal tract or the corpus callosum.

The real-world questions this answers: *Where do the major fiber bundles run?
Is a tumor displacing or infiltrating a tract? Is white matter degrading in
multiple sclerosis, or maturing in a developing brain?* (Educationally framed
only — this code makes **no** clinical claim.)

## 2. The math

**The signal model (Stejskal–Tanner).** For measurement `k` with gradient unit
direction `g_k = (gx, gy, gz)` and *b-value* `b_k` (s/mm², encoding gradient
strength and timing), the measured signal is

```
S_k = S0 · exp( − b_k · g_kᵀ D g_k )
```

- `S0` — the non-diffusion-weighted (b=0) signal (units: arbitrary intensity).
- `D` — the 3×3 **symmetric positive-definite diffusion tensor** (mm²/s), with 6
  unique entries `Dxx Dyy Dzz Dxy Dxz Dyz`.
- `g_kᵀ D g_k` — the *apparent diffusivity* along direction `g_k`.

**Linearising the fit.** Taking the natural log removes the exponential:

```
ln S_k = ln S0 − b_k ( gx² Dxx + gy² Dyy + gz² Dzz
                       + 2 gx gy Dxy + 2 gx gz Dxz + 2 gy gz Dyz )
```

This is **linear** in the 7 unknowns `θ = [ln S0, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]`.
Stacking all `NMEAS` measurements gives `y = B θ`, where `y_k = ln S_k` and row
`k` of the **design matrix** `B` (size `NMEAS × 7`) is

```
B_k = [ 1, −b_k gx², −b_k gy², −b_k gz², −2 b_k gx gy, −2 b_k gx gz, −2 b_k gy gz ].
```

With `NMEAS ≥ 7` this is over-determined; the **ordinary least-squares** solution is

```
θ = (BᵀB)⁻¹ Bᵀ y   ≡   M y ,      M = (BᵀB)⁻¹ Bᵀ  (size 7 × NMEAS).
```

Because the gradient scheme is the **same for every voxel**, `B` — and therefore
`M` — is computed **once** and reused for all N voxels. Per voxel we only do the
matrix-vector product `θ = M y`.

**Scalar maps from `D`.** Eigen-decompose `D` into eigenvalues `λ1 ≥ λ2 ≥ λ3`
(the principal diffusivities, mm²/s) and eigenvectors. Then

```
MD = (λ1 + λ2 + λ3) / 3                                   (Mean Diffusivity)
FA = sqrt(1/2) · sqrt[ (λ1−λ2)² + (λ2−λ3)² + (λ3−λ1)² ]
              / sqrt[ λ1² + λ2² + λ3² ]                    (Fractional Anisotropy)
```

FA ∈ [0, 1]: 0 = isotropic (sphere, e.g. CSF/gray matter), → 1 = strongly
directional (cigar, e.g. a dense fiber bundle). The eigenvector of `λ1` is the
**principal direction v1** — the local fiber orientation.

**Tractography.** Reconstruct a pathway by integrating the direction field:
`dr/ds = v1(r)`, starting from a seed. We use **deterministic** streamline
tractography with Euler steps and trilinear interpolation of `v1`, stopping when
FA drops below a threshold (left the white matter) or the path curves too sharply.

## 3. The algorithm

**Stage A — per-voxel DTI fit** (for each of N voxels, independently):

1. `y_k = ln(S_k)` for the `NMEAS` signals (floor at a tiny positive value).
2. `θ = M · y` — a fixed `7 × NMEAS` matrix-vector product; unpack `Dxx..Dyz`.
3. Eigen-decompose the 3×3 symmetric `D` **analytically** (Smith's 1961
   trigonometric formula for the cubic's three real roots), sort descending.
4. Compute FA, MD, and the principal eigenvector v1.

Cost per voxel: `O(NMEAS · 7)` for the matvec + `O(1)` for the 3×3 eigensolve →
**`O(N · NMEAS)`** total, embarrassingly parallel (no inter-voxel dependency).

**Stage B — deterministic tractography** (for each seed, independently): Euler-
integrate `dr/ds = v1(r)`, each step doing a trilinear interpolation of v1 over
the 8 neighbouring voxels (with eigenvector-sign alignment) plus FA/curvature
stop checks. Cost per streamline: `O(max_steps · 8)`; total `O(seeds · max_steps)`.

Why analytic eigensolve, not SVD/Jacobi? For a symmetric 3×3 the closed form is
branch-light, deterministic, and needs no library — perfect for one thread per
voxel. (See §4 for when you *would* reach for cuSOLVER.)

## 4. The GPU mapping

Both stages are the **"independent jobs"** pattern (PATTERNS.md §1; exemplar
`1.12` Tanimoto):

**Kernel 1 — `fit_kernel` (one thread per voxel).**
- **Thread → data:** thread `v = blockIdx.x·blockDim.x + threadIdx.x` fits voxel
  `v`; a grid-stride loop lets a fixed grid cover any N.
- **Launch:** 256 threads/block (a multiple of the 32-lane warp; 8 warps give the
  scheduler latency to hide), `min(ceil(N/256), 1024)` blocks.
- **Memory:** the OLS operator `M` (7×13 doubles = 728 B) lives in **constant
  memory** — every thread reads the *same* matrix and none writes it, so the
  constant cache **broadcasts** one address to a whole warp in a single
  transaction (vs. `7·13` redundant global loads per thread). The voxel's signals
  come from global memory; one `VoxelResult` is written out. **No shared memory,
  no atomics** — voxels are independent.

**Kernel 2 — `tract_kernel` (one thread per seed).**
- **Thread → data:** thread `s` traces seed `s` (forward + backward), writing its
  polyline into a fixed-size per-seed slot (no dynamic device allocation).
- **Per step:** a **trilinear interpolation** of the v1 field — a *gather* over 8
  neighbours. This is precisely what CUDA **texture memory** hardware does for
  free (`cudaAddressModeClamp` + linear filtering). We spell it out by hand here
  so the math is visible and verifiable; a production kernel would bind the
  direction field as a `cudaTextureObject_t` and let the texture units do the
  8-tap blend. That is the "texture memory for fODF interpolation" the catalog
  mentions.

```
         DWI volume (N voxels)                 seeds (few)
   ┌───────────────────────────┐        ┌───────────────────┐
   │ signals[v*NMEAS .. ]       │        │ (x,y,z) per seed  │
   └───────────────────────────┘        └───────────────────┘
        │ fit_kernel                          │ tract_kernel
        │ thread v → voxel v                  │ thread s → seed s
        │ M in __constant__ (broadcast)       │ trilerp v1 field (gather;
        ▼                                     ▼   texture-shaped)
   VoxelResult[v] {FA,MD,λ,v1}  ─────────────► Streamline[s] (polyline)
```

**Where a library would slot in (no black boxes):**
- **cuBLAS** — the catalog's CSD variant needs per-voxel spherical-harmonic
  matrix products (`GEMM`); our simpler DTI OLS is a *fixed 7×13 matvec*, so a
  hand-written unrolled loop (in `dti_core.h`) is both faster and clearer at this
  size. At whole-brain scale a batched `cublas<t>gemmBatched` would fit all
  voxels' OLS at once.
- **cuSOLVER** — for a *general or larger* symmetric eigenproblem you would call
  batched `syevj` (as flagship `1.08` does). For 3×3 the analytic form wins.
- **cuRAND** — **probabilistic** tractography (iFOD2) draws each step from the
  fiber orientation *distribution* using a per-thread RNG; we omit randomness on
  purpose so the demo is reproducible (§5).

## 5. Numerical considerations

**Precision.** All per-voxel math is **FP64 (double)**. The tensor eigenvalues are
~1e-3 and FA is a ratio of differences, so single precision would lose meaningful
digits in near-isotropic voxels. Double is cheap here (the kernel is memory- and
launch-bound, not FLOP-bound).

**Determinism.** stdout must be byte-identical every run, so:
- No atomics and no order-dependent reductions — each thread writes its own
  independent output, so there is nothing to reorder.
- Tractography uses **no RNG** (deterministic streamlines), unlike probabilistic
  iFOD2. The seed selection is a deterministic `partial_sort` (ties broken by
  lower voxel index).

**The degenerate-eigenvalue trap (a real lesson).** Smith's formula returns the
three roots via `acos`/`cos`. When two eigenvalues are (nearly) **degenerate** —
exactly the case for an axially-symmetric fiber tensor, whose two across-fiber
diffusivities are equal — a difference of ~1e-16 between the host and device
`acos`/`cos` implementations can **flip which root is labelled largest**. FA and
MD are *symmetric* in the eigenvalues so they stay identical, but `λ1/λ2/λ3`
would disagree between CPU and GPU by ~1e-3 (a *labeling* artifact, not a real
error). The fix in `dti_core.h::sym3_eigen_analytic` is to **explicitly sort** the
three eigenvalues descending; that took the observed CPU-vs-GPU fit discrepancy
from **1.4e-3 down to 8e-12** (true machine precision).

**Tractography is threshold-sensitive.** A streamline stops when FA crosses
`FA_MIN` or the curvature exceeds a limit. Because those are *hard thresholds*, a
harmless ~1e-12 difference in the underlying fit can move *where* a streamline
terminates and thus change its point count. This is real and worth internalising
(PATTERNS.md §4). We handle it honestly: the **fit is verified first** (CPU vs GPU
to 1e-9), and then **both** tractographies trace through the **one verified fit
field** (the GPU fit). That isolates the two verification stages and makes the
streamline comparison exact — rather than pretending a chaotic integrator is
bit-stable across two independently-computed fields.

## 6. How we verify correctness

`src/reference_cpu.cpp` is an independent, obviously-correct serial implementation
of the *same* computation. The per-voxel physics (`dti_core.h`) and the per-step
tractography (`tract_core.h`) are shared `__host__ __device__` code, so CPU and
GPU run identical arithmetic. `main.cu` runs both and checks:

- **Fit:** `max |Δ|` over FA, MD, and the three eigenvalues, tolerance **1e-9**.
  Justified: identical FP64 operations on both sides; we observe ~8e-12
  (residual `acos`/`cos` library differences), comfortably inside 1e-9.
- **Tractography:** `max |Δ|` over all streamline point coordinates, tolerance
  **1e-3 voxels** (a physically negligible fraction of a voxel). With both halves
  tracing the one verified field we observe an **exact 0** here; the small
  tolerance is an honest guard for FMA-level divergence on other inputs.

A second, *science-level* check (not just CPU==GPU): the synthetic phantom has a
**known answer**. The fit recovers **FA ≈ 0.80** on the bundle (matching the
ground-truth eigenvalues λ∥=1.7e-3, λ⊥=0.3e-3 → FA = 0.799…) and a principal
direction v1 **tangent to the arc**, and the streamlines trace the curve — so the
*model*, not only the two implementations, is validated.

Edge cases handled: all-zero (background) voxels → FA=0 not NaN (guarded division);
`acos` argument clamped to [−1,1]; fully-isotropic voxels → a fixed +x direction
so tractography never divides by zero.

## 7. Where this sits in the real world

This is a **reduced-scope teaching version** (the catalog marks 4.15 🟡 Active
R&D). Production dMRI pipelines go far beyond single-tensor DTI:

- **MRtrix3** — the gold standard: **Constrained Spherical Deconvolution (CSD)**
  resolves *crossing* fibers within a voxel (DTI's single tensor cannot — its one
  v1 averages crossings into a meaningless direction), then **iFOD2**
  *probabilistic* tractography samples the fiber orientation distribution, and
  **SIFT2** re-weights streamlines to match the diffusion data quantitatively.
- **FSL BEDPOSTX (GPU)** — Bayesian MCMC estimation of a multi-fiber model per
  voxel (~200× on GPU), feeding probabilistic tractography (`probtrackx`).
- **NODDI** — a richer biophysical model separating intra-/extra-neurite and CSF
  compartments (neurite density, orientation dispersion).
- **TractSeg / deep-learning tractography** — CNNs that segment known tracts
  directly, sidestepping streamline propagation.

What our version deliberately omits: crossing-fiber resolution (single tensor
only), motion/eddy-current preprocessing, multi-shell acquisition, anatomical
priors, and any statistical streamline filtering. What it faithfully teaches: the
Stejskal–Tanner model, the per-voxel OLS tensor fit, the FA/MD/eigenvector
derivation, and streamline integration with trilinear field interpolation — the
foundation every one of the above builds on.

---

## References

- **MRtrix3** — <https://github.com/MRtrix3/mrtrix3> — study `dwi2tensor` (the OLS
  fit we mirror) and `tckgen` (streamline propagation, including the curvature and
  FA stopping criteria).
- **DIPY** — <https://github.com/dipy/dipy> — `dipy.reconst.dti.TensorModel` is a
  readable reference for the exact fit here; `dipy.tracking` for deterministic vs.
  probabilistic tracking.
- **FSL** (`dtifit`, `bedpostx`) — <https://fsl.fmrib.ox.ac.uk/> — the Bayesian /
  multi-fiber counterpart, and the GPU-acceleration case study the catalog cites.
- **TractSeg** — <https://github.com/MIC-DKFZ/TractSeg> — the deep-learning
  approach to tract segmentation.
- Basser, Mattiello & LeBihan (1994), *Estimation of the effective self-diffusion
  tensor from the NMR spin echo* — the foundational DTI paper (FA/MD definitions).
- O. K. Smith (1961), *Eigenvalues of a symmetric 3×3 matrix* — the analytic
  eigensolver used in `dti_core.h`.
