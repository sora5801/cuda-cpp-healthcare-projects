# THEORY — 4.25 Image Harmonization Across Scanners/Sites

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. Diagrams in Mermaid/ASCII
> are welcome. See [README.md](README.md) for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## The science

Imaging biomarkers — cortical thickness, regional brain volumes, radiomic texture
features, DTI scalars — are increasingly pooled across **many hospitals** to get the
sample sizes that rare-disease and effect-size studies need. But an MRI or CT feature
measured on a Siemens 3T scanner is **not** the same number as the "same" feature on a
GE 1.5T scanner. Field strength, gradient hardware, reconstruction kernel, coil design,
and site-specific protocol all impose a systematic offset and a change in variance on
every feature. This is the **scanner (batch/site) effect**.

The danger is that this hardware signature is **confounded** with the biology. If site A
happens to scan more patients and site B more controls, a naive group comparison will
report a "disease effect" that is really a scanner effect. Multi-site consortia (ENIGMA,
ABIDE, ADNI) treat harmonization as a mandatory preprocessing step.

Two families of methods exist:

1. **Statistical harmonization** (this project): model the batch effect on **extracted
   features** and subtract it. **ComBat**, borrowed from genomics microarray
   normalization, is the de-facto standard (NeuroComBat, `neuroCombat`).
2. **Image-level harmonization:** learn a mapping between the *appearance* of images from
   different scanners with deep nets — **CycleGAN**, **CALAMITI**, **DeepHarmony**. These
   operate on voxels, need paired or unpaired training, and are expensive (§real world).

We implement (1) because it is statistically transparent, GPU-tractable as a teaching
project, and — importantly — is what a large fraction of published pipelines actually run.

## The math

Let there be **N** samples, **P** features, and **B** batches (scanners). For a single
feature (one row of the data), model each sample's value `y_n` as

```
    y_n  =  x_nᵀ β_cov  +  γ_{b(n)}  +  δ_{b(n)} · ε_n ,      ε_n ~ N(0, σ²)
            └── biology ──┘  └ batch loc ┘ └ batch scale ┘
```

where `b(n)` is sample n's batch, `x_n` are the **biological covariates** to preserve,
`β_cov` their coefficients, `γ_b` the batch's additive (location) shift, and `δ_b` its
multiplicative (scale) shift. **Harmonization = estimate (γ_b, δ_b) and remove them**,
keeping `x_nᵀβ_cov` and the grand mean.

**Step 1 — model fit.** Stack the design `X = [ covariates | batch-indicators ]`
(an `N × M` matrix, `M = C + B`) and solve ordinary least squares per feature:

```
    β = (Xᵀ X)⁻¹ Xᵀ y                                        (normal equations)
```

The grand mean is the batch-size-weighted average of the batch coefficients,
`α = Σ_b (n_b / N) · β_batch[b]`, and the pooled residual SD is
`σ = sqrt( (1/N) Σ_n (y_n − x_nᵀβ)² )`.

**Step 2 — standardize.** Remove the *preserved* part (grand mean + covariate fit) and
scale by `σ`:

```
    z_n = ( y_n − α − x_cov,nᵀ β_cov ) / σ
```

**Step 3 — raw batch L/S.** For each batch b (with `n_b` members):

```
    γ̂_b = mean_{n∈b}  z_n                (location signature)
    δ̂_b = var_{n∈b}   z_n                (scale signature)
```

**Step 4 — empirical Bayes (the heart of ComBat).** Rather than trust each feature's
noisy `γ̂, δ̂`, ComBat treats the collection of features as draws from a common prior and
**shrinks** each estimate toward that prior. Assuming `γ_b ~ N(γ̄_b, τ²_b)` and
`δ_b ~ InverseGamma(a_b, b_b)`, the posterior means are

```
    γ*_b = ( n_b · τ²_b · γ̂_b  +  δ*_b · γ̄_b ) / ( n_b · τ²_b  +  δ*_b )

    δ*_b = ( b_b  +  ½ Σ_{n∈b} (z_n − γ*_b)² ) / ( n_b/2  +  a_b − 1 )
```

The priors `(γ̄_b, τ²_b, a_b, b_b)` are fit **once, across all P features**, by
method-of-moments (NeuroComBat's `aprior`/`bprior`). Small batches (small `n_b`) get
pulled harder toward the prior — they *borrow strength* from the whole panel. This is
precisely what makes ComBat more robust than a per-batch z-score.

**Step 5 — adjust and back-transform.**

```
    z*_n = ( z_n − γ*_{b(n)} ) / sqrt( δ*_{b(n)} )
    y*_n = z*_n · σ  +  α  +  x_cov,nᵀ β_cov              (harmonized value)
```

## The algorithm

Serial pseudocode for the whole table (`combat_cpu` in `reference_cpu.cpp` is exactly this):

```
build design X = [covariates | batch indicators]          # once, shared
fit EB priors (γ̄, τ², a, b) from raw batch stats of all P # once, an across-feature reduce
for each feature p in 0..P-1:                             # ← embarrassingly parallel
    β        = solve(XᵀX, Xᵀ y_p)                         # OLS, M×M system
    α, σ     = grand-mean + pooled residual SD
    z        = (y_p − α − covariate_fit) / σ              # standardize
    γ̂, δ̂    = per-batch mean/var of z
    γ*, δ*   = EB-shrink(γ̂, δ̂ ; priors)                  # borrow strength
    y*_p     = z-adjust(z, γ*, δ*) · σ + α + covariate_fit
```

**Complexity.** Per feature: `O(N·M²)` to form `XᵀX`, `O(M³)` to solve (M ≤ 16, trivial),
`O(N·B)` for the batch stats — so `O(N·M² + M³)` dominated by the accumulation. Across the
table: **`O(P · (N·M² + M³))` total, but the P features are fully independent**, so the
parallel *span* (critical path) is just one feature's `O(N·M² + M³)`. That independence is
the entire reason the GPU wins as P grows.

## The GPU mapping

```
   features (P)          one CUDA thread  ─────────────►  one feature's ComBat
   ┌───────────────┐     ┌───────────────────────────────────────────────┐
 p0│■■■■■■■■■■■■■■■│ →   │ fit β  → α,σ → z → γ̂,δ̂ → γ*,δ* → y*  (in regs) │
 p1│■■■■■■■■■■■■■■■│ →   └───────────────────────────────────────────────┘
 …│      …        │            grid = ceil(P / 128) blocks
pP│■■■■■■■■■■■■■■■│            block = 128 threads
   └───────────────┘
```

- **Thread-to-data map.** `p = blockIdx.x·blockDim.x + threadIdx.x` owns feature row `p`,
  i.e. it reads `d_Y[p*N .. p*N+N-1]` and writes `d_out[p*N ..]`. The ragged last block is
  guarded by `if (p >= P) return;`.
- **Memory hierarchy.** The feature row and the shared design/priors live in **global**
  memory (read-only, coalesced across threads for the design). *All* per-feature scratch —
  the `M×M` normal matrix `AtA`, `β`, `γ`, `δ` — lives in **registers / local memory**
  inside `cb_harmonize_feature`. That is why the design width is capped at `CB_MAX_M = 16`:
  it keeps the scratch small enough to stay register-resident, so threads never touch a
  shared global scratch buffer and stay independent.
- **No shared memory, no atomics, no cross-thread reduction.** Unlike the k-means flagship
  (11.09), ComBat's per-feature independence means there is *nothing to reduce across
  threads* in the harmonization kernel — the only reduction (the prior fit) is a cheap
  host step over P. This is the **ensemble** pattern (PATTERNS.md §1, like 9.02 / 13.02).
- **The shared `__host__ __device__` core.** `cb_harmonize_feature` (in `combat.h`) is
  compiled for *both* host and device. The CPU reference loops it; the GPU kernel calls it
  from one thread. Same code, same math → exact verification (PATTERNS.md §2).
- **Why not cuBLAS/cuSOLVER here?** The natural library call would be a *batched* small
  solve (`XᵀX β = Xᵀy` for every feature). But with `M ≤ 16` and the whole system fitting
  in registers, a hand-written Gauss-Jordan solve (`cb_solve_normal_equations`) is faster,
  allocation-free, and — crucially for a study repo — a readable **white box**. For very
  wide designs (large B, or voxel-level covariates) you would batch the GEMM `XᵀX = XᵀX`
  with `cublasDgemmStridedBatched` and the solves with `cusolverDnDpotrfBatched`; we note
  that path but keep the didactic version. (Contrast the flagships that *do* use the
  library because the problem is genuinely large: `3.11` cuBLAS GEMM, `2.06` cuSOLVER.)

## Numerical considerations

- **Precision: FP64 throughout.** ComBat's OLS + variance estimates are sensitive to
  cancellation; double precision keeps the CPU and GPU within ~`1e-15`.
- **Full-rank design — the load-bearing subtlety.** We use `X = [covariates | B batch
  dummies]` **with no intercept column**. The B batch dummies already sum to 1 for every
  sample, so they *span* the intercept; adding a separate all-ones column makes `XᵀX`
  **singular**. A singular normal system has infinitely many solutions, and Gauss-Jordan
  then hits a (near-)zero pivot whose handling depends on floating-point rounding order —
  which differs between the host compiler and nvcc's device FMA, and even between two host
  compilers. The symptom is `CPU ≠ GPU` by ~1 unit (we hit exactly this during
  development). **Dropping the intercept restores full rank** and makes the solve
  well-conditioned and reproducible; the grand mean is recovered from the batch
  coefficients instead. (Exercise 3 lets you reproduce the failure on purpose.)
- **Determinism.** With a full-rank design, `cb_harmonize_feature` is a fixed sequence of
  double-precision operations in the same order on host and device. No atomics, no
  reduction-order dependence → the stdout report is byte-identical every run.
- **Guards.** Constant features (`σ² ≈ 0`), empty batches (`n_b = 0`), and degenerate
  priors are clamped to small positive floors so a pathological feature cannot divide by
  zero or crash the whole launch.

## How we verify correctness

Two independent checks, following PATTERNS.md §4:

1. **CPU == GPU (exactness of the port).** `combat_cpu` and `combat_gpu` call the *same*
   `__host__ __device__` core on the *same* FP64 inputs. `main.cu` computes
   `max |y*_GPU − y*_CPU|` over the whole table and asserts it ≤ `1e-9`. We observe
   `~7e-15` — pure FMA rounding, far below the tolerance. This proves the GPU port is
   faithful, not merely "close".
2. **The science actually happened.** We compute the **max across-scanner feature-mean
   gap** on the raw table and on the harmonized table. It collapses from `7.74` to `0.80`
   — the scanner location signature is removed. It is deliberately *not* exactly zero:
   empirical-Bayes shrinkage trades a little residual gap for robustness (that is the whole
   point vs a naive z-score). Both numbers are printed to stdout.

## Where this sits in the real world

- **NeuroComBat / neuroCombat** implement this exact statistical model (plus optional
  iterative EM refinement of the EB posterior and a non-parametric prior variant). Our
  closed-form single-step update matches the parametric path.
- **LongCombat** adds within-subject random effects for longitudinal (repeated-scan)
  studies; **CovBat** additionally harmonizes the covariance structure.
- **Image-level deep harmonization** — the catalog's headline methods — is a different
  beast: **CycleGAN** learns an unpaired voxel-to-voxel mapping between scanner "styles"
  with a cycle-consistency loss; **CALAMITI** disentangles anatomy from contrast in a
  latent space; **DeepHarmony** uses paired traveling-subject data. These train on 256³
  volumes (~8 GB each) for ~100 GPU-hours using cuDNN convolutions and Tensor-Core FP16,
  and harmonize *images* rather than *features*. Inference is a single forward pass.
  **Federated** variants (NCCL all-reduce of weights) train across sites without sharing
  raw scans. They are more powerful but far heavier and harder to validate than ComBat —
  which is why ComBat remains the workhorse for feature-level studies, and why we chose it
  as the teachable core here (CLAUDE.md §13).
