# THEORY — 4.16 Functional MRI Analysis

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See
> [README.md](README.md) for the quick tour and build steps, and
> [`src/glm.h`](src/glm.h) for the shared host/device math this document derives.

---

## The science — what fMRI measures and the activation question

Functional MRI does not measure neural firing directly. It measures the **BOLD**
signal — *Blood-Oxygen-Level-Dependent* contrast. When a patch of cortex works
harder, local neurons consume oxygen; the vasculature over-compensates a few
seconds later by flooding the region with oxygenated blood. Oxygenated and
deoxygenated hemoglobin have different magnetic susceptibilities, so the MR signal
in that voxel rises and falls with this hemodynamic response.

An fMRI scan is therefore a **4-D movie**: at every voxel (a small 3-D cube of
brain, typically 2–3 mm on a side) we get a **time-series** of `T` samples, one per
scanner *repetition time* (`TR`, often ~1–2 s). A whole-brain run is on the order
of `V ≈ 10⁵` gray-matter voxels × `T ≈ 10²–10³` scans.

The classic **task-fMRI** question is: *which voxels responded to the experimental
task?* In a **block design**, the subject alternates between a task condition (e.g.
finger-tapping) and rest in fixed-length blocks. We want a per-voxel **activation
map**: a statistic at each voxel measuring how strongly its BOLD time-course tracks
the task.

Two facts make this non-trivial and set up the model:

1. **The response is delayed and shaped.** A brief neural event does not produce a
   spike in BOLD; it produces a smooth **hemodynamic response function (HRF)** that
   rises, peaks ~5–6 s later, and dips below baseline (a *post-stimulus
   undershoot*) before recovering ~20–30 s later. So the predicted BOLD is the task
   on/off *boxcar* **convolved with the HRF**, not the boxcar itself.
2. **The signal drifts and is noisy.** Scanner instabilities add slow drift; thermal
   and physiological noise add fast fluctuations. A good model must absorb the drift
   and estimate the noise so the activation statistic is trustworthy.

---

## The math — the General Linear Model and the t-statistic

The workhorse of task-fMRI (SPM, FSL FEAT, AFNI) is the **mass-univariate General
Linear Model (GLM)**: the *same* linear model is fit **independently at every
voxel**. For one voxel with time-series `y ∈ ℝ^T`:

```
    y  =  X β  +  ε ,        ε ~ N(0, σ² I)
```

- `X ∈ ℝ^{T×K}` is the **design matrix**, *shared by every voxel*. Its columns are
  the regressors. Here `K = 3`:
  - **col 0 — task**: the block boxcar convolved with the canonical HRF (the
    "activation" predictor).
  - **col 1 — drift**: a linear ramp mapped to `[-1, +1]` (soaks up slow drift).
  - **col 2 — intercept**: a constant `1` (the baseline mean).
- `β ∈ ℝ^K` are the fitted weights for this voxel; `β₀` is the task amplitude.
- `ε` is the residual noise.

### Ordinary least squares (OLS)

The least-squares estimate minimizes `‖y − Xβ‖²`, giving the **normal equations**:

```
    (Xᵀ X) β  =  Xᵀ y        ⇒        β̂ = (Xᵀ X)⁻¹ Xᵀ y
```

`Xᵀ X` is a `K×K = 3×3` matrix that depends only on `X` — **not on the voxel** — so
it (and its inverse) is computed **once** and reused for all `V` voxels.

### The activation statistic

We test the contrast `c = [1, 0, 0]` — "is the task weight `β₀` significantly
non-zero?". The residual variance and the standard error of the contrast are

```
    RSS   = ‖y − X β̂‖²
    σ̂²   = RSS / (T − K)                          (unbiased noise variance)
    Var(cᵀβ̂) = σ̂²  cᵀ (Xᵀ X)⁻¹ c  =  σ̂² · (Xᵀ X)⁻¹₀₀
    t     = cᵀβ̂ / sqrt(Var(cᵀβ̂))  =  β̂₀ / sqrt(σ̂² (Xᵀ X)⁻¹₀₀)
```

`t` follows a Student-t distribution with `T − K` degrees of freedom under the null.
Large `|t|` ⇒ strong, reliable activation. The per-voxel `t` is exactly what
[`fit_voxel()`](src/glm.h) returns and what the demo ranks.

### The HRF

We use SPM's canonical HRF, a **difference of two gamma densities**:

```
    h(t) = g(t; 6, 1)  −  (1/6) · g(t; 16, 1),     g(t; a, b) = bᵃ t^{a−1} e^{−bt} / Γ(a)
```

The first gamma is the positive peak (~6 s); the second, scaled by 1/6, is the
undershoot (~16 s). We evaluate `g` in **log space** (`lgamma` + `exp`) for
numerical safety — `t^{a−1}` and `Γ(a)` overflow individually but their logs are
tame. See `canonical_hrf()` / `gamma_pdf()` in `src/glm.h`.

---

## The algorithm — steps and complexity

Per voxel (all in [`fit_voxel()`](src/glm.h)):

1. **Build `Xᵀy`** — one pass over `t`, accumulating `K` dot-products. `O(T·K)`.
2. **Solve** `β̂ = (XᵀX)⁻¹ (Xᵀy)` — a `3×3` matrix-vector product. `O(K²)`.
3. **Residual sum of squares** — second pass over `t` recomputing the fitted value.
   `O(T·K)`.
4. **`σ̂²`, contrast SE, `t`** — `O(1)`.

Precomputed **once** for the whole dataset (in `compute_XtX_inv()`):

- Assemble `XᵀX` (a `3×3` Gram matrix) by summing outer products over `t`: `O(T·K²)`.
- Invert it in closed form via cofactors (`invert_sym3()`): `O(1)` for fixed `K=3`.

**Total serial cost:** `O(T·K² + V·T·K)`. With `K=3` fixed this is `O(V·T)` — linear
in the data. The point is the constant factor and the **parallelism**: the `V`
voxel fits are completely independent.

Building the design columns on the fly (instead of materializing `X`) costs a little
arithmetic per voxel but keeps memory traffic to just the `y`-rows — the right trade
on a GPU, where compute is cheap and bandwidth is precious.

---

## GPU mapping — one thread per voxel

This is the **"many identical small solves"** pattern (docs/PATTERNS.md §1),
structurally identical to the 9.02 SEIR ensemble: the same computation runs for many
independent items, so we assign **one GPU thread per voxel**.

```
   grid  : ceil(V / 256) blocks (capped at 4096; a grid-stride loop covers any V)
   block : 256 threads  (multiple of the 32-lane warp; good occupancy sm_75..sm_89)
   thread (blockIdx.x, threadIdx.x) -> voxel v = block*blockDim + thread, stride grid
```

```
      voxel-major BOLD in global memory            per-voxel outputs
      row v = [ y_0 y_1 ... y_{T-1} ]              t[v], beta[v]
              │                                        ▲
   thread v ──┘  reads its own T-length row,          │
                 calls fit_voxel(), writes ───────────┘
```

### Memory hierarchy — why constant memory for the shared operands

Two operands are **voxel-independent** and read by *every* thread but written by
none during the launch: the design parameters (`GlmDesign`) and the precomputed
`(XᵀX)⁻¹` (9 doubles). Those live in **`__constant__` memory**
(`c_design`, `c_XtX_inv` in [`kernels.cu`](src/kernels.cu)). The constant cache
*broadcasts* one address to an entire warp in a single transaction — the ideal home
for a read-only value shared warp-wide. This is the same idiom flagship 1.12 uses
for its query fingerprint.

Each thread streams its own `y`-row from **global memory**; the `β`, `RSS`
accumulators live in **registers**. No **shared memory** and no **atomics** are
needed because the outputs are fully independent — this is pure *map* parallelism.

### Coalescing — the honest caveat

We store BOLD **voxel-major** (`bold[v*T + t]`), so each thread reads a contiguous
`y`-row. That is cache-friendly *per thread* but **not** coalesced *across* threads:
at a given loop index `t`, adjacent threads `v` and `v+1` read addresses `T` doubles
apart. A **time-major** layout (`bold[t*V + v]`) would coalesce adjacent threads
into one wide transaction and is what a throughput-tuned implementation would use.
We keep voxel-major here because the one-thread-one-row mapping is far easier to
read and the sample is tiny; the layout trade-off is left as an exercise.

### Why not cuSOLVER/cuBLAS here?

The catalog notes cuBLAS/cuSOLVER for the *large* fMRI workloads (ICA SVD, `V×V`
connectivity GEMM). For the **GLM with `K=3`**, each solve is a `3×3` system — far
too small to hand to a library call per voxel. A hand-rolled cofactor inverse in
registers is both faster and, crucially, runs the **same operations on host and
device** so verification is near-exact. The `V×V` connectivity matrix (a genuine
`cuBLAS Dsyrk`/`Dgemm` job) is described under "real world" below and left as an
exercise.

---

## Numerical considerations

- **Double precision throughout.** The t-statistic divides by a standard error that
  can be small; FP64 gives head-room and keeps the CPU/GPU results essentially
  identical. All of `glm.h` is `double`.
- **HRF in log space.** `gamma_pdf` computes `lgamma`/`exp` rather than
  `pow`/`tgamma` to avoid intermediate overflow.
- **Rank-deficiency guard.** `invert_sym3()` returns the determinant; `main.cu`
  aborts with a clear message if `det(XᵀX) = 0` (a degenerate design), rather than
  dividing by zero.
- **No atomics ⇒ deterministic.** Each thread writes its own outputs; there is no
  cross-thread floating-point reduction, so results do not depend on thread
  scheduling. stdout is byte-identical every run (t-stats printed to 4 decimals).
- **FMA divergence.** The only host/device difference is that the GPU may fuse a
  multiply-add where the host compiler does not, reassociating a few FP64 products.
  That shows up at the ~`10⁻¹³` level here — far below our `10⁻⁹` tolerance and
  utterly invisible at 4 printed decimals (docs/PATTERNS.md §4).

---

## How we verify correctness

Verification is **two-layered**:

1. **CPU == GPU (mechanical).** `glm_cpu()` and `glm_kernel()` both loop over voxels
   calling the *same* `fit_voxel()` from `glm.h` (the HD-core idiom,
   docs/PATTERNS.md §2). So the arithmetic is identical up to FMA. `main.cu` computes
   `max_abs_err` over the two t-statistic vectors and asserts it is `≤ 1e-9`. On the
   committed sample the observed error is ~`3.7e-13`.
2. **Recovers the planted answer (scientific).** The synthetic generator injects a
   real HRF-convolved task response into a known subset of voxels and labels them
   `active` (labels the fit never reads). The demo reports that the **top-6 voxels by
   t-statistic are all truly active** ("recovered 6/6") — evidence the whole pipeline
   (HRF, design, OLS, t-test) is correct, not just that CPU and GPU agree.

Edge cases handled: ragged last block (grid-stride guard `v < V`), singular design
(`det == 0` abort), `T ≤ K` (degrees of freedom guard → `t = 0`).

---

## Where this sits in the real world

This is a faithful but **reduced-scope teaching version** of a task-fMRI GLM.
Production tools do much more:

- **Preprocessing.** Real pipelines (fMRIPrep, FSL, SPM) do motion correction,
  slice-timing correction, spatial smoothing, and registration to a standard
  template *before* the GLM. We assume clean, aligned data.
- **Better noise models.** We use OLS with a single drift regressor. Real analyses
  add motion regressors, high-pass filters (a DCT/cosine basis), physiological
  nuisance regressors, and **prewhitening** (GLS/AR(1)) because BOLD noise is
  temporally autocorrelated — OLS t-values are otherwise inflated.
- **Multiple comparisons.** With `V ≈ 10⁵` voxels, thresholding raw t-maps produces
  many false positives. Real work uses cluster-wise inference, random field theory,
  or **permutation testing** (the catalog's cuRAND note) to control family-wise error
  or FDR.
- **Beyond the GLM.** The catalog also lists **ICA/MELODIC** (a `T×V` SVD via
  cuSOLVER), **resting-state functional connectivity** (a `V×V` correlation matrix —
  a genuine `cuBLAS` GEMM/`Dsyrk`), graph-theoretic network analysis, HMM dynamic
  connectivity, and CNN/transformer biomarkers. Those are where the heavy CUDA
  libraries earn their keep; this project deliberately teaches the *simplest complete*
  activation model first.

**Not for clinical use.** Everything here is educational and runs on synthetic data.

### Exercises

See [README.md](README.md#exercises).
