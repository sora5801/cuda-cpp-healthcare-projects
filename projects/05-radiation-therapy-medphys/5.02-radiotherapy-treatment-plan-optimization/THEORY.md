# THEORY — 5.2 Radiotherapy Treatment-Plan Optimization

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See
> [README.md](README.md) for the quick tour and build steps.

---

## 1. The science

**Radiotherapy** treats cancer by depositing ionizing radiation in a tumor. The
central engineering problem is *conformity*: deliver a lethal dose to the tumor
while sparing the healthy tissue and critical organs packed around it. Modern
techniques — **IMRT** (intensity-modulated radiotherapy) and **VMAT** (volumetric
modulated arc therapy) — achieve this by breaking each beam into a grid of tiny
**beamlets**, each with an independently adjustable intensity ("fluence"). By
tuning thousands of beamlet weights, a planner can sculpt a dose cloud that hugs
the tumor.

Nobody tunes those weights by hand. Instead, **inverse planning** states the
clinical goal as a numerical objective and lets an optimizer find the weights.
The three actors:

- **PTV** (Planning Target Volume): the tumor plus a margin. We want its dose at
  the **prescription** `d_rx` (e.g. 60 Gy), uniform — both under-dose (missed
  tumor) and over-dose (hot spots) are bad.
- **OAR** (Organ At Risk): a sensitive organ (spinal cord, parotid, ...). We want
  its dose **below a tolerance** `d_max`. Anything under tolerance is fine.
- **BODY**: everything else. Keep stray dose low.

The physics that makes this tractable is **linearity**: dose adds up. If beamlet
`j` at unit intensity deposits `D[v,j]` gray in voxel `v`, then at intensity
`x_j` it deposits `D[v,j]·x_j`, and the total dose in `v` is the sum over all
beamlets. `D` — the **dose-influence** or **"dij" matrix** — is precomputed once
by a dose engine (Monte Carlo or pencil-beam). It is the bridge from "beam knobs"
`x` to "patient dose" `d`.

> **Not for clinical use.** This project uses a *synthetic 1-D phantom* and a
> simplified quadratic objective to teach the GPU pattern. Real planning uses 3-D
> anatomy, DVH/biological objectives, hard constraints, and validated dose
> engines (see §7).

---

## 2. The math

Let `x ∈ ℝ^{n_beam}` be the beamlet fluences (intensities), constrained to
`x ≥ 0` (a beam cannot emit negative intensity). The dose is the linear map

```
    d = D x ,        D ∈ ℝ^{n_vox × n_beam},   d ∈ ℝ^{n_vox}
```

`D` is **sparse**: a beamlet only irradiates a narrow corridor, so each column
(and hence each row) has only a handful of nonzeros. We store it in **CSR**
(§4.1). We minimize a **weighted-quadratic objective** that encodes the clinical
trade-off, with a per-structure penalty `pen_v`:

```
    F(x) = Σ_v  w_v · pen_v(d_v)

    PTV  :  pen_v(d) = (d − d_rx)^2                two-sided (any deviation hurts)
    OAR  :  pen_v(d) = max(0, d − d_max)^2         one-sided (only overdose hurts)
    BODY :  pen_v(d) = max(0, d − 0)^2             one-sided above zero
```

`w_v ≥ 0` is the clinical importance of voxel `v`. Because each `pen_v` is a
convex quadratic (or a convex one-sided quadratic) and `d = D x` is linear, `F`
is a **convex** function of `x`. On the feasible set `{x ≥ 0}` it therefore has a
global minimum, and gradient descent converges to it — no local-minima trap.

**The gradient.** By the chain rule, with the per-voxel *residual*
`r_v = ∂F/∂d_v`:

```
    r_v = w_v · pen_v'(d_v)
        = 2 w_v (d_v − d_rx)              (PTV, two-sided; sign carries direction)
        = 2 w_v · max(0, d_v − d_max)     (OAR/BODY, one-sided; ≥ 0)

    ∇F(x) = D^T r                         (transpose maps voxel residuals → beamlet grad)
```

So the whole problem is two linear maps sandwiching an element-wise nonlinearity:
`x → (D) → d → (residual) → r → (Dᵀ) → ∇F`. Those two maps, `D x` and `Dᵀ r`, are
**sparse matrix–vector products (SpMV)** — the computational heart of FMO and the
reason it lives on a GPU.

The per-voxel penalty and residual formulas above are exactly `voxel_penalty()`
and `voxel_residual()` in [`src/fmo.h`](src/fmo.h), shared verbatim by the CPU
reference and the GPU kernels.

---

## 3. The algorithm

We minimize with **projected gradient descent** (a.k.a. projected steepest
descent). Each iteration steps downhill, then *projects* back onto the feasible
box `x ≥ 0`:

```
    x ← 0                                   (start with the beams off → zero dose)
    repeat  iters  times:
        d    = D x                          (forward SpMV)              O(nnz)
        r_v  = w_v · pen_v'(d_v)  ∀v         (per-voxel residual)        O(n_vox)
        g    = D^T r                        (transpose SpMV)            O(nnz)
        x    = max(0, x − η g)              (grad step + project ≥ 0)   O(n_beam)
```

`η` is the step size (learning rate). The projection `max(0, ·)` is the closed
form of "nearest feasible point" for a non-negativity box — that is what makes
this *projected* gradient descent rather than plain gradient descent.

**Complexity.** Per iteration the cost is dominated by the two SpMVs at `O(nnz)`
each; the vector steps are `O(n_vox) + O(n_beam)`. Total: `O(iters · nnz)`. For a
clinical case `nnz ≈ 10^8` and `iters ≈ 100s`, so we do ~`10^{10}` multiply-adds —
serially that is seconds-to-minutes per plan; the GPU brings it to well under a
second, which is what enables **adaptive re-planning** (re-optimizing between
fractions as the anatomy changes).

```
   fluence x            dose d              residual r          gradient g
  ┌─────────┐   D x   ┌─────────┐  pen_v'  ┌─────────┐   Dᵀ r  ┌─────────┐
  │ n_beam  │ ──────▶ │ n_vox   │ ───────▶ │ n_vox   │ ──────▶ │ n_beam  │
  └─────────┘  SpMV   └─────────┘ per-voxel└─────────┘  SpMV   └─────────┘
       ▲                                                            │
       └──────────────── x ← max(0, x − η g)  (projected step) ◀────┘
```

We deliberately choose the *teachable* first-order method. Production planners use
**L-BFGS** (quasi-Newton: it builds curvature information from past gradients and
converges in far fewer iterations) or interior-point QP solvers (IPOPT). Both call
the same two SpMVs as their inner kernel, so the GPU lesson transfers directly —
see §7.

---

## 4. The GPU mapping

### 4.1 CSR: how the sparse matrix is stored

Compressed Sparse Row stores only the nonzeros, row by row:

```
   row_ptr : length n_vox + 1.  Row v's nonzeros are the range
             [row_ptr[v], row_ptr[v+1]) into the two arrays below.
   col_idx : length nnz.        col_idx[k] = beamlet index j of nonzero k.
   values  : length nnz.        values[k]  = D[v, j].
```

Instead of `n_vox·n_beam` numbers (10^{10} for a clinical case), CSR stores `nnz`
(~10^8) — the only reason `D` fits in GPU memory. It is also the exact layout
cuSPARSE expects. The matrix is uploaded to the device **once** and stays
resident; every iteration reads it without re-transferring.

### 4.2 The two SpMVs → cuSPARSE

The forward and transpose products are the classic GPU sparse kernel. We use
**cuSPARSE**, NVIDIA's tuned sparse library, via its modern *generic* API:

- `cusparseCreateCsr(...)` wraps our three device arrays in a matrix descriptor
  (records the layout; copies nothing).
- `cusparseCreateDnVec(...)` wraps each dense vector (`x`, `dose`, `r`, `grad`).
- `cusparseSpMV(handle, op, α, D, x, β, y, ...)` computes
  `y = α · op(D) · x + β · y`. We call it with `α=1, β=0` and
  - `op = NON_TRANSPOSE` → `dose = D x`,
  - `op = TRANSPOSE`     → `grad = Dᵀ r`.

**Why a library and not hand-rolled?** A correct, *fast* CSR SpMV is surprisingly
subtle: the naive "one thread per row" kernel suffers severe **load imbalance**
when row lengths vary (a fat row stalls its warp while neighbors idle), and it
reads `x` with poor coalescing. Production SpMV uses vectorized/merge-based row
assignment, warp-level reduction of each row's partial products, and careful
memory access — dozens of tuned variants selected by matrix shape. cuSPARSE picks
one for us (`CUSPARSE_SPMV_ALG_DEFAULT`). Writing it by hand is a whole project of
its own; using the library lets us focus on the *optimizer*, while §7 and these
comments keep it from being a black box. cuSPARSE may request a scratch
**workspace** buffer (queried with `cusparseSpMV_bufferSize`); we allocate the max
of the two ops' needs once.

### 4.3 The two element-wise steps → our kernels

The residual and the projected update are trivially parallel (one output element
per thread, no communication), so they are two small hand-written kernels
([`src/kernels.cu`](src/kernels.cu)):

- `residual_kernel`: thread `v` computes `r[v] = voxel_residual(spec[v], dose[v])`.
  Grid = `⌈n_vox / 256⌉`, block = 256. No shared memory, no atomics.
- `update_kernel`: thread `j` computes `x[j] = max(0, x[j] − η·g[j])`. Grid =
  `⌈n_beam / 256⌉`, block = 256.

Both call the **same** `__host__ __device__` functions from `fmo.h` that the CPU
reference calls, so the scalar math is byte-identical across CPU and GPU.

### 4.4 Memory hierarchy

`D` (CSR), `x`, `dose`, `r`, `grad`, and the voxel specs all live in **global
memory**. SpMV is **memory-bandwidth bound** — it does ~2 flops per matrix element
loaded — so the win comes from the GPU's high bandwidth and from keeping `D`
resident (no PCIe traffic in the loop). Block size 256 gives the scheduler eight
warps to hide memory latency. Our two element-wise kernels are also
bandwidth-bound and need no shared memory.

---

## 5. Numerical considerations

**Precision.** We use **FP32** throughout: dose-influence values carry only a few
significant figures (a dose engine's own uncertainty is ~1–2%), and FP32 halves
the memory footprint and bandwidth of the giant matrix — the right engineering
choice. The clinical quantities (dose in Gy) are `O(10–60)`, comfortably inside
FP32's range and precision.

**Determinism and the tolerance.** The per-voxel scalar math is shared, so the
*only* source of CPU-vs-GPU divergence is the **summation order inside the two
SpMVs**. Floating-point addition is not associative, so cuSPARSE's parallel
row-reduction and Dᵀ scatter accumulate in a different order than the CPU's serial
loop. Over a single SpMV this is ~`1e-6` relative; over **hundreds of gradient
iterations** it compounds. Per PATTERNS.md §4, this is a *long iterative solver*,
so we verify to a small **physical** tolerance rather than bit-equality:

- fluence agreement within `1e-2` (fluence units),
- **dose agreement within `1e-2` Gy** — negligible against a ~60 Gy prescription.

In our run the measured differences are ~`1e-5` (fluence) and ~`8e-6` Gy (dose),
far inside tolerance. We do **not** claim bit-identical results, and we say so.

**stdout determinism.** The reported plan stats are computed from the **CPU**
dose, which is identical on every run, so `demo/expected_output.txt` is stable.
The run-varying numbers (timings, the measured CPU-vs-GPU error) go to **stderr**,
which the demo shows but does not diff (PATTERNS.md §3). No atomics are used in our
kernels, so there is no atomic-ordering nondeterminism to worry about; cuSPARSE's
SpMV is itself deterministic for a fixed matrix and algorithm.

**Step size / convergence.** `η` must be small enough for stability (too large and
the quadratic diverges) but large enough to converge in `iters` steps. The sample
ships tuned values (`η = 0.02`, 400 iters). The optimal `η` relates to the largest
eigenvalue of `DᵀW D`; L-BFGS sidesteps the choice by estimating curvature.

---

## 6. How we verify correctness

Three independent checks, in increasing strength:

1. **CPU vs GPU (agreement).** `optimize_cpu()` and `optimize_gpu()` run the same
   projected-gradient algorithm; `main.cu` compares the final fluences and the
   doses they produce, asserting agreement within the §5 tolerance. This catches
   any bug in the GPU plumbing (wrong transpose op, bad CSR upload, indexing).

2. **The science is recovered (physical sanity).** After optimization the
   synthetic plan reaches **PTV mean ≈ 59.2 Gy** against a 60 Gy prescription
   (good coverage) with a homogeneity index ≈ 0.13, while the OAR mean is held to
   ~11 Gy — the optimizer visibly *learned* to aim intensity at the tumor and pull
   it off the organ. The objective drops from its initial value (all dose zero →
   pure PTV under-dose penalty) to a small residual. This validates the *model*,
   not just CPU==GPU.

3. **Structural / build gates.** `tools/verify_project.py` checks the layout,
   required doc sections, and comment density; both `Release|x64` and `Debug|x64`
   build with zero warnings.

**Edge cases** handled: the ragged last thread block (guarded in both kernels), an
empty/short sample (the loader throws), a zero-union/zero-mean structure (guarded
in `compute_stats`), and `β=0` SpMV (output need not be pre-zeroed).

---

## 7. Where this sits in the real world

This is a **reduced-scope teaching version**. How production inverse planning
differs:

- **Anatomy & dose engine.** Real `D` comes from a 3-D CT, a beam model, and a
  validated Monte-Carlo or collapsed-cone/pencil-beam dose engine — not a 1-D
  Gaussian phantom. `nnz` is `10^8`–`10^9`.
- **Objective.** Real objectives combine DVH-point objectives, **dose-volume
  constraints** (e.g. "≤ 30% of the rectum above 40 Gy"), **biological** models
  (TCP/NTCP — tumor-control / normal-tissue-complication probability), and
  **robust** optimization over many setup/range-uncertainty scenarios (the
  minimax over ~50–100 scenarios that multiplies the SpMV count). Some of these
  are non-convex.
- **Optimizer.** L-BFGS or interior-point QP (IPOPT), often with a **direct
  aperture optimization (DAO)** or VMAT layer that converts fluence into
  deliverable MLC (multi-leaf collimator) leaf sequences and gantry motion.
- **Deep learning.** Knowledge-based planning (e.g. **OpenKBP**) trains a U-Net to
  *predict* an achievable dose from anatomy, which then seeds or constrains the
  optimizer — cutting planning time further.

Crucially, in **all** of these the innermost hot loop is still `D x` and `Dᵀ r`
SpMVs on a GPU-resident CSR matrix. The pattern you learn here is exactly the one
matRad, pyRadPlan, and OpenTPS accelerate. Prior art to study: **matRad**
(MATLAB, photon/proton/carbon), **pyRadPlan** (its Python port), **CERR** (DICOM-RT
research platform), **OpenTPS** (Python/GPU). See the README for links.

---

### Further reading

- Bortfeld, "IMRT: a review and preview," *Phys. Med. Biol.* (2006).
- Nocedal & Wright, *Numerical Optimization* — projected gradient, L-BFGS.
- NVIDIA cuSPARSE documentation — the generic SpMV API used here.
- Babier et al., "OpenKBP: the open-access knowledge-based planning grand
  challenge," *Med. Phys.* (2021).
