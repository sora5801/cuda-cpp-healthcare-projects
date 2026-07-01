# THEORY — 4.14 Digital Breast Tomosynthesis

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

**Mammography** is a projection X-ray of the compressed breast: the tube fires,
the beam passes through the breast, and the detector records how much was
absorbed. The problem is that the breast is a **thick, low-contrast** organ —
overlapping fibroglandular tissue can hide a real lesion (tissue superposition),
and a clump of benign tissue can mimic one. A single flat projection collapses all
depth into one plane.

**Digital Breast Tomosynthesis (DBT)** attacks this by adding a little depth
information. The X-ray tube sweeps a **narrow arc** — typically ±7.5° to ±25°
(9–25 projections total) — while the breast stays compressed. From these few
angled views a stack of **thin in-focus slices** (planes parallel to the detector)
is reconstructed. Structures at different depths now separate across slices, so a
lesion that was hidden behind tissue in the flat view can become conspicuous.

The catch is the **limited angular range**. A full CT scan spins 180–360° and
samples the object from all directions; DBT sees it only through a thin wedge. As
§2 shows, that leaves a huge region of the object's spatial-frequency content
**unmeasured** (the "missing wedge"), which makes the reconstruction **ill-posed**:
the analytic inverse (Filtered BackProjection, project 4.01) amplifies noise and
smears structures in depth. This is why DBT relies on **iterative** algebraic /
statistical reconstruction, which incorporates prior constraints (non-negativity,
smoothness) and degrades gracefully under missing data. This project implements the
canonical iterative method, **SART**.

## 2. The math

**The forward model.** X-ray attenuation follows Beer–Lambert: a ray of initial
intensity `I₀` through a medium of linear attenuation coefficient `μ(x,y)` exits
with `I = I₀·exp(−∫ μ dℓ)`. Taking the log linearizes it, so the *measured
projection* along a ray `R` is a **line integral**:

```
p(R) = −ln(I/I₀) = ∫_R μ(x, y) dℓ
```

Parameterize a 2-D ray by its angle `θ` and signed detector offset `s`. The ray is
the line `{ (x,y) : x·cosθ + y·sinθ = s }`. The map from the image `μ` to the full
set of projections is the **Radon transform** `A`:

```
p(θ, s) = (A μ)(θ, s) = ∫∫ μ(x,y) · δ(x cosθ + y sinθ − s) dx dy
```

**Discretized**, the image is a vector `x ∈ ℝ^{N²}` (pixel attenuations) and the
projections are a vector `b ∈ ℝ^{M}` (M = n_angles × n_det). The forward operator
becomes a sparse matrix `A ∈ ℝ^{M×N²}` where `A[r, i]` is the length of ray `r`'s
intersection with pixel `i`. Reconstruction is the linear system:

```
A x = b        (find the image x whose projections match the measured data b)
```

**Symbols** (matching the code): `μ` / `x` = attenuation image (arbitrary units in
our synthetic phantom); `θ_k` = the k-th projection angle, in DBT confined to the
wedge `[−half_span, +half_span]`; `s_j = (j − (n_det−1)/2)·ds` = detector bin
offset; `W` = world half-extent (image spans `[−W, W]²`); `λ` = SART relaxation.

**Why limited angle is ill-posed (the Fourier-slice theorem).** The 1-D Fourier
transform of the projection at angle `θ` equals a **radial slice** at angle `θ`
through the 2-D Fourier transform of the image. A full 180° scan fills all of
Fourier space; a ±25° wedge fills only two opposing 50°-wide sectors, leaving a
**missing wedge** of ~130° in each direction totally unmeasured. No analytic
formula can invert what was never measured — hence iterative methods that *impose*
what we know a priori (μ ≥ 0, piecewise smoothness).

**SART as a solve.** SART is a relaxed, block-iterative solver for `A x = b`. One
sweep is, conceptually,

```
x ← x + λ · D_col · Aᵀ · D_row · (b − A x)
```

where `A x` is the forward projection, `b − A x` is the residual, `Aᵀ(·)` is the
backprojection (the transpose of forward projection), and `D_row`, `D_col`
normalize by ray and pixel weights. Our teaching version uses a simple, fully
deterministic normalization: divide the backprojected residual by the number of
angles (`D_col ≈ 1/n_angles`) and clamp to ≥ 0. It converges to a least-squares /
minimum-norm solution of the (under-determined) system.

## 3. The algorithm

For a fixed number of iterations `T` (here 20):

```
x ← 0                                        # start: empty breast (all air)
repeat T times:
    sim  ← forward_project(x)                # A x     : one line integral per ray
    res  ← b − sim                           # residual: measured − simulated
    corr ← backproject(res) / n_angles       # Aᵀ res  : one correction per pixel
    x    ← max(x + λ · corr, 0)              # relaxed, non-negative update
```

**Forward projection** (`forward_ray_integral` in `dbt_geometry.h`): for ray
`(k, j)`, march `n_steps ≈ 2N` samples along the line `p(t) = s·(cosθ,sinθ) +
t·(−sinθ,cosθ)` for `t ∈ [−√2·W, √2·W]`, **bilinearly** sampling the image and
summing × the per-step world length. That √2·W half-length guarantees the whole
image is covered at every angle.

**Backprojection** (`backproject_update`): for each pixel `(px,py)` at world
`(wx,wy)`, compute where it projects on each detector, `s = wx·cosθ + wy·sinθ →`
fractional bin, **linearly interpolate** the residual there, and average over
angles.

**Complexity.** Let `N²` pixels, `A` angles, `D` detectors, `T` iterations, `S ≈
2N` ray steps.
- Forward: `O(A·D·S) = O(A·D·N)` per iteration.
- Backproject: `O(N²·A)` per iteration.
- Total serial: `O(T·A·(D·N + N²))`. For our sample (T=20, A=15, D=96, N=64) that
  is a few million float ops — milliseconds. A clinical volume (N~800, 3-D, A~15,
  T~10) is **10¹¹–10¹² ops**, which is why the GPU matters.

**Data-access pattern.** Both stages are **gathers**: forward reads many image
pixels and writes one `sim` value per ray; backproject reads many residual samples
and writes one image pixel. No output element is written by more than one
producer → naturally parallel, no atomics.

## 4. The GPU mapping

Each SART stage maps to one kernel; the image and scratch buffers stay
**device-resident** across all `T` iterations, so there is **zero per-iteration
PCIe traffic** (we upload `b` + angles once, download the final image once).

**Kernel 1 — `forward_project_kernel` (1-D grid, thread per ray).**
- Thread-to-data: global index `r = blockIdx.x·blockDim.x + threadIdx.x`; decode
  `k = r / n_det`, `j = r % n_det`. Guard `r ≥ n_angles·n_det`.
- Block = 256 threads (a multiple of the 32-lane warp; 8 warps to hide the memory
  latency of the ray march). Grid = `⌈A·D / 256⌉`.
- Memory: reads the image estimate along the ray (global memory, via
  `bilinear_sample`), reads the cos/sin tables; writes one `sim[r]`. Registers hold
  the running accumulator. It is **memory-bandwidth-bound**, the classic profile of
  a projector.

**Kernel 2 — `backproject_update_kernel` (2-D grid, thread per pixel).**
- Thread-to-data: `px = blockIdx.x·blockDim.x + threadIdx.x`, `py = …y…`; pixel
  `(px,py)`. Guard `px,py ≥ N`.
- Block = 16×16 = 256 threads: a square tile matching the 2-D image, so adjacent
  threads in `x` write adjacent `image[]` cells (**coalesced** stores). Grid =
  `⌈N/16⌉ × ⌈N/16⌉`.
- Memory: gathers `residual` from every angle (global), writes one `image` pixel in
  place. No shared memory / atomics needed because each pixel has one owner.

**Kernel 3 — `residual_kernel`** is a trivial element-wise `b − sim` so the whole
iteration stays on the device.

```
  measured b (device)        image estimate x (device)
        |                            |
        v                            v
  +-------------------- SART iteration (all on GPU) --------------------+
  | forward_project_kernel   :  1-D grid, thread per RAY   ->  sim      |
  | residual_kernel          :  1-D grid, thread per elem  ->  res=b-sim|
  | backproject_update_kernel:  2-D grid, thread per PIXEL ->  x += ...  |
  +--------------------------------------------------------------------+
        (repeat T times; only the final x is copied back to host)

   forward: gather image along a ray          backproject: gather residual over angles
     detector bin j                                  pixel (px,py)
        \  |  /   rays sample image                   /  |  \  angles project to detector
         \ | /                                       /   |   \
   [======image======]                        [====residual rows (per angle)====]
```

**Where the catalog's other accelerations fit.** The catalog envisions cuFFT (for a
ramp-filter FBP initializer), texture memory (hardware bilinear interpolation via
`tex2D`), constant memory (the small angle/geometry tables, broadcast to all
threads), and an ADMM/TV inner loop. This build keeps the interpolation in software
and the geometry in plain arrays so the math is fully legible; wiring a texture
(Exercise 3) or constant memory is a small, well-scoped change — **no black boxes**.

## 5. Numerical considerations

- **Precision: FP32.** Attenuation reconstruction is not stiff; single precision is
  what production GPU projectors use, and it halves bandwidth (the bottleneck). The
  accumulators are `float`. Exercise 5 explores FP64.
- **Determinism.** Both projection kernels are **gathers**: every output element is
  written by exactly one thread, and each thread's internal sum runs in a fixed loop
  order. So there are **no atomic float accumulations** whose (nondeterministic)
  order would perturb the result (docs/PATTERNS.md §3). The stdout is therefore
  **byte-identical across runs** — verified by running three times and diffing.
- **CPU/GPU parity by construction.** The per-ray forward integral and the bilinear
  sampler live in **one shared `__host__ __device__` header** (`dbt_geometry.h`),
  and the cos/sin tables are precomputed **once** in double then stored as float and
  read by both sides. So the CPU reference and the GPU kernel execute the *same*
  float operations; the only divergence is the GPU's fused multiply-add (FMA)
  contraction vs. the host compiler — a few ULP.
- **Stability / convergence.** SART converges for `0 < λ ≤ 2`; small λ (0.3 here) is
  slow but robust and avoids the salt-and-pepper oscillation large λ causes on
  limited-angle data. The non-negativity clamp is both physical (μ ≥ 0) and a mild
  regularizer.

## 6. How we verify correctness

Two independent checks:

1. **GPU vs. CPU agreement.** `src/reference_cpu.cpp` runs the *entire* SART
   pipeline serially. `main.cu` runs both and reports `max_abs_err`. Because both
   implementations share `dbt_geometry.h`, agreement is exact up to float rounding,
   so we use an **absolute tolerance of `1e-3`** (docs/PATTERNS.md §4: the "long
   iterative float / FMA divergence" class, same as flagship 4.01). Observed error
   is ~`2e-7` — three orders of magnitude inside tolerance. An independent serial
   implementation agreeing to ULP-level with the parallel one is strong evidence the
   parallelization introduced no logic bug.
2. **Science check (recovering a known phantom).** The synthetic data is the
   forward projection of a phantom with **two lesions at known locations**. A
   correct reconstruction must place its intensity peak on a lesion — and it does:
   the reported peak sits at pixel `(23, 31)`, i.e. world `x ≈ −0.27, y ≈ 0`, which
   is planted lesion 1. That validates the *geometry and physics*, not just
   CPU==GPU.

Edge cases handled: rays that step outside the image (clamped / skipped), the ragged
last thread block (guarded), and a malformed/short data file (the loader throws).

## 7. Where this sits in the real world

Production DBT reconstruction (ASTRA, RTK, TIGRE, vendor pipelines) differs by:

- **True 3-D fan/cone-beam geometry** with a divergent source, a specific
  source-to-detector distance, a tilting detector, and per-slice focal planes —
  versus our 2-D parallel-beam wedge. The *algorithm* is the same; the ray geometry
  and system matrix are more elaborate.
- **Statistical / regularized reconstruction.** Real systems use **OS-EM**
  (ordered-subsets expectation-maximization with a Poisson noise model),
  **ASD-POCS** (SART interleaved with **total-variation minimization** to suppress
  the missing-wedge streaks), or full **MBIR** (a penalized-likelihood optimization).
  SART is the didactic ancestor of all of these.
- **Physics corrections**: scatter, beam hardening, detector PSF/MTF modeling, and
  anti-scatter grids — omitted here.
- **Deep learning** post-processing (U-Net denoising of an FBP initializer) and mass
  detection CNNs, all GPU inference.
- **Scale**: ~800×700×60 voxels at 85 µm, ~30 GB of raw projections — the regime
  where GPU acceleration turns hours into seconds, exactly the motivation the
  catalog cites.

Our version is a faithful, fully-legible **reduced-scope teaching model**: the same
forward/back-project/update loop, the same limited-angle ill-posedness, minus the
production geometry, noise model, and regularizers.

---

## References

- Andersen & Kak (1984), *Simultaneous Algebraic Reconstruction Technique (SART): a
  superior implementation of the ART algorithm*, Ultrasonic Imaging — the source of
  the method implemented here.
- Sidky & Pan (2008), *Image reconstruction in circular cone-beam CT by constrained,
  total-variation minimization* — ASD-POCS, the standard limited-angle upgrade.
- Wu et al. (2003), *Tomographic mammography using a limited number of low-dose
  cone-beam projection images* — foundational DBT reconstruction paper.
- **ASTRA Toolbox** (github.com/astra-toolbox) — study its GPU forward/back-projector
  for arbitrary geometry and its texture-based interpolation.
- **RTK** (github.com/RTKConsortium/RTK) — a full iterative pipeline (SART/OS-EM +
  regularizers) at clinical scale.
- **TIGRE** (github.com/CERN/TIGRE) — DBT-compatible cone-beam geometry and iterative
  solvers; good for comparing convergence behavior.
- Kak & Slaney, *Principles of Computerized Tomographic Imaging* — the Radon
  transform, Fourier-slice theorem, and FBP background for §2.
