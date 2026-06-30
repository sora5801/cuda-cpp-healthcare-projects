# THEORY — 4.28 GPU-Accelerated DRR Generation for 2D/3D Registration

> The deep "why". Read this alongside the heavily-commented source. The per-ray
> physics lives in `src/drr_core.h` (shared by CPU and GPU); the GPU mapping is in
> `src/kernels.cu`; the trusted serial baseline is `src/reference_cpu.cpp`.
>
> _Educational only — not for clinical use._

---

## 1. The science — what a DRR is and why we make one

In image-guided radiotherapy, a patient is scanned once on a planning **CT**
(a 3-D map of tissue density). On each treatment day the patient is positioned and
a quick 2-D X-ray (a kV image or an MV **portal** image) is taken. To deliver the
dose where it was planned, the daily X-ray must be aligned to the planning CT —
this is **2D/3D registration**: find the rigid patient pose (3 rotations + 3
translations) under which a *simulated* X-ray of the CT best matches the *real*
X-ray.

The simulated X-ray is the **Digitally Reconstructed Radiograph (DRR)**. It is
literally "what an X-ray of this CT volume, taken from this source/detector pose,
would look like." Generating DRRs is the inner loop of registration: an optimizer
perturbs the pose, re-renders the DRR, scores its similarity to the real image, and
repeats — **50–200 DRRs per iteration**, dozens of iterations. DRR generation is
therefore the computational bottleneck, and it is exactly the kind of
embarrassingly-parallel gather a GPU eats for breakfast.

The physics is **X-ray attenuation**. A monoenergetic beam of intensity `I0`
passing through material loses intensity exponentially with how much attenuating
matter it crosses (the **Beer–Lambert law**):

```
I = I0 · exp( − ∫ μ(s) ds )
```

where `μ(s)` is the **linear attenuation coefficient** (units 1/mm) along the ray.
A DRR pixel is that line integral `∫ μ ds` (or `exp(−∫μ ds)`); we report the
integral itself because intensity-based registration compares it consistently on
both the DRR and the real image.

CT volumes store **Hounsfield Units (HU)**, a normalized density scale
(water = 0 HU, air = −1000 HU). We convert HU → μ once at load time:

```
HU = 1000 · (μ − μ_water) / μ_water   ⟹   μ = μ_water · (1 + HU/1000)
```

with `μ_water ≈ 0.019 /mm` at a ~70 keV effective beam energy (see `hu_to_mu()`).

---

## 2. The math — the ray integral and its discretization

### Geometry (cone beam)

A point X-ray **source** `S` and a flat **detector** panel. Detector pixel `(u, v)`
sits at world position

```
D(u,v) = origin + u·du + v·dv
```

where `origin` is the panel's pixel-(0,0) corner and `du`, `dv` are the per-column
and per-row edge vectors (mm). The ray for that pixel runs from `S` to `D(u,v)`;
its unit direction is `r̂ = (D − S)/‖D − S‖`.

### The DRR pixel

```
DRR(u,v) = ∫₀ᴸ μ( S + s·r̂ ) ds ,    L = ‖D − S‖
```

We approximate the integral by **ray-marching**: step along the ray in fixed
increments `Δ = step_mm`, sampling μ at each step's midpoint and summing rectangles
(the **midpoint rule**):

```
DRR(u,v) ≈ Σ_{i=0}^{n−1} μ( S + (i+0.5)·Δ · r̂ ) · Δ ,   n = ⌊L/Δ⌋
```

### Sampling μ between voxels — tri-linear interpolation

The march point `p` almost never lands on a voxel center, so we **tri-linearly
interpolate** the 8 surrounding voxels. With fractional voxel coords
`(fx,fy,fz) = (p.x/sx, p.y/sy, p.z/sz)`, lower corner
`(i,j,k) = ⌊(fx,fy,fz)⌋`, and in-cell fractions `(tx,ty,tz)`:

```
μ(p) = Σ_{a,b,c∈{0,1}} c_{i+a, j+b, k+c} · w_a(tx) · w_b(ty) · w_c(tz)
       where  w_0(t)=1−t,  w_1(t)=t
```

Voxels outside the grid are treated as air (μ = 0), which also makes rays that
graze the volume edge safe. This is `sample_trilinear()` in `drr_core.h`.

### Complexity

Per pixel: `O(n)` steps, each an `O(1)` tri-linear fetch (8 reads). Total for a
`W×H` panel with `n` steps per ray: **`O(W·H·n)`**. For the catalog's headline case
(400×400 DRR, 512³ CT) that is ~6.4×10⁸ interpolations *per DRR*, ×(50–200 DRRs)
×(iterations) ≈ 10¹¹ operations — the motivation for the GPU.

---

## 3. The algorithm — step by step

```
load_volume(path):                # reference_cpu.cpp
    read nx,ny,nz,sx,sy,sz; read HU body; mu[i] = hu_to_mu(HU[i])   # convert once

make_demo_geometry(volume, W, H, delta):
    place point source on -x side, flat W x H detector on +x side, both centered;
    size the panel to cover the volume's y/z extent with a margin

render(volume, geometry):         # the parallel part
    for each detector pixel (u,v):                # INDEPENDENT  -> one GPU thread
        build ray S -> D(u,v);  L = ||D-S||;  n = floor(L/delta)
        acc = 0
        for i in 0..n-1:
            p   = S + (i+0.5)*delta*rhat
            acc += sample_trilinear(mu, p) * delta     # tri-linear gather
        DRR[v][u] = acc
```

CPU reference (`render_drr_cpu`) runs the double loop serially; the GPU kernel
(`drr_kernel`) runs the body of one `(u,v)` per thread. Both call the **identical**
`integrate_ray()` — that shared core is what makes verification exact.

This is the **"gather" pattern** (docs/PATTERNS.md §1, flagship 4.01 CT
backprojection): each output element independently *reads* (gathers) from a shared
read-only input. No atomics, no shared memory, no inter-thread communication.

---

## 4. The GPU mapping — threads, blocks, memory

### Thread-to-data mapping

A 2-D grid of `16×16` thread blocks tiles the `W×H` detector panel. Thread
`(blockIdx, threadIdx)` owns one pixel:

```
u    = blockIdx.x · blockDim.x + threadIdx.x      # detector column
vrow = blockIdx.y · blockDim.y + threadIdx.y      # detector row
```

Threads in the ragged edge tiles (`u ≥ W` or `vrow ≥ H`) return immediately. A
`16×16 = 256`-thread block gives good occupancy on `sm_75…sm_89`. A *square* tile is
deliberate: neighbouring threads cast nearby rays that sample nearby voxels, so
their memory accesses cluster — friendly to the L1/L2 caches (and ideal for a 3-D
texture; see below).

### Memory hierarchy

- **Global memory** holds the attenuation volume `d_mu` (`nx·ny·nz` floats) — the
  one large upload, read by every thread. `__restrict__` tells the compiler `d_mu`
  and `d_img` don't alias, so the running integral stays in a **register**.
- **Registers** hold the per-thread state (ray origin/direction, the accumulator,
  the step index) — tiny and fast.
- **Constant/parameter space**: `VolumeDesc` and `DrrGeometry` are small PODs
  passed *by value*; CUDA places kernel parameters in constant-banked memory, so
  every thread reads the geometry without a global fetch.
- **No shared memory and no atomics** — each thread writes exactly one output
  pixel that no other thread touches.

### The texture-memory upgrade (the "right" way, kept as a comment)

A production DRR engine binds the volume to a **CUDA 3-D texture** and replaces
`sample_trilinear()` with a single `tex3D<float>(tex, x, y, z)` call. The texture
units then perform the tri-linear interpolation **in hardware, essentially for
free**, and a 3-D-locality texture cache services the 8-voxel neighbourhood each
sample needs. That is the single biggest real-world speedup for DRR, and it is why
the catalog flags "CUDA 3-D texture with hardware tri-linear interpolation
(zero-cost)" as *the* pattern here. We deliberately do the interpolation in plain
device code so every multiply and add is visible to the learner; swapping in a
texture is **Exercise 1** in the README.

### Why the GPU wins here

The work is `W·H·n` *independent* fused-multiply-adds against a read-mostly volume.
A GPU runs thousands of these rays concurrently and hides the volume's memory
latency behind arithmetic. In the demo (128×128 panel, ~600 steps/ray) the kernel
is ~0.6 ms vs ~170–240 ms on one CPU core — and the gap widens with panel size and
the many DRRs per registration iteration. (Timing is a *teaching artifact*, never a
benchmark — CLAUDE.md §12.)

---

## 5. Numerical considerations

- **Precision (FP32).** μ values are O(0.02 /mm) and a ray crosses ~tens of mm, so
  a DRR pixel is O(1–10) and float has ample precision. We use `float` throughout
  to match how real-time DRR engines run (and how a texture returns samples).
- **Determinism.** Every ray takes a *data-independent* number of steps
  `n = ⌊L/Δ⌋`, computed identically on CPU and GPU, and the tri-linear blend
  evaluates the *same float operations in the same order* on both sides (the blend
  order is fixed in `sample_trilinear`). So the per-pixel sums are reproducible run
  to run, and the GPU's stdout is byte-identical (PATTERNS.md §3). There are **no
  atomics**, so none of the float-reordering nondeterminism that tallies/reductions
  suffer from applies here.
- **CPU vs GPU difference.** The only divergence is the GPU's fused-multiply-add
  (FMA) contraction vs the host compiler's separate multiply/add. Over a few
  hundred summed terms this is ~1e-6 absolute — far below any visible difference.
- **Quadrature error.** The midpoint rule with step `Δ` has error `O(Δ²)` per step;
  halving `Δ` quarters it but doubles the work. `Δ = 1 mm` on a 2 mm-voxel volume is
  a sensible teaching default (sub-voxel sampling). Aliasing appears if `Δ` is much
  larger than the voxel size (you skip past thin structures) — Exercise 3.

## 6. How we verify correctness

1. **CPU == GPU (primary).** `main.cu` renders the DRR on both paths and reports
   `max_abs_err`. Because both call the shared `integrate_ray()`, agreement is
   exact up to FP32 rounding; we accept `≤ 1e-3` (the demo achieves ~8e-7). This
   tolerance is the honest "FMA over a few-hundred-term sum" budget (PATTERNS.md
   §4), not a fudge factor.
2. **Physical sanity (secondary, science-level).** The synthetic phantom embeds a
   dense bone sphere **offset** from center, so the brightest DRR pixel *must* land
   off-axis in the offset direction. The program prints `max attenuation at (u,v)`;
   the demo reports `(76,61)` — right of the 64-px midline, exactly the +y offset.
   The central-row profile is 0 in the air, rises through soft tissue, and peaks at
   the bone, matching the phantom. This validates the *geometry*, not just CPU==GPU.
3. **Determinism check.** Running the program repeatedly yields identical stdout;
   `demo/run_demo` diffs it against `expected_output.txt`.

## 7. Where this sits in the real world

Real DRR / 2D/3D registration tools differ from this teaching version in several
ways, all described so nothing is a black box:

- **Texture hardware** for the interpolation (see §4) — the dominant production
  optimization.
- **Siddon's exact ray–box traversal** instead of fixed-step marching. Siddon
  computes the *exact* intersection lengths of the ray with each voxel it crosses,
  giving an exact line integral with no quadrature error and no oversampling — at
  the cost of a more complex per-voxel loop. Fixed-step ray-marching (used here) is
  simpler to read and maps cleanly to texture sampling, which is why GPU engines
  often prefer it. Both are in the catalog's "Key algorithms".
- **Similarity metrics + optimizer.** Registration wraps DRR generation in a
  similarity score — **normalized cross-correlation (NCC)**, **mutual information
  (MI)**, or **gradient-magnitude** similarity — and a derivative-free or
  stochastic-gradient optimizer over the 6-DOF pose. We render one fixed pose;
  the full loop is Exercise 5.
- **Differentiable DRR (DiffDRR).** Modern work makes the DRR *differentiable* in
  the pose, so gradient-based optimization (and deep-learning init / "neural DRR")
  can drive registration far faster. See the catalog's DiffDRR reference.
- **Production tools.** **Plastimatch** and **RTK** provide GPU DRR/ray-casting;
  **DiffDRR** provides differentiable DRRs; `CUDA_DigitallyReconstructedRadiographs`
  is a compact GPU DRR library. Study these for the texture path, the geometry
  conventions, and the registration wrappers — reimplement didactically, don't
  copy.

---

### Cross-references

- Code tour and build/run steps: [`README.md`](README.md).
- Shared per-ray physics: [`src/drr_core.h`](src/drr_core.h).
- GPU kernel + launch reasoning: [`src/kernels.cu`](src/kernels.cu).
- The "gather" pattern and its exemplar (4.01): `docs/PATTERNS.md`.
