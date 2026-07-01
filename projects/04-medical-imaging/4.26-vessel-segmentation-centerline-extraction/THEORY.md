# THEORY — 4.26 Vessel Segmentation & Centerline Extraction

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

A **CT angiogram (CTA)** is a 3-D X-ray image acquired after a radio-opaque
contrast agent is injected into the bloodstream. The contrast makes vessels
**bright** relative to surrounding soft tissue. Clinicians and planning software
need the vessel tree as an explicit object: a **segmentation** (which voxels are
vessel) and a **centerline** (the 1-D skeleton with a radius at each point). These
feed, for example, coronary **FFR-CT** (a simulated fractional flow reserve that
estimates whether a narrowing is flow-limiting) and **aortic endograft planning**
(sizing a stent to a patient's anatomy).

The core visual fact we exploit: **a vessel is locally a tube.** Walk along the
vessel and the intensity barely changes; step across it and the intensity rises to
a bright core and falls off — a ridge. A blob (e.g. a contrast-filled chamber)
brightens in *all* directions; a flat sheet (an organ boundary) brightens in only
*one*. So "tube-ness" is a statement about the **local second-order shape** of the
intensity in the three principal directions — which is exactly what the Hessian
and its eigenvalues encode. Frangi *et al.* (1998) turned this observation into the
**vesselness filter** used across medical imaging.

## 2. The math

Let `I(x)` be the (smoothed) image intensity at voxel position `x = (x,y,z)`. The
**Hessian** is the 3×3 matrix of second partial derivatives:

```
      | I_xx  I_xy  I_xz |
H  =  | I_xy  I_yy  I_yz |          (symmetric: I_xy = I_yx, etc.)
      | I_xz  I_yz  I_zz |
```

`H` is real and symmetric, so it has three **real eigenvalues** `λ1, λ2, λ3` with
orthogonal eigenvectors. Order them by **magnitude**: `|λ1| ≤ |λ2| ≤ |λ3|`. The
eigenvectors are the local principal directions; the eigenvalues are the intensity
curvature along them. For a **bright tube on a dark background**:

- along the vessel axis, `I` is nearly constant → `|λ1| ≈ 0`;
- across the vessel, `I` is strongly concave (bright core) → `λ2, λ3` **large and
  negative**.

Frangi builds three dimensionless quantities (units: `λ` are intensity per voxel²;
the ratios are unitless):

```
R_A = |λ2| / |λ3|                 in [0,1]  — distinguishes line (≈1) from plate (≈0)
R_B = |λ1| / sqrt(|λ2·λ3|)                  — distinguishes line (≈0) from blob (≈1)
S   = sqrt(λ1² + λ2² + λ3²)                 — "structureness" (Frobenius norm of H)
```

and combines them into the **vesselness** `V ∈ [0,1]`:

```
       0                                             if λ2>0 or λ3>0   (wrong polarity)
V  =  (1 - exp(-R_A²/2α²)) · exp(-R_B²/2β²) · (1 - exp(-S²/2c²))   otherwise
```

Reading the three factors: the first `→1` when the structure is a line (`R_A→1`);
the second `→1` when it is *not* a blob (`R_B→0`); the third `→1` only where there
is real structure (`S` large), which suppresses response in flat, noisy regions.
`α, β` are fixed (0.5); `c` is a scale set to roughly half the maximum Hessian norm
(here tuned to the synthetic intensity range — see §5 and `data/README.md`).

**Inputs:** a scalar volume + `(σ, α, β, c, polarity)`. **Output:** a vesselness
volume `V` in `[0,1]`; a threshold on `V` yields a binary segmentation.

## 3. The algorithm

Single-scale Frangi over an `N = nx·ny·nz` voxel volume:

1. **Gaussian smooth** at scale `σ` (separable): 3 one-dimensional convolutions,
   `O(N · r)` with kernel radius `r ≈ 3σ`. Smoothing both denoises and sets the
   vessel radius the filter is tuned to.
2. **Per voxel** (`O(N)` independent work):
   a. **Hessian** by central finite differences — 6 unique entries from the 3×3×3
      neighbourhood (`I_xx = I(x+1)-2I(x)+I(x-1)`, mixed terms from the 4 diagonal
      neighbours / 4).
   b. **Eigenvalues** of the symmetric 3×3 `H` in **closed form** (below).
   c. **Frangi score** from the sorted eigenvalues.
3. **Reduce** to a segmentation (threshold) and a centerline **seed** (argmax).

**Closed-form symmetric-3×3 eigenvalues (the interesting kernel).** The
characteristic polynomial `det(H - λI) = 0` is a cubic. For a *symmetric* matrix
all three roots are real, so Cardano's formula lands in the "trigonometric"
(casus irreducibilis) branch (Smith 1961):

```
q   = trace(H)/3
p   = sqrt( (Σ (H-qI)_ij²) / 6 )
B   = (H - qI)/p                         (normalized; det(B) ∈ [-2,2])
φ   = acos( clamp(det(B)/2, -1, 1) ) / 3
λ   = q + 2p·cos(φ + k·2π/3),  k=0,1,2
```

This is ~40 flops, branch-light, and — crucially — the **same** flops on CPU and
GPU (see §5). We chose it over the catalog-suggested **Jacobi iteration** because
(a) Jacobi's data-dependent iteration count makes exact CPU/GPU parity fragile, and
(b) we only need eigen*values* (a scalar score), not eigen*vectors*. Jacobi is the
right tool when you also want the vessel *direction* — that is a README exercise.

**Complexity.** Serial: `O(N)` with a small constant (finite differences + a cubic
solve per voxel). Parallel: **work** `O(N)`, **depth** `O(1)` — every voxel is
independent, so on `P` cores the time is `O(N/P)`. This embarrassingly-parallel
structure is why the GPU wins at clinical volume sizes.

## 4. The GPU mapping

**Pattern: map (one thread per voxel).** The Hessian at a voxel reads only its
3×3×3 neighbourhood and writes one output, with no dependence on any other voxel's
output — so there is nothing to synchronize.

- **Thread-to-data map:** thread `(x,y,z) = (blockIdx·blockDim + threadIdx)` owns
  output voxel `(x,y,z)`; it writes `V[vox_idx(x,y,z)]`.
- **Launch config:** blocks of `8×8×4 = 256` threads (a multiple of the 32-lane
  warp; 3-D to match the volume; thin in `z` because volumes are thin in `z`). The
  grid rounds each axis up: `grid = (⌈nx/8⌉, ⌈ny/8⌉, ⌈nz/4⌉)`. A guard
  `if (x≥nx||y≥ny||z≥nz) return;` kills the ragged edge threads.
- **Memory hierarchy:** inputs live in **global** memory; the layout is `x`-fastest
  so a warp walking `x` reads a **coalesced** cache line. The Hessian's temporaries
  and the eigen solve live entirely in **registers**. No shared memory, no
  constant memory, no atomics in the teaching kernel.
- **The obvious optimization (why we didn't):** the 27-point stencil re-reads each
  neighbour up to 27× from global memory. A tiled kernel would stage a block's
  tile+halo into **shared** memory once, then compute from there — the classic
  bandwidth win. We keep the simple version for legibility and leave tiling as an
  exercise (README ex. 3).
- **No CUDA library here on purpose.** The catalog lists cuDNN (for a U-Net) and
  Thrust (for a fast-marching priority queue). This project deliberately
  hand-rolls the eigendecomposition — a library eigensolver (cuSOLVER) is built
  for *large dense* matrices, not for `10⁷` independent `3×3`s, where the
  closed-form inline function is far faster and teaches the actual math.

```
   3-D volume (nx x ny x nz)            3-D grid of 8x8x4 blocks
   ┌───────────────┐                    ┌────┬────┬────┐
   │ . . . . . . . │  one thread  ==>   │ B  │ B  │ B  │   each block: 256 threads
   │ . [x,y,z] . . │  per voxel         ├────┼────┼────┤   each thread: 1 voxel ->
   │ . . . . . . . │                    │ B  │ B  │ B  │     read 3x3x3 nbhd,
   └───────────────┘                    └────┴────┴────┘     write 1 vesselness
```

## 5. Numerical considerations

- **Precision.** Intensities are stored `float` (imaging data is ~12-bit anyway),
  but the **Hessian, eigen solve, and Frangi score run in `double`**. The cubic
  solve involves `acos`/`cos` and subtractions of nearby quantities; double
  protects against catastrophic cancellation in `det(B)` and keeps the three roots
  well separated.
- **Robustness guards.** We `clamp(det(B)/2, -1, 1)` before `acos` (round-off can
  push it a hair outside the domain), special-case the diagonal matrix `p==0`
  (→ triple root `q`), and add tiny epsilons under the square roots in `R_B` and to
  the `|λ3|` denominator so a perfectly flat voxel scores 0 instead of `NaN`.
- **Determinism.** There are **no atomics and no parallel float reductions** — each
  thread writes its own voxel, so there is no order-dependent summation. The final
  scalar summaries in `main.cu` are computed by a single fixed-order host scan
  (first-wins tie-break on the argmax), so **stdout is byte-identical every run**
  (docs/PATTERNS.md §3). The reported checksum is `round(Σ V · 1000)` — an integer,
  immune to the last-bit float wobble.
- **Why CPU == GPU exactly here.** Both paths smooth on the host (identical data),
  then call the **same** `frangi.h` functions. The only possible divergence is the
  GPU fusing a multiply-add (FMA) where the host does not, a ~1e-16 effect per op;
  over the handful of ops per voxel it stays far below our `1e-6` tolerance — and
  on this sample the observed max difference is literally **`0.000e+00`**. Contrast
  this with long iterative solvers (docs/PATTERNS.md §4), where FMA drift
  accumulates and a physical `1e-3` tolerance is the honest choice.
- **Honest edge effect.** Finite differences with clamp-to-edge borders slightly
  inflate the response at the volume faces; in the demo that puts the global peak
  at the `x = 23` boundary. This is a real property of the discretization, not a
  bug (see README "Limitations").

## 6. How we verify correctness

The trusted baseline is `src/reference_cpu.cpp`: an obviously-correct triple loop
with no parallelism. `main.cu` runs it and the GPU kernel on the same smoothed
volume and takes the **max absolute difference** over all voxels of the two
vesselness fields.

- **Tolerance:** `1e-6` (docs/PATTERNS.md §4, "same exact operations on both
  sides"). We could assert exact equality on this input, but `1e-6` documents the
  FMA caveat honestly while still being far tighter than any meaningful change in
  the score.
- **Why this is convincing:** the CPU and GPU implementations were written to be
  structurally different (host loop vs. one thread per voxel, different memory
  order) yet share only the pure-math header. Independent code paths converging on
  the same answer is strong evidence the math and the indexing are right.
- **Second, physical check:** the demo also confirms the *science* — the
  across-vessel profile is a clean single peak centred on the tube, and the
  segmented voxel count (96) is consistent with a radius-2 tube of length ~24. A
  correct-but-meaningless number would not localize the vessel.
- **Edge cases exercised:** border voxels (clamp handling), the flat background
  (score 0, no `NaN`), and the sign gate (bright-on-dark polarity).

## 7. Where this sits in the real world

This is the **classic hand-crafted filter**, one stage of a modern pipeline:

- **Multi-scale.** Real Frangi runs at several `σ` and takes the per-voxel max, so
  it catches 1 mm and 8 mm vessels alike. We show a single scale.
- **Learned segmentation.** State-of-the-art coronary/organ vessel segmentation
  uses 3-D CNNs — **U-Net / V-Net** via **MONAI**, or detection framings like
  **nnDetection** — often with the vesselness map as an input channel. Inference is
  GPU-bound (cuDNN), which is the catalog's other GPU angle.
- **Centerlines & topology.** **VMTK** turns a segmentation into a centerline
  **graph** with per-point radius using level-set / fast-marching methods; the
  catalog notes GPU fast-marching via parallel priority queues (Thrust). Our
  "centerline" is only the peak-response **seed** — enough to demonstrate the idea.
- **Clinical use.** Downstream, coronary centerlines drive **FFR-CT** CFD and stent
  planning. None of that is claimed here — this is a teaching filter on synthetic
  data, **not for clinical use**.

---

## References

- **A. F. Frangi, W. Niessen, K. Vincken, M. Viergever (1998),** "Multiscale
  vessel enhancement filtering," *MICCAI*. The original vesselness filter — the
  three ratios and the combination formula implemented here.
- **Y. Sato *et al.* (1998),** "Three-dimensional multi-scale line filter…"
  *Medical Image Analysis*. The contemporaneous line-filter; an alternative
  eigenvalue combination worth comparing.
- **O. K. Smith (1961),** "Eigenvalues of a symmetric 3×3 matrix," *Comm. ACM*.
  The closed-form trigonometric eigenvalue formula used in `frangi.h`.
- **VMTK** — <https://github.com/vmtk/vmtk>. How production centerline extraction
  and vascular meshing work end to end.
- **MONAI** — <https://github.com/Project-MONAI/MONAI>. Modern learned 3-D vessel
  segmentation networks; the deep-learning counterpart to this filter.
- **nnDetection** — <https://github.com/MIC-DKFZ/nnDetection>. GPU object detection
  for tubular structures.
