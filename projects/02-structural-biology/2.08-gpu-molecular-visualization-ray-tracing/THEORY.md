# THEORY — 2.8 GPU Molecular Visualization & Ray Tracing

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See
> [README.md](README.md) for the quick tour and build steps.
>
> _Educational only — not for clinical use. The molecule rendered here is
> **synthetic**, not a real structure._

---

## 1. The science

Structural biologists spend their lives **looking at molecules**. A protein
solved by X-ray crystallography or cryo-EM is, computationally, just a list of
atoms — each an element at a 3-D coordinate. To reason about it (where is the
binding pocket? does this mutation clash?) a scientist needs a **picture**, and
the most literal picture is the **space-filling / van-der-Waals (VDW)**
representation: draw every atom as a solid sphere whose radius is that element's
VDW radius (carbon ≈ 1.7 Å, oxygen ≈ 1.52 Å, …). Where spheres overlap, you see
the molecule's *surface* — its actual physical bulk.

A flat, evenly-lit sphere image is hard to read: without shadows your eye cannot
tell a bump from a dent. The trick that makes molecular renders legible is
**ambient occlusion (AO)**: darken each surface point in proportion to how much
of the surrounding "sky" is blocked by other atoms. Crevices between atoms get
little sky and go dark; exposed tops stay bright. AO is what gives VMD's and
PyMOL's ray-traced images their sculpted, obviously-3-D look — and it is exactly
what this project computes, plus a single directional light with a hard shadow.

This is **production-relevant**: VMD ray-traces multi-million-atom systems at
interactive rates using CUDA and NVIDIA OptiX; the per-ray math it runs is the
same ray/sphere intersection and AO sampling we implement here, just accelerated
by a hardware bounding-volume hierarchy. We render a small synthetic molecule so
the geometry and the GPU mapping are crystal clear.

## 2. The math

**Inputs.** A scene of `n` atoms; atom `i` has centre `cᵢ ∈ ℝ³` (Å), VDW radius
`rᵢ > 0` (Å), and a colour id. A camera defines, per pixel `(px,py)`, a **ray**
`R(t) = o + t·d`, with origin `o ∈ ℝ³`, unit direction `d`, and `t ≥ 0` the
distance along it. We use an **orthographic** camera looking down `−z`: every
pixel's ray has the same direction `d = (0,0,−1)` and an origin obtained by
linearly mapping the pixel grid onto the image-plane rectangle.

**Ray/sphere intersection.** A point `R(t)` lies on sphere `i` when
`|o + t·d − cᵢ|² = rᵢ²`. With `d` a unit vector this is a quadratic in `t`:

```
t² + 2b t + c = 0,   where  b = d·(o − cᵢ),   c = |o − cᵢ|² − rᵢ².
```

The discriminant is `Δ = b² − c`. If `Δ < 0` the ray misses. Otherwise the two
roots are `t = −b ∓ √Δ`; the nearest hit in front of the origin is the smaller
positive root. The **nearest** atom along the ray is `argminᵢ tᵢ`.

**Surface normal.** At a hit point `p = o + t·d` on sphere `i`, the outward unit
normal is `n = (p − cᵢ)/|p − cᵢ|`. Lighting depends only on `n`.

**Shading.** Let `ℓ` be the unit direction toward a directional light. The
**Lambert** (diffuse) term is `max(0, n·ℓ)`, set to 0 if a **shadow ray** from
`p` toward `ℓ` hits any atom (the point is in shadow). The **ambient-occlusion**
factor is

```
ao = (1/N) · Σ_{k=1..N} [ hemisphere ray k from p escapes within radius ρ ],
```

i.e. the fraction of `N` rays cast into the hemisphere about `n` that are *not*
blocked by another atom within distance `ρ`. The final pixel luminance is

```
L = base(colour) · ( a·ao + (1−a)·diffuse·ao + 0.15·ao ),   clamped to [0,1],
```

with ambient floor `a` (here 0.45). Finally we **quantize** to a byte:
`q = round(255·L) ∈ {0,…,255}`.

## 3. The algorithm

```
for each pixel (px,py):                      # W·H pixels, all independent
    R   = primary_ray(camera, px, py)
    t,i = trace_nearest(R, atoms)            # O(n): nearest sphere hit
    if no hit: image[px,py] = background; continue
    p   = R(t);  n = normalize(p − c_i)
    # ambient occlusion: N any-hit rays into the hemisphere about n
    esc = 0
    for k in 1..N:  esc += not occluded(p, hemisphere_dir(n,k,N), ρ, atoms)   # O(n) each
    ao  = esc / N
    # direct light + hard shadow: one any-hit ray toward the light
    dif = max(0, n·ℓ);  if occluded(p, ℓ, ∞, atoms): dif = 0                   # O(n)
    image[px,py] = quantize( base(colour_i) · (a·ao + (1−a)·dif·ao + 0.15·ao) )
```

**Complexity.** Each pixel casts `1 + N + 1` rays, and our brute-force
`trace_nearest`/`occluded` test every atom, so the cost is
`Θ(W·H·(N+2)·n)`. For the demo (`W=H=200`, `N=32`, `n=77`) that is ≈
`200·200·34·77 ≈ 1.05·10⁸` ray/sphere tests — about 100 million, in a fraction
of a second on a GPU. The **work** is fully parallel across the `W·H` pixels;
the **depth** (critical path) is just the per-pixel loop, `Θ((N+2)·n)`.

The brute-force `Θ(n)` per ray is the teaching-friendly choice. Production
renderers replace it with a **bounding-volume hierarchy (BVH)** so each ray
touches `Θ(log n)` atoms — essential at millions of atoms, and exactly what
hardware ray tracing accelerates (see §7).

## 4. The GPU mapping

This is the **per-output-pixel gather** pattern (docs/PATTERNS.md §1), the same
shape as flagship 4.01 (CT backprojection): every output pixel reads the shared,
read-only scene and writes only its own pixel — no inter-thread communication,
no atomics, no shared memory.

- **Thread-to-data map.** A 2-D grid of `16×16` thread blocks tiles the image.
  Thread `(px,py) = (blockIdx.x·16+threadIdx.x, blockIdx.y·16+threadIdx.y)`
  renders pixel `(px,py)` by calling the **shared** `shade_pixel()` from
  `render_core.h` — the identical function the CPU reference loops over.
- **Launch config.** `block = dim3(16,16)` (256 threads = 8 warps, good latency
  hiding and occupancy on sm_75…sm_89); `grid = (⌈W/16⌉, ⌈H/16⌉)`. The ragged
  edge tiles are guarded by `if (px>=W || py>=H) return;`.

```
        image (W x H)                     grid of 16x16 blocks
   +-------------------------+        +------+------+------+ ...
   | pixel(px,py) <- thread  |        |  B0  |  B1  |  B2  |
   |  casts 1 primary ray,   |   ==>  +------+------+------+
   |  N AO rays, 1 shadow ray|        |  B3  |  B4  | ...  |   each thread =
   +-------------------------+        +------+------+------+   one pixel
```

- **Memory hierarchy.** The scene lives in **constant memory** (`__constant__
  Atom c_atoms[MAX_ATOMS]`). Every thread reads every atom; when a warp reads
  the *same* atom address (which happens because neighbouring pixels trace in
  lockstep), constant memory **broadcasts** that read from its cache in a single
  transaction — ideal for small, read-only, uniformly-accessed data. (Flagship
  1.12 uses the same constant-memory trick for its single query fingerprint.)
  The output image is one byte per pixel in **global memory**, written once,
  fully coalesced. Per-thread state (ray, normal, accumulators) lives in
  **registers**. There is no shared memory because pixels never cooperate.
- **No CUDA library is needed.** The per-ray work is simple arithmetic; we write
  it by hand so nothing is a black box. Production tools instead call **OptiX**,
  which builds and traverses a BVH on the RT cores (§7). The catalog also
  mentions `cuFFT` for density-map smoothing — relevant to the *volume*-rendering
  extension, not to this atom (sphere) renderer.

## 5. Numerical considerations

- **Precision: FP32.** Real-time renderers use single precision: it halves
  memory traffic, the GPU is far faster at it, and an 8-bit display cannot show
  more than ~256 levels anyway. We roll our own `Vec3` so the host and device use
  byte-identical types.
- **Determinism — no floating RNG.** Ambient occlusion is a Monte-Carlo
  estimate, which naively needs random hemisphere directions. A floating RNG
  would make the result depend on draw order and break reproducibility. Instead
  we use a **fixed low-discrepancy sequence** (the Hammersley / van-der-Corput
  set, `hemisphere_dir()` in `render_core.h`): sample `k` of `N` is a
  deterministic function of `k`, `N`, and the normal `n`. Same inputs → same
  samples → same image, every run, on CPU and GPU alike.
- **No atomics, no reordering.** Each pixel is written by exactly one thread, so
  there are no races and no non-associative float sums (contrast 5.01/11.09,
  which must accumulate in integers to stay deterministic). The AO count is an
  **integer** (`escaped / N`), which is exact.
- **FMA contraction.** We compile the device code with `--fmad=false` (see the
  `.vcxproj`/`CMakeLists.txt`) so the GPU evaluates `a*b+c` as a separate
  multiply-then-add, like the host, removing one source of CPU/GPU divergence.

## 6. How we verify correctness

The CPU reference (`render_cpu` in `src/reference_cpu.cpp`) and the GPU kernel
(`render_kernel` in `src/kernels.cu`) call the **same** `shade_pixel()` from
`render_core.h`. So the two implementations differ only in *where* the loop runs,
not in *what* it computes — the strongest possible form of cross-check.

`main.cu` renders both and compares the byte images. They are **not** bit-
identical, and we are honest about why: the host C library and the CUDA device
library compute `cosf`, `sinf`, `sqrtf` to slightly different last-bit values
(~1e-6). For almost every pixel that washes out under 8-bit quantization (the
bytes are equal). The exception is **silhouette-edge pixels**: a ray that grazes
a sphere has discriminant `Δ ≈ 0`, so a 1e-6 difference can flip a *hit* into a
*miss*, swapping that one pixel between "atom edge" and "background" — a few grey
levels. This is the textbook **aliasing** sensitivity of single-sample ray
tracing, not a bug (super-sampling fixes it; see the README Exercises).

So the **documented tolerance** (docs/PATTERNS.md §4, the "small *physical*
tolerance" case) is: **PASS** when no pixel differs by more than **8 grey
levels** (≈ 3 % brightness) and **fewer than 0.1 %** of pixels differ at all. On
the reference RTX 2080 the actual disagreement is **2 pixels of 40 000, max diff
2** — far inside tolerance. We additionally print a whole-frame **FNV-1a
checksum** of the CPU image to `stdout` as a portable fingerprint, and the GPU's
own checksum + the exact pixel counts to `stderr` (which is shown, not diffed,
because it can shift by a pixel across GPUs). Edge cases handled: a ray that hits
nothing (background), `ao_samples = 0` (AO disabled → fully open), and a
degenerate zero-length normal (guarded in `normalize`).

## 7. Where this sits in the real world

Production molecular viewers do the *same per-ray math* but at a completely
different scale and polish:

- **Acceleration structure.** We brute-force every atom per ray (`Θ(n)`). VMD,
  PyMOL, OVITO and Mol* build a **BVH / k-d tree** so each ray touches
  `Θ(log n)` atoms — the only way to reach millions of atoms at 30+ fps. NVIDIA
  **OptiX/RTX** builds and traverses that BVH on dedicated **RT cores** in
  hardware, so primary, shadow, and AO rays are all hardware-accelerated.
- **Camera & representations.** Real viewers use a **perspective** camera and
  many representations beyond VDW spheres: ball-and-stick (add cylinders for
  bonds), cartoon ribbons (a spline through the backbone), and **molecular
  surfaces** (solvent-accessible / SES via the MSMS or a marching-cubes
  isosurface — the catalog's "marching cubes on GPU").
- **Volumes.** Cryo-EM maps (EMDB) are 3-D density *grids*, not atoms; rendering
  them is **volume ray-marching / compositing** (NVIDIA IndeX), with `cuFFT` used
  to smooth the density — a natural extension of this project.
- **Quality.** Real ray tracers super-sample (anti-aliasing), add soft shadows,
  multiple lights, depth of field, and physically-based materials. We keep one
  sample per pixel, one light, and a Lambert+AO model so the algorithm stays
  legible.

This project is the **conceptual core** of all of the above: cast a ray, find
the nearest sphere, shade it with occlusion and shadows — parallelized one thread
per pixel.

---

## References

- **VMD** — <https://www.ks.uiuc.edu/Research/vmd/> — the canonical CUDA/OptiX
  molecular ray tracer; study its `Tachyon`/`OptiXRenderer` for how AO and BVH
  traversal scale to huge systems.
- **PyMOL** (open source) — <https://github.com/schrodinger/pymol-open-source> —
  GPU-rendered molecular graphics; good reference for representations.
- **OVITO** — <https://www.ovito.org> — GPU scientific visualization for MD; clean
  sphere/cylinder rendering.
- **Mol\*** — <https://github.com/molstar/molstar> — WebGL viewer; shows how the
  same ideas map onto a rasterizer + screen-space AO in the browser.
- **NVIDIA OptiX Programming Guide** — the production answer to "ray tracing on
  the GPU"; the hardware BVH this project hand-waves with brute force.
- **Möller & Trumbore / "Ray Tracing in One Weekend"** — the ray/sphere and
  shading fundamentals reimplemented here didactically.
