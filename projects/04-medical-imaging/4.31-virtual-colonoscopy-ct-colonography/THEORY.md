# THEORY — 4.31 Virtual Colonoscopy & CT Colonography

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See
> [README.md](README.md) for the quick tour and build steps.
>
> _Educational only — not for clinical use._

This project builds **one frame of a virtual-colonoscopy fly-through** by GPU
**volume ray-casting**. It is the reduced-scope teaching slice of the full CT
colonography (CTC) pipeline in the catalog — we isolate the single step that is
the GPU's reason for being (real-time rendering of the lumen interior) and make
it legible end-to-end, CPU reference included.

---

## 1. The science

A **colonoscopy** looks for **polyps** (small growths on the colon wall) that can
become cancer. The optical version threads a camera through the bowel.
**CT colonography** is the non-invasive alternative: the colon is inflated with
air (a CO₂ insufflator distends it), a CT scanner acquires a 3-D volume, and
software lets a radiologist "fly through" the colon's hollow interior on screen —
a **virtual colonoscopy**.

The trick that makes this possible is **contrast**: air reads very low on CT
(about −1000 Hounsfield Units), while the soft-tissue wall reads near 0 HU. So the
**lumen** (the air-filled hollow) and the **wall** are separated by a sharp
density jump. If we put a virtual camera inside the lumen and render the wall
where that density crosses a threshold (an **iso-surface**), we reproduce exactly
what the inside of the colon looks like — and a **polyp** shows up as a convex
bump protruding into the lumen.

**What we model here (synthetic, reduced scope).** Our input is a tiny synthetic
CT volume: an air-filled tube (the lumen, density ≈ 0) through soft tissue (the
wall, density ≈ 1), gently bent, with **one spherical polyp** bulging into the
lumen on the lower wall. We place the endoscope camera at the mouth of the tube
looking down its axis and render one frame. The polyp is our **known answer**:
its convex surface catches the camera's headlamp and reads ~2× brighter than the
smooth wall around it, which the demo measures and reports.

> Our densities are unitless 0..1 for clarity, **not** Hounsfield Units. The scene
> is **synthetic** and implies nothing clinical (CLAUDE.md §8).

---

## 2. The math

**Inputs.**
- A density field sampled on a grid, $\rho : \mathbb{Z}^3 \to \mathbb{R}$, stored
  as `vol[(z\,n_y + y)\,n_x + x]`. Between voxels we define a continuous field by
  **trilinear interpolation** $\tilde\rho(\mathbf{p})$.
- A camera: eye $\mathbf{e}$, orthonormal basis $(\mathbf{r},\mathbf{u},\mathbf{f})$
  (right/up/forward), and a field-of-view scale $s$.
- An **iso-value** $\tau$ (here $0.5$): the density of the wall surface we render.

**Per-pixel ray.** Pixel $(p_x,p_y)$ of a $W\times H$ frame defines a ray
$\mathbf{x}(t)=\mathbf{e}+t\,\hat{\mathbf{d}}$ with

$$
u = \Big(\tfrac{2(p_x+0.5)}{W}-1\Big)s,\quad
v = \Big(\tfrac{2(p_y+0.5)}{H}-1\Big)s,\quad
\mathbf{d}=\mathbf{f}+u\,\mathbf{r}+v\,\mathbf{u},\quad
\hat{\mathbf{d}}=\mathbf{d}/\lVert\mathbf{d}\rVert .
$$

**First-hit iso-surface.** March $t = 0, h, 2h, \dots$ (step $h$ in voxels) and
find the first interval where the interpolated density crosses $\tau$:

$$
\tilde\rho(\mathbf{x}(t_{i})) < \tau \le \tilde\rho(\mathbf{x}(t_{i+1})) .
$$

Refine the crossing by one linear (secant) step to get the sub-step hit point
$\mathbf{h}$.

**Surface normal.** The outward normal is the negated, normalized **gradient** of
the density (density rises *into* the wall), estimated by central differences:

$$
\nabla\rho \approx \tfrac{1}{2h}\big(\tilde\rho(\mathbf{h}+h\mathbf{e}_x)-\tilde\rho(\mathbf{h}-h\mathbf{e}_x),\;\dots\big),
\qquad \mathbf{N} = -\widehat{\nabla\rho}.
$$

**Shading (Blinn-Phong, headlamp).** With light and view both along $-\hat{\mathbf{d}}$
(a lamp on the scope), half-vector $\mathbf{H}=\widehat{\mathbf{L}+\mathbf{V}}$:

$$
I = k_a + k_d\max(\mathbf{N}\!\cdot\!\mathbf{L},0) + k_s\max(\mathbf{N}\!\cdot\!\mathbf{H},0)^{m},
\qquad (k_a,k_d,k_s,m)=(0.15,0.70,0.35,16),
$$

clamped to $[0,1]$. Rays that never hit the wall return $0$ (background).

---

## 3. The algorithm

```
render(volume, camera, iso, step, max_steps, W, H):
  for each pixel (px,py):                 # W*H independent rays
      (origin,dir) = pixel_ray(camera,px,py,W,H)
      prev = sample(origin)
      for i in 0..max_steps:              # march the ray
          p2  = p + step*dir
          cur = sample(p2)                # 8 voxel reads (trilinear)
          if prev<iso and cur>=iso:       # air -> wall crossing
              hit = secant_refine(p,p2,prev,cur,iso)
              N   = normalize(-gradient(hit))   # 6 more samples
              return phong(N, -dir)
          p,prev = p2,cur
      image[py*W+px] = 0                   # ray escaped -> black
```

**Complexity.** Per ray: up to `max_steps` marches, each one trilinear sample = 8
reads; a hit adds 6 samples for the gradient. Total work
$\Theta(W\cdot H\cdot \text{max\_steps})$ memory reads. It is **memory-bandwidth
bound**, not compute bound — the arithmetic per sample (a handful of FMAs) is
cheap; the cost is fetching voxels.

**Why parallel.** Every pixel's ray is **completely independent** — no shared
state, no ordering. That is the textbook **gather** pattern (PATTERNS.md §1, same
family as `4.01` CT backprojection). Serial depth is one ray's march; parallel
width is all `W*H` rays at once.

---

## 4. The GPU mapping

**Thread-to-data mapping.** A 2-D thread grid over the image: thread
$(p_x,p_y)=(\text{blockIdx}\cdot\text{blockDim}+\text{threadIdx})$ owns output
pixel $(p_x,p_y)$ and writes exactly `image[py*W+px]`. One thread = one ray.

**Launch configuration.** `block = 16×16 = 256` threads (a square tile matching the
square image; good occupancy on sm_75…sm_89), `grid = ⌈W/16⌉×⌈H/16⌉`. The ragged
edge tiles are guarded by `if (px>=W || py>=H) return;`.

```
        image (W x H)                     one block = 16x16 threads
   +----+----+----+----+ ...           +--------------------+
   | b00| b10| b20| ...|               | t(0,0) ... t(15,0) |   each thread t
   +----+----+----+----+               |   .            .   |   marches ONE ray
   | b01| ...|                         | t(0,15)... t(15,15)|   into the volume
   +----+----+ ...                     +--------------------+
   grid = ceil(W/16) x ceil(H/16)      thread (px,py) -> image[py*W+px]
```

**Memory hierarchy.**
- **Global memory** holds the CT volume (uploaded once) and the output image.
- **Registers** hold each ray's marching state (`p`, `prev`, accumulators) — no
  shared memory is needed because rays do not cooperate.
- **No atomics, no shared memory, no inter-thread sync** — the gather's defining
  simplicity.

**Where a real renderer differs (no black box).** Production fly-throughs bind the
volume to a **CUDA 3-D texture** and call `tex3D<float>()`. The texture unit then
(a) does the 8-corner **trilinear blend in hardware** (one instruction instead of
our ~15), (b) **caches 3-D neighborhoods** so nearby rays reuse fetched voxels,
and (c) handles clamp/border addressing for free. We deliberately do the blend by
hand in `sample_volume()` for two reasons: it is the teaching content (you see the
8-corner lerp), and — crucially — the texture unit's interpolation uses **9-bit
fixed-point weights**, so it would **not** match our FP32 CPU reference bit-for-bit
(see §6). Swapping `sample_volume()` for `tex3D` is the natural exercise.

This project links **no CUDA math library** — the gather is hand-written CUDA C++
(only `cudart`). The catalog mentions OptiX/OpenGL for rendering and cuDNN for the
polyp-detection CNN; those belong to the full pipeline (§7), not this teaching slice.

---

## 5. Numerical considerations

- **Precision: FP32.** Rendered intensities live in $[0,1]$; single precision is
  plenty and halves bandwidth vs FP64. The shared core
  ([`volume_render.h`](src/volume_render.h)) is FP32 on both CPU and GPU.
- **Determinism.** Each pixel is written by exactly one thread; there is no
  reduction and no `atomicAdd`, so there is **no float-summation reordering**. The
  stdout report uses integer pixel counts and rounded floats, so it is
  **byte-identical every run** (PATTERNS.md §3). Timings (which vary) go to stderr.
- **CPU vs GPU drift.** The only source of disagreement is the compiler's freedom
  to **fuse multiply-adds (FMA)** and reorder associative FP ops differently on
  host vs device. Because the CPU loop and the GPU kernel call the **same**
  `cast_ray()` in the **same order**, that drift is tiny — measured
  `max_abs_err ≈ 4.8e-7` here.
- **Branch divergence.** Rays in a warp hit the wall at different steps, so they
  exit the march loop at different iterations — some thread divergence. It is mild
  (neighboring pixels see similar geometry) and is the price of the gather's
  simplicity; a real renderer mitigates it with empty-space skipping.

---

## 6. How we verify correctness

Two independent checks (PATTERNS.md §4):

1. **CPU == GPU (max-abs-error ≤ 1e-3).** `render_cpu()` and `render_kernel()`
   call the **identical** shared `cast_ray()` math, so they should agree to FP32
   rounding. We require `max|I_cpu − I_gpu| ≤ 10^{-3}` over all pixels and report
   the actual error (~`5e-7`) on stderr. The tolerance is **not** zero because of
   the host/device FMA freedom in §5 — pretending it were exact would be dishonest.
   1e-3 in $[0,1]$ shading units is a physically negligible, well-justified bound.

2. **The known answer (the polyp).** The synthetic scene embeds a polyp whose
   convex surface, lit by the headlamp, reads **~2× brighter** than the smooth wall
   it sits on. The demo measures the mean brightness of the fixed frame window the
   polyp projects into and reports `polyp-region mean brightness ≈ 0.73` (vs ≈ 0.36
   for that window with the polyp removed). Recovering a planted feature validates
   the **science**, not just that two implementations of the same code agree.

Edge cases handled: clamp-to-edge addressing at the volume border (so no
out-of-bounds reads), a guard against divide-by-zero in the secant refinement and
the normalization, and the ragged-tile bounds check in the kernel.

---

## 7. Where this sits in the real world

The catalog's full CTC pipeline has **five** GPU-accelerated stages; we built a
faithful version of stage 4 and left the rest described here:

1. **Colon segmentation** from the 512³ CT (a 3-D CNN, e.g. nnU-Net via **MONAI**)
   to isolate the air-filled lumen.
2. **Electronic colon cleansing** — subtract orally-tagged residual stool/fluid so
   it is not mistaken for wall (thin-plate-spline tagged-material subtraction).
3. **Centerline extraction** (GPU **fast-marching**) to define an automatic
   fly-through path down the lumen.
4. **Volume rendering of the lumen interior** — *this project* — at 60 FPS, with the
   volume in a CUDA 3-D texture and OptiX/OpenGL doing the ray-casting and shading.
5. **Computer-aided polyp detection** — a CNN (via **cuDNN**) classifying 3-D
   patches or rendered views; shape-index / curvedness features flag candidates.

Production tools — **3D Slicer** and **VTK** (GPU volume ray-casting engine),
**MONAI** (segmentation), **VisIt** (large-volume visualization) — add: full DICOM
volumes, hardware texture interpolation, empty-space skipping and early-ray
termination for speed, transfer functions for tissue coloring, supine/prone
registration, and validated CAD detection. Our teaching version omits all of that
to keep the **per-pixel gather** in sharp focus.

---

## References

- **3D Slicer** — <https://github.com/Slicer/Slicer> — open-source platform; study
  its `VolumeRendering` and colon-segmentation modules for the production pipeline.
- **VTK** — <https://vtk.org/> — the GPU volume ray-casting engine many viewers
  build on; `vtkGPUVolumeRayCastMapper` is the reference implementation of stage 4.
- **MONAI** — <https://github.com/Project-MONAI/MONAI> — nnU-Net colon/lumen
  segmentation (stage 1) and 3-D CNN polyp detection (stage 5).
- **VisIt** — <https://visit-dav.github.io/visit-website/> — GPU visualization for
  very large CT volumes; useful for the scale story.
- Levoy, *Display of Surfaces from Volume Data* (1988) — the foundational
  volume-rendering paper; our iso-surface ray-cast is its first-hit special case.
- TCIA **CT Colonography** collection —
  <https://wiki.cancerimagingarchive.net/display/Public/CT+Colonography> — the real
  data this teaching scene stands in for.
