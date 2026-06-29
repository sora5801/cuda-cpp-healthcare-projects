# THEORY — 4.01 CT Reconstruction (Filtered Backprojection)

> For a reader who knows C++ but is new to CUDA and to tomography.
> See [README.md](README.md) for the tour and build. _Educational only._

## 1. The science

A CT scanner rotates an X-ray source/detector around the patient and measures how
much each ray is attenuated. Each measurement is a **line integral** of the
tissue's attenuation coefficient `μ(x,y)` along that ray. The set of all such
integrals, indexed by angle and detector position, is the **sinogram**. The
reconstruction problem is to invert this: recover the image `μ(x,y)` from its line
integrals. This is the mathematical core of every CT scanner, cone-beam dental
scanner, and linac on-board imager.

## 2. The math

The **Radon transform** of an image `f` is its set of line integrals:

```
p(θ, s) = ∫ f(x, y) δ(x·cosθ + y·sinθ − s) dx dy
```

i.e. the integral of `f` along the line at angle `θ` whose signed distance from
the origin is `s`. The **Fourier-slice theorem** says the 1-D Fourier transform of
`p(θ, ·)` equals a radial slice of the 2-D Fourier transform of `f`. Inverting
with the Jacobian `|ω|` of the polar-to-Cartesian change of variables gives
**Filtered BackProjection**:

```
f(x, y) = ∫_0^π  ( p(θ, ·) * h )( x·cosθ + y·sinθ )  dθ
```

where `*` is 1-D convolution and `h` is the **ramp filter** with frequency
response `|ω|` (Ram-Lak). The inner expression is "filter the projection";
the integral over `θ` is "backproject". Without `h`, plain backprojection gives a
`1/r`-blurred image.

## 3. The algorithm

```
for each projection row:  filtered = ramp_filter(projection)     # O(n_det log n_det) via FFT
for each pixel (x,y):                                             # backprojection
    f(x,y) = (pi/n_angles) * sum_k  interp( filtered_k, x*cos θ_k + y*sin θ_k )
```

**Complexity.** Filtering is `O(n_angles · n_det log n_det)`. Backprojection is
`O(img² · n_angles)` — the dominant term, and the part we put on the GPU. In 3-D
(FDK) it becomes `O(vox³ · n_proj)`, which is where the GPU becomes essential.

## 4. The GPU mapping

**Decomposition.** One thread per output pixel, on a 2-D grid of 16×16 blocks
that tiles the image. Thread `(px, py)` owns pixel `(px, py)`:

```
  for each angle k:
     s    = wx*cos[k] + wy*sin[k]       # where this pixel's ray hits the detector
     fidx = s/ds + center              # fractional detector index
     acc += lerp(filtered_k[fidx])     # linear interpolation in the detector
  image = acc * (pi / n_angles)
```

**Memory hierarchy.** `filtered`, `cos`, `sin`, and the output image live in
global memory. Each thread reads `n_angles` interpolated samples; consecutive
threads (neighbouring pixels) read nearby detector positions, so the access has
good locality and the kernel is **memory-bandwidth bound** — exactly the regime
GPUs dominate. Production code binds `filtered` to a **texture**, whose hardware
samplers perform the linear interpolation for free and cache 2-D neighbourhoods.

**CPU/GPU parity.** `cos`/`sin` are computed **once on the host** and uploaded, so
the GPU does not use `cosf` where the CPU uses `cos` — removing the largest source
of disagreement. The remaining difference (float rounding / fused multiply-add in
the accumulation) is ~`1e-5`, far inside our `1e-3` tolerance.

**Independence.** Pixels are independent: no shared memory, no atomics, no
synchronization. This is why the GPU shows a real ~30× speed-up on the sample
(unlike the launch-bound wavefront of `3.01`), and the gap widens with resolution.

## 5. Numerical considerations

- **Precision.** Single precision is standard for backprojection; the ramp filter
  is computed in double here for a clean kernel, then stored as float.
- **The ramp filter and noise.** `|ω|` amplifies high frequencies (and noise);
  real systems apodize it (Shepp-Logan, Hann windows) — Exercise 5/THEORY note.
- **Determinism.** The per-pixel sum is in a fixed angle order with no cross-thread
  reduction, so the GPU result is reproducible run to run.
- **Scaling.** Absolute intensity depends on the geometry/normalization
  conventions; here it is calibrated so a unit-density disc reconstructs to ≈ 1.

## 6. How we verify correctness

`main.cu` runs `backproject_cpu` and `backproject_gpu` on the *same* host-filtered
sinogram and compares the images (`max_abs_err`). They agree to ~`1e-5`. As a
sanity check beyond CPU/GPU parity, the reconstruction is physically meaningful:
the center pixel returns the main disc's density (≈ 1.0), values are flat inside
each disc and ≈ 0 outside — i.e. the algorithm actually inverts the Radon
transform, not just "the two implementations agree".

## 7. Where this sits in the real world

Clinical reconstruction uses **FDK** (Feldkamp-Davis-Kress): cone-beam geometry,
a 3-D voxel grid, 2-D projections, a cosine pre-weight, and the ramp filter along
detector rows. Helical scans add **Parker short-scan weighting** or exact
**Katsevich** reconstruction. Iterative methods (SART, MBIR, and now deep-learning
reconstruction) reduce dose and artifacts at far greater compute cost — which is
precisely why production toolkits (RTK, ASTRA, TIGRE, Plastimatch) are
GPU-accelerated. The backprojection gather you see here is the computational heart
of all of them.

## References

- Kak & Slaney, *Principles of Computerized Tomographic Imaging* (1988) — the standard FBP reference.
- Feldkamp, Davis & Kress (1984) — practical cone-beam FBP (FDK).
- ASTRA / TIGRE / RTK documentation — GPU projection/backprojection in practice.
- NVIDIA CUDA C++ Programming Guide — texture objects and 2-D grids.
