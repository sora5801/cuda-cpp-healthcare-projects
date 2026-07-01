# THEORY — 5.8 Linac QA & Machine Performance Assessment

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

A **linear accelerator** (linac) is the machine that delivers external-beam
radiotherapy: it accelerates electrons, slams them into a target to make
megavoltage X-rays, and shapes that beam with a **multi-leaf collimator** (MLC)
to paint a prescribed dose onto a tumour while sparing healthy tissue. A modern
treatment (IMRT / VMAT) modulates the beam continuously as the gantry rotates, so
the delivered dose is the sum of hundreds of small, precisely-timed apertures.

If the machine drifts — the beam output creeps up 2%, a leaf lags a millimetre,
the beam becomes asymmetric — the delivered dose no longer matches the plan, and
a patient could be over- or under-dosed. **Quality assurance (QA)** is the routine
of catching that *before* treatment. Two questions dominate daily and per-plan QA:

1. **Does the delivered dose match the planned dose?** The physicist measures the
   dose the machine actually produces (with an **EPID** — electronic portal
   imaging device — or a 2-D detector array) and compares it, pixel by pixel, to
   the planned dose. The comparison metric is the **gamma index**.
2. **Is the beam itself healthy?** From a measured beam profile the physicist reads
   off the **central-axis output**, the **flatness** (is the top of the beam flat?),
   the **symmetry** (is the left side the same as the right?), and the **field
   size**. These are the classic *machine performance* numbers.

This project computes both, on a 2-D dose plane, with the gamma map on the GPU.

### Why not just subtract the two doses?

A naive per-pixel dose difference `|D_measured − D_planned|` is far too harsh at a
**beam edge**. There the dose falls from 100% to 0% across a few millimetres (the
*penumbra*), so a sub-millimetre spatial misalignment — well within tolerance —
produces a huge dose difference. You would fail a perfectly good machine. The gamma
index (§2) fixes this by allowing a small spatial slack.

## 2. The math

### 2.1 The gamma index (Low et al., *Med Phys* 1998)

Let `D_m` be the **measured** dose at a point `m`, and let the **reference**
(planned) dose field be `D_r` sampled at points `r`. Choose two tolerances:

- `DD` — the **dose-difference criterion**, in dose units. We use *global* gamma:
  `DD = (dd% / 100) · D_norm`, where `D_norm` is a normalisation dose (here the
  maximum of the reference plane). Example: 3% of a 100-unit max ⇒ `DD = 3`.
- `DTA` — the **distance-to-agreement**, in millimetres (e.g. 3 mm).

Define the generalised distance from measured point `m` to reference point `r`:

```
Γ(m, r) = sqrt(  (D_m − D_r)² / DD²   +   |r − m|² / DTA²  )
```

Both terms are **dimensionless** — a dose difference measured in units of `DD`, and
a spatial distance measured in units of `DTA`. The **gamma value** at `m` is the
minimum over all reference points:

```
γ(m) = min_r  Γ(m, r)
```

Interpretation: `γ(m)` is the radius, in the combined dose/space ellipsoid, of the
smallest region around `m` that contains an acceptable reference point.

- `γ(m) ≤ 1` ⇒ there exists a reference point that is *simultaneously* close enough
  in dose **and** in space ⇒ the point **passes**.
- `γ(m) > 1` ⇒ no such point ⇒ the point **fails**.

The clinical scalar is the **gamma pass rate**:

```
pass rate = 100 · (# evaluated points with γ ≤ 1) / (# evaluated points)   [%]
```

"Evaluated" excludes near-zero background (a **low-dose threshold**, here 10% of
`D_norm`), where the ratios are noise. **AAPM TG-218** sets the universal action
limit for per-beam IMRT QA at **≥ 95% at 3%/3mm**.

### 2.2 The machine-performance metrics

From the measured plane's **central-axis (CAX) profile** (the middle row), with
pixel spacing `Δ` mm:

- **CAX output** `= D(centre)` — the beam's on-axis dose reading.
- **Field width (FWHM)** — distance between the two points where `D` crosses
  `0.5 · D(centre)` (the 50%-isodose beam edges).
- **Flat region** — the central 80% of the field width (edges/penumbra excluded).
- **Flatness** `= (D_max − D_min) / (D_max + D_min) · 100%` over the flat region.
- **Symmetry** `= max_x |D(+x) − D(−x)| / D(centre) · 100%` over the flat region —
  the worst left/right imbalance about the axis.

Typical clinical tolerances are ~±3% flatness and ~±2% symmetry (vendor-dependent).

## 3. The algorithm

```
INPUT : reference plane D_r[ny][nx], measured plane D_m[ny][nx],
        spacing Δ (mm), criteria dd% and DTA (mm), norm dose D_norm
OUTPUT: gamma map γ[ny][nx], pass rate, machine metrics

1.  DD  <- (dd%/100) · D_norm                         # % -> absolute dose
    R   <- ceil(3·DTA / Δ)  pixels                    # search half-width
2.  for each measured pixel m = (mx,my):              # <-- parallelised
        best <- +inf
        for each reference pixel r in the (2R+1)² window around m:
            dist2 <- ((rx-mx)Δ)² + ((ry-my)Δ)²        # mm²
            g2    <- (D_m[m]-D_r[r])²/DD²  +  dist2/DTA²
            best  <- min(best, g2)
        γ[m] <- sqrt(best)
3.  pass rate <- 100 · count(γ ≤ 1 among m with D_m[m] ≥ cut) / count(...)
4.  metrics   <- flatness/symmetry/output/FWHM from D_m central row
```

**Why the search window (step 1, R).** A reference point farther than `3·DTA`
already contributes a space term `dist²/DTA² > 9 ≫ 1`, so it can never be the
minimum for any point that might pass. Clipping the search to a `(2R+1)²` window
turns an `O(N²)` all-pairs search into an `O(N · R²)` local one with no change to
the answer for passing pixels.

**Complexity.** With `N = nx·ny` pixels and a window of `W = (2R+1)²`:

| | work | depth (parallel) |
|---|---|---|
| Serial CPU | `O(N · W)` | `O(N · W)` |
| GPU (one thread/pixel) | `O(N · W)` total | `O(W)` — every pixel in parallel |

The gamma step is **embarrassingly parallel**: each pixel's result depends only on
a read-only local window, so there is no communication, no reduction across
threads, no ordering constraint. Arithmetic intensity is modest (a few flops per
loaded dose value), so at large sizes the kernel is **bandwidth/cache-bound** —
which is exactly why the overlapping-window reuse (neighbouring pixels read almost
the same reference data) matters (see §4).

## 4. The GPU mapping

This is the **gather + per-thread min-reduction** pattern (PATTERNS.md §1),
the same shape as the CT-backprojection flagship `4.01`.

- **Thread-to-data mapping.** One thread owns exactly one *measured* pixel:
  `mx = blockIdx.x·blockDim.x + threadIdx.x`, `my = blockIdx.y·blockDim.y +
  threadIdx.y`. Thread `(mx,my)` writes `γ[my·nx + mx]` and nobody else touches
  that cell — so there are **no races, no atomics, no shared memory** needed.
- **Launch configuration.** A 2-D grid of **16×16 = 256-thread** blocks tiles the
  2-D plane: `grid = (⌈nx/16⌉, ⌈ny/16⌉)`. 256 threads/block is a good occupancy
  default on sm_75–sm_89; a square block matches the square access pattern and keeps
  the ragged-edge guard (`if (mx≥nx || my≥ny) return;`) trivial.
- **Memory hierarchy.** The two planes live in **global memory**; the running
  minimum and loop indices live in **registers**. The only "gather" is the
  `(2R+1)²` window read of the reference plane. Because adjacent threads in a block
  read heavily-**overlapping** windows, the L1/L2 cache serves most of those loads —
  the data-reuse the algorithm needs comes for free from the cache. On production
  code the delivered-dose field is often bound to a **texture** (the catalog's
  suggestion): texture memory adds hardware 2-D spatial caching and free bilinear
  interpolation for sub-pixel DTA. We use plain cached global loads here because
  they are the clearest teaching form and the L2 already captures the reuse.
- **No CUDA library.** This exact/deterministic core is a hand-written kernel — the
  point is to *see* the gather and the min-reduction, not hide them behind a library.
  The catalog's cuBLAS/texture suggestions belong to the ML-log and 3-D extensions
  (§7), which are deliberately out of scope for this teaching version.

```
        measured plane (nx x ny)                 one thread's job
     +-----------------------------+          +---------------------+
     | . . . . . . . . . . . . . . |          |  window of D_ref    |
     | . . . +-------+ . . . . . . |          |   (2R+1)x(2R+1)     |
     | . . . | block | . . . . . . |   --->    |  min over r of      |
     | . . . |16 x 16| . . . . . . |          |  (dD/DD)^2+(dist/DTA)^2
     | . . . +-------+ . . . . . . |          |   -> gamma[m]       |
     | . . . . . . . . . . . . . . |          +---------------------+
     +-----------------------------+
       grid = (ceil(nx/16), ceil(ny/16))   thread (mx,my) owns pixel (mx,my)
```

## 5. Numerical considerations

- **Precision: FP32.** Dose planes are 32-bit floats (EPID/detector data is ~16-bit
  physically; FP32 is ample). Every operation in `gamma_value_at` is single-precision.
- **Determinism — and why it is exact here.** The inner loop only ever takes a
  `min`, which is **associative and commutative** and *order-independent*: visiting
  the window in any order gives the same minimum. There is no floating-point *sum*
  whose result depends on order (contrast the Monte-Carlo tally in `5.01`, which
  must accumulate in integers to stay deterministic). Combined with the shared
  `__host__ __device__` math (§6), the CPU and GPU produce **bit-identical** gamma
  maps. The pass-rate count is pure integer arithmetic, so it is deterministic too.
- **No races.** Each thread writes one unique output cell; inputs are read-only.
  There are no atomics and no `__syncthreads()`.
- **Edge handling.** The window is clamped to the plane bounds (`continue` on
  out-of-range indices), so border pixels simply search a smaller, valid window —
  identical on host and device.
- **Guards.** Division by a zero tolerance is guarded (`inv_dd2`, `inv_dta2` fall
  back to 0), and the empty-evaluation case returns a 0% rate rather than dividing
  by zero.

## 6. How we verify correctness

The GPU result is checked against an independent **CPU reference**
(`src/reference_cpu.cpp`, `gamma_map_cpu`) in `main.cu`:

1. **Shared math ⇒ exact agreement.** Both paths call the *same*
   `gamma_value_at()` from `src/gamma.h`, decorated `__host__ __device__` so nvcc
   compiles it for the GPU and the host compiler compiles it for the CPU — the
   identical float operations in the identical order. So we demand
   **`max_abs_err = 0`** (PATTERNS.md §4: "exact" is the honest tolerance when the
   same operations run on both sides). If the two maps ever differed, that would
   signal a real bug (a mis-indexed window, a launch-bounds error), not rounding.
2. **Cross-checked pass counts.** The pass rate is recomputed from *both* maps and
   the integer pass/eval counts must match exactly (`gpu_pass == cpu_pass`).
3. **A meaningful, recoverable answer.** The synthetic sample bakes in a known
   error (1% low output, 2% right-side asymmetry). The metrics recover it: symmetry
   ≈ 1.96% ≈ the injected 2%, flatness ≈ 0.99% ≈ the injected 1%, CAX output ≈ 101.
   That validates the *science*, not just CPU==GPU agreement — the second, stronger
   check recommended in PATTERNS.md §4.

## 7. Where this sits in the real world

Production QA software does much more than this teaching kernel:

- **3-D volumetric gamma.** Clinical VMAT QA compares 3-D dose *volumes*
  (~200³ voxels), searching a 3-D sphere per voxel — the ~10⁹ distance searches the
  catalog cites. The extension is mechanical (a third loop, a sphere clip) but the
  cost is `O(N·R³)`; this is where the GPU truly dominates. **Plastimatch** ships a
  GPU 3-D gamma.
- **Sub-pixel DTA via interpolation.** Real tools interpolate the reference field
  between samples (bilinear/trilinear) so the DTA is continuous, not quantised to
  the grid — exactly what CUDA **texture** units accelerate for free.
- **EPID → dose reconstruction.** Converting a raw portal *image* to absolute dose
  needs a Monte-Carlo or convolution/superposition kernel (the catalog's item 2);
  that is a separate GPU workload (cf. flagship `5.01`).
- **Machine-log ML.** Vendors log every leaf position and monitor unit; predicting
  failures from millions of log rows is a matrix/ML problem (the catalog's cuBLAS
  feature-matrix note) — a different project entirely.
- **The clinical conventions we simplified.** Real flatness/symmetry use calibrated
  profiles on *both* axes, vendor-specific definitions, and careful edge detection.
  **Pylinac** is the open-source reference for all of these; **matRad** shows gamma
  inside a full planning system; **TG-218** defines the tolerances.

This project deliberately scopes to the **2-D, exact, offline** case so it builds
and verifies end-to-end on a normal machine — the didactic core from which the 3-D,
interpolated, clinical version is a set of well-understood extensions (see the
README exercises).

---

## References

- **Low DA, Harms WB, Mutic S, Purdy JA.** "A technique for the quantitative
  evaluation of dose distributions." *Med Phys* 25(5):656–661, 1998. — the paper
  that defines the gamma index (§2).
- **Miften M, et al. (AAPM TG-218).** "Tolerance limits and methodologies for IMRT
  measurement-based verification QA." *Med Phys* 45(4):e53–e83, 2018. — the source
  of the 3%/3mm, ≥95%, and low-dose-threshold conventions used here.
- **Pylinac** — https://github.com/jrkerns/pylinac — the reference open-source
  implementation of these exact analyses (gamma, flatness/symmetry, Winston-Lutz).
- **Plastimatch** — https://plastimatch.org/ — production GPU-accelerated (3-D)
  gamma; compare its search and interpolation to our 2-D kernel.
- **matRad** — https://github.com/e0404/matRad — research TPS with plan-vs-measurement
  gamma comparison, for the full-system context.
