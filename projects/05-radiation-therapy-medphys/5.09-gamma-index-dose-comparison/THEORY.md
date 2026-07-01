# THEORY — 5.9 Gamma-Index Dose Comparison

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

In modern radiotherapy — **IMRT** (intensity-modulated radiotherapy) and **VMAT**
(volumetric-modulated arc therapy) — a linear accelerator paints a highly
non-uniform 3-D dose onto a tumor while sparing nearby organs. The planned dose
is computed by a **treatment planning system (TPS)**. But the machine that
delivers it — moving multileaf-collimator leaves, a rotating gantry, a pulsed beam
— never reproduces the plan perfectly. So every patient-specific plan undergoes
**pre-treatment QA**: the plan is delivered to a phantom carrying a detector
(radiochromic film, an electronic portal imaging device / EPID, or a 2-D/3-D
diode/ion-chamber array), and the **measured** dose is compared to the
**planned** dose.

The hard part of that comparison is that dose maps are *smooth but steep*: near a
field edge the dose can fall from 100% to 20% over a couple of millimeters. A
naive point-by-point dose difference is then brutally unfair — a 1 mm spatial
misalignment in a steep gradient looks like a huge dose error even though the
delivery is essentially correct. Conversely, in a flat region a small dose error
is meaningful and a spatial tolerance should not excuse it.

Low ([1998](https://doi.org/10.1118/1.598248)) proposed the **gamma index** to
handle both regimes at once: it accepts a point if the measured dose is close in
*dose* **or** if a correct dose exists close by in *space*, combining a
**dose-difference (DD)** tolerance and a **distance-to-agreement (DTA)** tolerance
into a single scalar. It is now the near-universal metric for IMRT/VMAT QA, quoted
as a **pass-rate** (e.g. "98.5% of points pass 3%/3 mm"), with acceptance
thresholds standardized by **AAPM TG-218**.

## 2. The math

We have two dose distributions sampled on a grid:

- **reference** `D_r(r)` — the planned dose (the "gold standard" for the search),
- **evaluated** `D_e(r)` — the measured dose,

with `r` a spatial position (mm). Two acceptance criteria are fixed:

- **Δd_crit** — the dose-difference criterion. With *global* normalization
  (used here), Δd_crit = `P/100 · D_r^max`, i.e. P% of the maximum reference dose.
- **DTA_crit** — the distance-to-agreement criterion (mm).

For a fixed reference point `r_r`, and any evaluated point `r_e`, define the
**generalized distance** in the combined dose–space:

```
Γ(r_r, r_e) = sqrt(  (D_e(r_e) − D_r(r_r))² / Δd_crit²   +   |r_e − r_r|² / DTA_crit²  )
```

The **gamma index** at `r_r` is the minimum of this over all evaluated points:

```
γ(r_r) = min over r_e of  Γ(r_r, r_e)
```

Interpretation: γ ≤ 1 means *there exists* an evaluated point that is
simultaneously within the ellipse defined by the two tolerances — the point
**passes**. γ > 1 means no acceptable match exists — it **fails**. The two terms
are dimensionless (each is a ratio to its criterion), so adding them in quadrature
is well posed.

Two derived quantities drive the clinical decision:

- **pass-rate** = (# analyzed points with γ ≤ 1) / (# analyzed points), where
  "analyzed" excludes low-dose background below a threshold (here 10% of D_r^max).
- **γ_max / γ_mean** — where and how badly the delivery disagrees.

## 3. The algorithm

**Exhaustive, distance-limited search.** For each of the N reference voxels, scan
the evaluated voxels within a physical window and keep the running minimum of Γ²
(we work in *squared* space and take one `sqrt` at the end, since `sqrt` is
monotonic and argmin(Γ²) = argmin(Γ)):

```
for each reference voxel r_r:                         # N iterations
    if D_r(r_r) < dose_threshold: γ = 0; continue     # skip background
    best = +inf
    for each evaluated voxel r_e in window(r_r):       # K iterations
        term = (D_e−D_r)²/Δd_crit²  +  |r_e−r_r|²/DTA_crit²
        best = min(best, term)
    γ(r_r) = sqrt(best)
```

**Why a window and not all pairs.** An evaluated point at pure spatial distance
`s` contributes at least `(s/DTA_crit)²` to Γ² even with a perfect dose match. If
we already hold a candidate with Γ² ≤ 1, any point farther than `DTA_crit` in the
*best case* cannot beat it. In practice a window of a few DTA criteria captures
the true minimum; we use `search = 3·DTA_crit`. This turns the naive **O(N²)**
all-pairs search (in 2-D; **O(N⁶)** phrased over a 3-D N-cubed grid as in the
catalog) into **O(N·K)** with `K` = window area — the catalog's "distance-limited"
reduction.

**Complexity.**

| | serial work | parallel depth |
|---|---|---|
| CPU reference | `O(N·K)` | `O(N·K)` (one core) |
| GPU kernel | `O(N·K)` total | `O(K)` (N threads run the inner search concurrently) |

`K` here is `(2·radius+1)²` with `radius = ceil(3·DTA_crit / spacing)`. The
arithmetic intensity is modest (a few FLOPs per evaluated-voxel load), so the
kernel is **memory-latency bound**, not compute bound — which is exactly why the
read-only cache and (as an exercise) shared-memory tiling matter.

## 4. The GPU mapping

**Thread-to-data mapping.** The reference grid is 2-D, so we launch a **2-D grid
of 2-D blocks**. Thread `(gx, gy)` owns reference voxel `(rx = gx, ry = gy)`:

```
rx = blockIdx.x * blockDim.x + threadIdx.x
ry = blockIdx.y * blockDim.y + threadIdx.y
```

**Launch configuration.** Blocks are `16 × 16 = 256` threads (`TILE = 16`). 256 is
a multiple of the 32-lane warp, gives the scheduler 8 warps per block to hide the
memory latency of the gather, and keeps many blocks resident for occupancy on
sm_75…sm_89. The grid is `ceil(W/16) × ceil(H/16)` so it covers the whole voxel
grid; edge tiles are guarded by an `if (rx >= W || ry >= H) return;`.

**Memory hierarchy (and why).**

- `d_ref`, `d_eval` live in **global memory**, read-only in the kernel. Marking
  them `const … __restrict__` lets the compiler route loads through the
  **read-only data cache**. Because neighboring threads scan overlapping windows,
  those evaluated-dose loads hit L2/read-only cache heavily — the naive gather is
  already fast for small windows.
- The running minimum `best_sq` lives in a **register** — the reduction is
  entirely per-thread, so there is **no shared memory and no atomics**. Each
  thread writes exactly one output voxel `d_gamma[ry*W+rx]`, so there are no write
  conflicts. (Contrast flagship 5.01 Monte-Carlo dose, where many photon threads
  tally into shared bins and *must* use atomics.)
- The catalog suggests **shared-memory tiling** of the evaluated tile (load the
  block's tile + a halo of width `radius` once, then search from shared memory).
  That trades redundant global loads for one cooperative load; it is left as
  Exercise 1 because the naive version teaches the mapping most clearly.

```
Reference grid (W x H)                 One thread's job
+----+----+----+----+                  thread (rx,ry):
|    |    |    |    |   16x16 block       best = +inf
+----+----+----+----+   = 256 threads     for r_e in window(rx,ry):   <- gather
|    | ## |    |    |     each owns ONE       best = min(best, Γ²)     <- reduce
+----+----+----+----+     reference voxel  γ[rx,ry] = sqrt(best)       <- 1 write
|    |    |    |    |
+----+----+----+----+   grid = ceil(W/16) x ceil(H/16) blocks
```

**No CUDA library is needed.** Earlier drafts of the catalog note suggest cuBLAS
for cohort pass/fail statistics; here the statistics are a handful of **integer
counts** (analyzed / passing), which are both trivial and, crucially,
**deterministic** — so we compute them directly rather than through a library.
Were we to hand off a genuinely heavy linear-algebra step, we would document what
the library computes and what hand-rolling would cost (CLAUDE.md §6.1.6); there is
no such step here, so there is no black box.

## 5. Numerical considerations

- **Precision.** The per-pair math (`gamma_core.h`) is `double`. Dose values are
  stored `float` (detector precision is far coarser than FP32), but promoting to
  `double` inside the term keeps the squared differences and the accumulation of
  the two terms clean, and matches the host exactly.
- **Reduction determinism — the key point.** The reduction is a **minimum**, not a
  sum. Floating-point `min` is both **associative and commutative and exact**: no
  rounding happens, and the answer does not depend on the order in which
  candidates are visited. This is fundamentally different from `atomicAdd`-based
  float sums, whose result depends on the (nondeterministic) thread order (see
  PATTERNS.md §3 and flagship 5.01/11.09). Because our reduction is an exact `min`
  over the *same fixed candidate set* on both sides, the GPU reproduces the CPU
  **bit-for-bit**.
- **Race conditions.** None. Each thread writes one distinct output voxel; inputs
  are read-only.
- **Pass-rate stability.** The pass-rate is an integer ratio; we print it in
  tenths-of-a-percent computed from integer counts, and the gamma-map slice is
  printed as integer *milli-gamma* (γ×1000), so **stdout never wobbles** in a last
  decimal place — a requirement for the demo's byte-exact diff.

## 6. How we verify correctness

**Independent baseline.** `reference_cpu.cpp` recomputes the entire gamma map with
plain nested loops — no CUDA, no parallelism. It is short enough to read and
believe. The GPU kernel adds only the thread decomposition around the *same*
`gamma_sq_term()` from `gamma_core.h`.

**Two checks, both in `main.cu`:**

1. **Map agreement:** `max_abs_err(γ_cpu, γ_gpu) ≤ 1e-6`. The tolerance is a tiny
   epsilon kept only as insurance against a compiler contracting a multiply-add
   differently on one side; the **observed error is exactly `0.000000e+00`**.
2. **Statistics agreement:** the integer counts (analyzed, passing) and the
   pass-rate must match *exactly* between CPU and GPU.

**Why this is convincing.** Two independently written implementations — one
serial on the CPU, one massively parallel on the GPU — landing on the *identical*
gamma map, plus a sample whose answer we *engineered* (a known localized hot spot
that must fail while the biased-but-in-tolerance background passes), is strong
evidence the computation is right. The synthetic sample also validates the
*science*, not just CPU==GPU: the +1.5% global bias stays within the 3% criterion
(pass), while the +12% hot spot exceeds 3%/3 mm (fail), yielding the expected 99.7%
pass-rate with γ_max = 1.141 in the hot-spot region.

**Edge cases exercised:** below-threshold background (γ set to 0, excluded from
the pass-rate on both sides); window clamping at the grid boundary; the ragged
last block on the GPU.

## 7. Where this sits in the real world

Production gamma tools (PyMedPhys, Plastimatch, commercial QA software like
SNC Patient / Verisoft / Mobius) go beyond this teaching version:

- **3-D and true DICOM I/O.** They read RTDOSE volumes, **resample** the evaluated
  and reference grids onto a common frame, and search in 3-D. Here we assume 2-D
  maps already on the same grid; the extension is one more nested loop (Exercise 4).
- **Sub-voxel DTA via interpolation.** They bilinearly/trilinearly interpolate the
  evaluated grid so the DTA is continuous rather than quantized to the voxel pitch,
  and some use the analytic gradient method of Low & Dempsey to avoid the discrete
  search entirely (Exercise 3).
- **Local vs. global normalization**, dose thresholding, and **maximum-γ / mean-γ**
  reporting, all configurable per protocol (Exercise 2).
- **Scale.** Clinical grids are 512² per slice × dozens of slices; the GPU's
  advantage (the 100–1000× the catalog cites, and the Gu et al. result) appears at
  that scale, where the naive CPU search takes minutes. On our 32×32 demo the GPU
  is launch/copy-bound and *slower* — the timing line is a teaching artifact, not a
  benchmark (CLAUDE.md §12).

The core idea, though, is exactly what this project implements: one independent
min-search per reference voxel, mapped one-thread-per-voxel onto the GPU.

---

## References

- **Low DA, Harms WB, Mutic S, Purdy JA. "A technique for the quantitative
  evaluation of dose distributions." Med Phys 25(5):656–661, 1998.**
  [doi:10.1118/1.598248](https://doi.org/10.1118/1.598248) — the original gamma
  index definition; read for the DD/DTA motivation and the ellipse picture.
- **Miften M, et al. "Tolerance limits and methodologies for IMRT measurement-based
  verification QA: AAPM TG-218." Med Phys 45(4):e53–e83, 2018.** — the modern
  standard for how gamma is used clinically (criteria, thresholds, pass-rate action
  levels).
- **Gu X, Jia X, Jiang SB. "GPU-based fast gamma index calculation." Phys Med Biol
  56(5):1431, 2011.** [PMID 21317484](https://pubmed.ncbi.nlm.nih.gov/21317484/) —
  the one-thread-per-reference-point GPU gamma with geometric search-space
  reduction; the direct inspiration for this project's GPU mapping.
- **[PyMedPhys](https://github.com/pymedphys/pymedphys)** — read `gamma_shell` for a
  clean, well-tested reference implementation and its interpolation handling.
- **[Plastimatch](https://plastimatch.org/)** — study its `gamma` tool for DICOM
  RTDOSE handling and grid resampling.
