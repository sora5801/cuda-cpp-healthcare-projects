# THEORY — 5.15 Proton CT & Ion Imaging Reconstruction

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use. This is a reduced-scope 2-D teaching
> version of a 🔴 frontier problem (see §7)._

---

## 1. The science

**The clinical problem.** Proton therapy kills tumours by placing the proton
beam's *Bragg peak* — the sharp dose maximum at the end of a proton's range —
inside the tumour. Where that peak lands depends on the **relative stopping
power (RSP)** of every tissue the beam crosses (RSP = how fast that tissue slows
a proton, relative to water). Getting RSP wrong by a few percent moves the Bragg
peak by millimetres, potentially into healthy tissue or past the tumour.

Today RSP is obtained *indirectly*: an ordinary **X-ray CT** measures photon
attenuation (Hounsfield units, HU), and a calibration curve converts HU → RSP.
That conversion carries a **~3% range uncertainty**, the dominant geometric
uncertainty in proton treatment planning.

**Proton CT (pCT) removes the conversion.** Instead of X-rays, pCT sends
**protons at therapeutic energy straight through the patient** and measures each
one's *residual energy* on the far side. Residual energy → residual range →
**water-equivalent path length (WEPL)**: the integral of RSP along that proton's
path. Reconstruct RSP from many WEPL measurements and you have *exactly* the
quantity treatment planning needs — measured, not converted.

**Why pCT is its own algorithm.** X-rays travel in straight lines; a proton does
not. Multiple Coulomb scattering off atomic nuclei nudges it thousands of times,
so it follows a slightly **curved path**. Given a proton's measured entry and
exit position **and direction** (from tracker planes front and back), the best
estimate of where it actually went is the **most-likely path (MLP)** — a curved
trajectory. Reconstruction must integrate and backproject along that *curve*,
which is what makes pCT reconstruction distinct from X-ray CT (project 4.01).

---

## 2. The math

**Unknowns.** A 2-D RSP image `x ∈ R^(n·n)`, `x_v ≥ 0` = RSP of voxel `v`
(dimensionless, water = 1). (We do one slice; real pCT is 3-D but the math per
slice is identical — §7.)

**Each proton = one linear equation.** For proton `p`, its measured WEPL is the
line integral of RSP along its MLP `Γ_p`:

```
    ∫_{Γ_p} x(s) ds  =  WEPL_p                      (continuous form)
```

Discretised on the voxel grid, the integral becomes a weighted sum:

```
    Σ_v  a_pv · x_v  =  b_p ,      b_p = WEPL_p       (row p of A x = b)
```

where `a_pv` = the path length proton `p` spends in voxel `v` (cm). Stack all
protons and you get a huge, **sparse, over-determined** linear system

```
    A x = b ,   A ∈ R^(P×V),  P ≈ 10^8 protons,  V = n·n voxels.
```

Each row `a_p` is nearly empty (a proton touches only the ~`n` voxels along its
path), and `A` is far too large to form explicitly — it is applied *matrix-free*
by walking each proton's MLP.

**The most-likely path.** The rigorous MLP (Schulte et al. 2008) maximises the
likelihood of the measured entry `(y_0, θ_0)` and exit `(y_1, θ_1)` under a
Gaussian multiple-Coulomb-scattering model. The scattering angle variance for a
step is the **Highland formula**

```
    σ_θ²(ℓ) ≈ ( 13.6 MeV / (β p c) )² · (ℓ / X_0) · [1 + 0.038 ln(ℓ/X_0)]²
```

(`p` = momentum, `βc` = velocity, `X_0` = radiation length). Propagating that
covariance and combining the two boundary conditions yields a trajectory that is
**cubic in depth**. In the small-angle, near-uniform limit that cubic is exactly
the **cubic Hermite curve** matching both endpoints' *position* and *slope* —
which is what we implement (`src/pct_physics.h::mlp_point`):

```
    h(t) = (t³ − 2t² + t)·m₀ + (t³ − t²)·m₁,   t ∈ [0,1]
    position(t) = chord(t) + h(t)·v̂
```

with `m₀ = tan(θ_0)·L`, `m₁ = tan(θ_1)·L`, `L` = entry→exit chord length, `v̂` =
unit normal to the chord. `h(0)=h(1)=0` (proton is on the chord at both tracker
planes); the prescribed slopes bend it into the characteristic S-shape.

**Objective.** Solve `A x = b` in the least-squares / feasibility sense, with the
physical constraint `x_v ≥ 0` (RSP is non-negative). We use SART (below); POCS
adds the `x ≥ 0` and box constraints as projections (§7).

---

## 3. The algorithm

We solve `Ax=b` with **SART** (Simultaneous Algebraic Reconstruction Technique),
a row-action iterative solver ideal for the matrix-free, list-mode setting.

One **sweep** (iteration) updates every voxel using *all* protons:

```
  for each proton p:                                     # forward + residual
      est_p = Σ_v a_pv x_v            (integrate RSP along the MLP)
      r_p   = b_p − est_p            (WEPL residual, cm)
      L_p   = Σ_v a_pv               (proton's in-grid path length, ||a_p||₁)
      for each voxel v on p's path:                       # scatter correction
          num_v += a_pv · (r_p / L_p)
          den_v += a_pv
  for each voxel v:                                       # apply update
      x_v += λ · num_v / den_v        (λ = relaxation, 0 < λ ≤ 1)
```

This is the standard SART step `x_v ← x_v + λ · [Σ_p a_pv (b_p−a_p·x)/‖a_p‖₁] /
[Σ_p a_pv]`. We sample each MLP at `path_samples` midpoints and set
`a_pv = seg_len = L_chord / path_samples` for every sample landing in voxel `v`
(a nearest-voxel projector). More samples → a finer quadrature of `∫ RSP ds`.

**Complexity.** Per sweep: `O(P · path_samples)` work (each proton walks its path
twice — once to project, once to scatter), plus `O(V)` for the update. Total
`O(iters · P · path_samples)`. The serial CPU cost is exactly this; the parallel
version does the `P` protons of each sweep concurrently, so its *depth* per sweep
is `O(path_samples)` plus the atomic contention on shared voxels.

**Data-access pattern.** Each proton reads ~`path_samples` scattered voxels
(a **gather**, like CT backprojection 4.01) and writes ~`path_samples` scattered
voxels (a **scatter** with contention, like the k-means centroid accumulation
11.09 or Monte-Carlo tallying 5.01). Low arithmetic intensity → memory/atomic
bound, exactly the regime GPUs with high bandwidth win.

---

## 4. The GPU mapping

**Thread-to-data map.** Within one sweep every proton is independent, so we give
**one GPU thread per proton**:

```
    thread i  = blockIdx.x · blockDim.x + threadIdx.x   →   proton i
```

Two kernels per sweep (`src/kernels.cu`):

- **`tally_kernel`** — thread `i` forward-projects proton `i` along its MLP,
  forms the residual, and `atomicAdd`s a fixed-point correction into the shared
  `num`/`den` accumulators. Block = 256 threads (multiple of the 32-lane warp,
  enough warps to hide global-memory latency); grid = `ceil(P/256)`.
- **`update_kernel`** — thread `v` owns voxel `v`, applies `x_v += λ·num/den`.
  Grid = `ceil(V/256)`.

The host driver `reconstruct_gpu` loops these `iters` times. The RSP image and
accumulators **live on the device across all sweeps** — only the final image is
copied back, so there is no per-sweep PCIe traffic.

```
   protons[]            RSP image (device, persists across sweeps)
   ┌───┬───┬───┐        ┌───────────────┐
   │p0 │p1 │p2 │ ...    │  n × n voxels │
   └─┬─┴─┬─┴─┬─┘        └───────────────┘
     │   │   │   one thread per proton         ▲ gather (read RSP along MLP)
     ▼   ▼   ▼                                  │
   tally_kernel  ── atomicAdd (fixed point) ──► num[V], den[V]   ◄ scatter
                                                     │
   update_kernel: thread per voxel  x_v += λ·num/den ┘
```

**Memory hierarchy.**
- **Global memory** holds `protons`, `rsp`, and the two `int64` accumulators —
  the working set is far too large for shared memory, and each proton's voxels
  are data-dependent (its MLP), so there is no reuse pattern to tile.
- **Registers** hold each thread's running `est`, `n_hit`, `resid` — the hot
  inner loop is register-resident.
- **Constant/texture** are *not* used here (a production projector would put the
  RSP image in a **texture** for hardware bilinear interpolation and caching —
  see 4.01 and the exercises).

**Where the catalog's libraries would go (no black boxes).** The catalog lists
cuRAND, Thrust sort, and cuBLAS for a full pipeline. This teaching version does
not need them, and the comments say why:
- **cuRAND** — only for *simulating* a proton beam (Monte-Carlo history
  generation). Our data is pre-generated synthetically, so no RNG at run time.
- **Thrust `sort`** — a full pCT bins protons by projection angle for an FBP-style
  reconstruction; our list-mode SART touches every proton every sweep in input
  order, so no sort is needed. (`thrust::sort_by_key` on the angle would be the
  one-liner; hand-rolling it is a radix/merge sort — see 3.15/12.xx.)
- **cuBLAS** — used in the rigorous MLP to update scattering **covariance
  matrices** per step (small dense 2×2/4×4 solves). Our closed-form cubic Hermite
  MLP avoids the per-proton matrix solve entirely, so cuBLAS is unnecessary here.

---

## 5. Numerical considerations

**Precision.** The RSP image and the MLP geometry are **FP32** — pCT WEPL
measurements are ~mm-accurate, far coarser than float precision, so FP64 buys
nothing for the physics. The **accumulation** and the fixed-point conversion use
**FP64/int64** to keep the reduction exact (below).

**The determinism problem.** Many protons cross the same voxel, so the scatter is
a **many-writer reduction** into `num`/`den`. Floating-point `atomicAdd` is
**not associative**: the sum depends on the (hardware-nondeterministic) order in
which threads arrive, so two runs — or the CPU vs the GPU — would differ. That
would make the demo's stdout unreproducible.

**The fix (docs/PATTERNS.md §3).** Accumulate in **fixed-point integers**: store
`value · 1e6` rounded to `int64`. Integer addition *commutes*, so the tally is
**order-independent** — identical regardless of thread scheduling, and identical
between CPU and GPU. Both sides use the same `llround(double · 1e6)` conversion,
so the accumulators match bit-for-bit. `int64` has ample headroom for ~10⁴
protons × order-1 corrections.

**The one residual float caveat.** The *forward projection* `est += rsp[v]·seg`
is a float sum, and host vs device compilers may contract `a·b+c` into an FMA
differently, differing by ~1 ULP. Over `iters` sweeps that can nudge a voxel by
a tiny amount, and — right at a rounding boundary — flip one fixed-point
increment. So we do **not** claim bit-identical images; we verify to a small
**physical tolerance** (§6). This is the honest situation for any long iterative
solver (see 10.02, 14.02).

**Constraints.** This teaching SART does not clamp `x_v ≥ 0`, so you can see the
characteristic ART edge ringing (small negative RSP just outside the object) —
a real artefact worth seeing. Adding the `x ≥ 0` projection is the "POCS" of the
catalog and a one-line exercise.

---

## 6. How we verify correctness

**Two independent checks.**

1. **GPU vs CPU.** `src/reference_cpu.cpp` runs the *same* SART, serially, using
   the *same* shared physics (`pct_physics.h`) and the *same* fixed-point tally.
   `main.cu` computes `max_abs_err(GPU, CPU)` over all `n·n` voxels and requires
   it below **`1.0e-3`** RSP units. The tolerance is a *physical* one, chosen for
   the FMA caveat in §5 — the *observed* error is ~`1e-6` (printed on stderr),
   i.e. the two agree to ~6 digits. An independent serial re-derivation matching
   the parallel one to 6 digits is strong evidence both implement the same math.

2. **Recovery of a known answer** (docs/PATTERNS.md §6). The synthetic data is
   generated from a *known* phantom (water disc + dense + light inserts). The
   demo reports **RMSE vs. ground truth** (`0.0935`) and the **mean RSP inside
   the phantom** (`1.0244` vs. true `1.057`), plus a central-row profile that
   peaks at `1.5959` through the RSP-1.6 insert. This validates the *science*
   (SART recovers the phantom), not just CPU==GPU agreement.

**Edge cases handled:** protons that miss the grid entirely (`n_hit == 0` → no
contribution), untouched voxels in a sweep (`den == 0` → no update), degenerate
zero-length chords (guarded in `mlp_point`).

---

## 7. Where this sits in the real world

This is a deliberately **reduced-scope, 2-D teaching version** of a 🔴 frontier
problem. A production pCT reconstruction differs by:

- **Scale & speed.** Clinical scanners (IBA, PRaVDA) produce ~**10⁸ proton
  events/second**; reconstruction is 3-D over ~512³ voxels. GPU is mandatory —
  our 1440-proton 2-D slice is a toy to keep the geometry legible.
- **The rigorous MLP.** Real code uses the full Schulte covariance MLP (with the
  Highland formula and per-step scattering matrices — where the catalog's cuBLAS
  goes), not a closed-form Hermite. The Hermite is the correct small-angle limit
  and captures the curvature, but the covariance form weights positions by their
  scattering uncertainty.
- **Better projectors.** Siddon/bilinear path-length weighting instead of our
  nearest-voxel sampling; the RSP image lives in a **texture** for cached
  hardware interpolation.
- **Constraints & regularisation.** POCS with RSP **box constraints** (`x ≥ 0`,
  tissue-range bounds) and scattering/TV regularisation for noise; list-mode
  variants CSPACS / MLSD.
- **FBP variant.** Alternatively, bin protons by angle (Thrust sort) and run a
  distance-driven filtered backprojection along MLPs.

Prior art to study: **TOPAS/GATE** (Monte-Carlo simulation to *generate* pCT
data), **FRED** (fast GPU proton transport), and the **UCSC/Baylor pCT**
reconstruction codes.

---

## References

- R. W. Schulte et al., "A maximum likelihood proton path formalism for
  application in proton computed tomography," *Med. Phys.* 35(11), 2008 — the MLP
  our Hermite curve approximates.
- V. L. Highland, "Some practical remarks on multiple scattering," *Nucl. Instr.
  Meth.* 129, 1975 — the scattering-angle formula behind the MLP covariance.
- A. C. Kak & M. Slaney, *Principles of Computerized Tomographic Imaging*, ch. 7
  (ART/SART) — the iterative reconstruction we implement.
- G. Poludniowski et al., "Proton radiography and tomography with application to
  proton therapy," *Br. J. Radiol.* 88, 2015 — accessible pCT overview.
- TOPAS (<https://github.com/OpenTOPAS/OpenTOPAS>), FRED
  (<https://www.fredonline.eu/>) — simulation/transport tools to make real data.
