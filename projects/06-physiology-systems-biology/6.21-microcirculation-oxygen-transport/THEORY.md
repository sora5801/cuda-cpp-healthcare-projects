# THEORY — 6.21 Microcirculation & Oxygen Transport

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

Every cell in the body needs oxygen, and oxygen has a problem: it is delivered by
blood but consumed in tissue, and the two are separated by distance. Blood flows
through **capillaries** — vessels a few micrometres wide — carrying oxygen bound to
**hemoglobin** inside red blood cells. Oxygen leaves the blood, crosses the vessel
wall, and **diffuses** through the tissue to reach the cells, which **consume** it
to make ATP. Diffusion is slow and short-ranged, so a cell more than ~100–200 µm
from the nearest capillary can starve for oxygen even while the blood right next to
another vessel is fully oxygenated. Regions that fall below a critical PO2 become
**hypoxic** — the mechanism behind tumour hypoxia (which blunts radiotherapy),
ischemic tissue damage, and the design constraints on engineered tissue.

The quantitative question is therefore spatial: **given where the capillaries are
and how oxygenated their blood is, what is the oxygen partial pressure (PO2) at
every point in the tissue, and where are the hypoxic pockets?** August Krogh won a
Nobel Prize (1920) for the first model of this — the **Krogh cylinder**, a single
capillary supplying a concentric cylinder of tissue. Modern work (Secomb, Hsu,
Popel) generalises Krogh to arbitrary 3-D capillary networks via the **Green's-
function method**, which is what this project implements in reduced-scope form.

## 2. The math

**Governing equation.** At steady state, oxygen concentration `C(x)` in tissue obeys
a reaction–diffusion (Poisson-type) equation: diffusion balances consumption plus
the sources supplied by capillaries:

```
D ∇²C(x) = M(x) − Σ_j q_j δ(x − x_j)
```

- `D` — oxygen diffusivity in tissue (µm²/s).
- `C(x)` — O2 concentration; related to partial pressure by Henry's law
  `C = α·P`, where `α` is the solubility and `P` is PO2 (mmHg).
- `M(x)` — local consumption rate (a sink).
- `q_j`, `x_j` — strength and position of capillary segment `j` (a point source).

**Green's function.** The Green's function `G` solves `D ∇²G = −δ(x)`; in free
3-D space it is the familiar `1/r` kernel (identical in form to the electrostatic
potential of a point charge — hence APBS, an electrostatics solver, is listed as
reusable). Converting concentration to partial pressure and lumping `K = D·α`:

```
G(r) = 1 / (4π K r),      r = |x − x_j|
```

By **linearity** (the Laplacian is linear), the total field is the superposition of
each source's Green's function plus the background:

```
P(x_i) = P_inflow + Σ_j q_j · G(|x_i − x_j|) − M(P_inflow)          (★)
```

This equation (★) is exactly what `solve_point()` computes for one grid point `i`.

**Biology in the coefficients.**
- **Hill hemoglobin saturation** sets how much O2 a segment can release:
  `S(P) = Pⁿ / (P50ⁿ + Pⁿ)`, `P50 ≈ 26 mmHg`, `n ≈ 2.7`. We take `q_j ∝ S(blood_PO2_j)`.
- **Michaelis-Menten consumption** is the demand: `M(P) = M0·P/(P + Km)` — saturating
  at high PO2, vanishing as `P → 0` so tissue never "consumes oxygen that isn't there".

**Regularization.** `G(r) → ∞` as `r → 0`. Physically a capillary has radius
`R_cap`, so we evaluate `G` at `max(r, R_cap)`; the field is ~flat inside the vessel
core. This keeps (★) finite.

**Inputs:** grid (`nx,ny,nz,spacing`), physiology (`P_inflow, M0, Km, P50, n`), and
`n_src` segments (`x,y,z,blood_PO2`). **Output:** PO2 (mmHg) at all `nx·ny·nz` points.

## 3. The algorithm

```
load problem (grid + physiology + segments)
for each segment j:                      # done once, host side
    q_j = q_scale · Hill(blood_PO2_j, P50, n)
for each grid point i:                   # the parallel loop
    P_i = P_inflow
    for each source j:                   # the gather
        P_i += q_j · G(|x_i − x_j|)
    P_i -= M(P_inflow, M0, Km)
    P_i = max(P_i, 0)
```

**Complexity.** The double loop is `O(N_grid · N_src)` — an all-pairs sum. Each pair
costs one distance (a sqrt), one reciprocal, and a fused multiply-add: high
**arithmetic intensity**, low memory traffic per flop (each source is reused across
all grid points). Serial depth is `O(N_grid · N_src)`; the parallel version has
**work** `O(N_grid · N_src)` and **depth** `O(N_src)` (each thread's inner loop),
because all `N_grid` points are computed simultaneously.

This `O(N²)` scaling is the honest baseline. A realistic 1 mm³ tissue block has
~10⁴ capillary segments and ~10⁶–10⁷ grid points, so `N_grid · N_src ≈ 10¹⁰–10¹¹` —
which is why production solvers use a **fast multipole method (FMM)** to reach
`O(N log N)` (see §7).

## 4. The GPU mapping

**Thread-to-data mapping.** One thread computes one tissue grid point:
`idx = blockIdx.x · blockDim.x + threadIdx.x` owns grid point `idx` (the same linear
index `grid_point_coords()` decodes to `(ix,iy,iz)`). No inter-thread communication
for the output — each thread owns exactly one `po2_out[idx]`, so **no atomics** are
needed.

**Launch configuration.** `blockDim = 128` threads (a multiple of the 32-lane warp;
4 warps per block to hide latency). `gridDim = ceil(N_grid / 128)`. Blocks are
independent; the scheduler runs as many as fit.

**Memory hierarchy — the key optimisation.** Every one of the `N_grid` threads reads
*every* source. Read naïvely from global memory, that is `N_grid · N_src` global
loads of the source array. Instead the block **stages sources into shared memory in
tiles**: the 128 threads cooperatively load 128 sources into a `__shared__` array,
`__syncthreads()`, then every thread reads that tile from on-chip shared memory
(~100× lower latency than global), and repeat for the next tile. This is the same
idea as tiled GEMM and N-body force summation.

```
   sources in GLOBAL memory:  [ s0 s1 s2 ... s_{M-1} ]
                                 |   staged tile-by-tile
                                 v
   __shared__ tile[128]:       [ s_base ... s_{base+127} ]   <- filled by all threads, then
                                 ^  read by every thread in the block (fast, broadcast-friendly)

   grid points (one per thread):
     block 0: idx 0..127     block 1: idx 128..255   ...   (each thread sums over all tiles)
```

**Registers.** Each thread keeps its running `po2`, its `(x,y,z)`, and loop indices
in registers — the hot accumulator never touches memory until the final write.

**Why no CUDA library here.** The catalog points at cuFMM/NUFFT (for the fast sum)
and cuSPARSE (for a coupled network-flow solve). This teaching version does the
*direct* sum with a hand-written kernel precisely so the learner sees the baseline
those libraries accelerate — no black box. §7 explains what the libraries add.

## 5. Numerical considerations

- **Precision: FP64 (double).** PO2 differences between grid points can be small
  relative to the inflow baseline, and we want CPU/GPU agreement well below any
  physiological scale. Double precision costs throughput on consumer GPUs but keeps
  the teaching result clean. `sqrt`, reciprocal, and `pow` all run in double.
- **No atomics, no races.** Each output element has exactly one owning thread, so
  there is no write contention and nothing to serialise.
- **Determinism.** The inner sum over sources runs in a **fixed index order**
  (0..N_src−1, tile by tile, in order within a tile). Floating-point addition is not
  associative, so order matters — but because the CPU reference and the GPU kernel
  use the *identical* order, their double-precision partial sums agree to round-off.
  The shared-memory tiling changes *where* sources are read from, never the *order*
  they are summed. (See `docs/PATTERNS.md` §3.)
- **Clamping** `PO2 ≥ 0` is done in one shared `clamp_po2()` so both sides clamp
  identically.

## 6. How we verify correctness

`src/reference_cpu.cpp` computes the same field with a single serial loop over grid
points, calling the **same** `solve_point()` the GPU thread calls (both include
`oxygen.h` / `reference_cpu.h`; the HD-macro idiom compiles the very same code for
host and device). `main.cu` then compares the two fields point-by-point and reports
the worst absolute difference.

- **Tolerance: 1e-9 mmHg.** This is the "exact-ops" case from `docs/PATTERNS.md` §4:
  identical operations in identical order on both sides. The observed worst
  difference is ~1e-14 mmHg (pure double-precision round-off from the compiler
  scheduling FMAs slightly differently), far below 1e-9 and astronomically below any
  physiological significance (~1 mmHg).
- **Why this is convincing.** The CPU path is obviously correct (a plain loop); an
  independent-looking parallel implementation reproducing it to round-off is strong
  evidence the parallel decomposition (indexing, tiling, sync) has no bug.
- **A physical sanity check** beyond CPU==GPU: the field's spatial structure matches
  the physics — PO2 is highest near the capillaries (centre ~24 mmHg in the sample)
  and lowest in the corner farthest from every vessel (~8.4 mmHg), producing an
  engineered hypoxic pocket. A monotone-decreasing PO2 with distance from sources is
  the qualitative signature of diffusion from point sources.

## 7. Where this sits in the real world

The real Secomb–Hsu Green's-function method differs from this teaching version in
three big ways:

1. **Self-consistent source strengths.** Here `q_j` is fixed from the blood PO2 up
   front. In reality the O2 a segment releases depends on the tissue PO2 around it
   (you cannot release more O2 than the gradient allows), so the `q_j` and the tissue
   field are coupled: the real method sets up and **solves a dense linear system**
   for the `q_j` (Exercise 2 sketches the iterative version). This is where
   **cuBLAS/cuSOLVER** or an iterative solver enters.
2. **Fast multipole / NUFFT.** The `O(N²)` all-pairs sum is replaced by an FMM that
   groups distant sources into multipole expansions, reaching `O(N log N)` — essential
   at 10⁶–10⁷ grid points. **cuFMM/NUFFT** implement this; our direct sum is the
   baseline they accelerate.
3. **Coupled convective flow.** Blood PO2 is not prescribed; it drops along each
   capillary as O2 is given up (a 1-D convection–diffusion equation per segment),
   and flow splits at bifurcations by hematocrit-dependent rules. Network flow is a
   sparse linear solve (**cuSPARSE**), and RBC flux partitioning adds the
   hematocrit physics. HemeLB (LBM flow) and USERMESO (deformable RBCs) model the
   fluid/cellular side; OpenFOAM does volume-averaged continuum oxygenation.

This project deliberately keeps the *diffusion-superposition* core and the *biology
of the coefficients* (Hill, Michaelis-Menten) faithful while omitting the coupled
solves — enough to teach the GPU pattern and the physiology honestly.

---

## References

- **Krogh, A. (1919)** "The number and distribution of capillaries..." *J. Physiol.*
  — the original single-capillary-supplying-a-cylinder model.
- **Secomb, T.W., Hsu, R., et al.** Green's-function method for O2 transport in
  microvascular networks — the reference method this project reduces (see
  <https://secomb.org> for papers and the reference code; verify terms).
- **Popel, A.S. (1989)** "Theory of oxygen transport to tissue" — the review that
  frames Krogh/Green's-function/continuum approaches.
- **Hill, A.V. (1910)** the oxyhemoglobin saturation equation used here.
- **Greengard & Rokhlin (1987)** the fast multipole method — how §7's `O(N log N)`
  sum works.
- **Starter repos:** HemeLB (capillary flow, LBM), USERMESO-2.0 (GPU deformable
  RBCs), APBS (electrostatics = same Poisson math), OpenFOAM (continuum oxygenation)
  — each a different slice of the full microcirculation problem.
