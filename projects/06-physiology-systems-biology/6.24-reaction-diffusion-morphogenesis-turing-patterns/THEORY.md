# THEORY — 6.24 Reaction-Diffusion Morphogenesis (Turing Patterns)

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

How does a nearly-uniform ball of cells decide *where* to put a stripe, a spot, a
finger, or a follicle? A developing embryo has no blueprint drawn on it; the
periodic structures of an animal (a zebra's stripes, a leopard's rosettes, the
even spacing of hair follicles, the pre-pattern that becomes five digits) must
**self-organize**. In 1952 Alan Turing proposed a startlingly simple mechanism:
suppose two diffusible chemicals — he called them **morphogens** — react with each
other, and one diffuses much faster than the other. He proved that such a system,
starting from a *stable, uniform* state, can become **unstable to spatial
perturbations** and settle into a stationary, periodic pattern. The uniform state
is stable if you nudge the *whole* domain up or down together, but unstable if you
nudge it *spatially* — a **diffusion-driven instability**, deeply counterintuitive
because diffusion normally smooths things out.

The biological reading is **short-range activation, long-range inhibition**:

- An **activator** `a` promotes its own production (autocatalysis) — a local spike
  of activator grows.
- The activator also produces an **inhibitor** `h`, which **diffuses away fast**
  and suppresses activation in the surrounding neighbourhood.

So a peak of activator reinforces itself locally while forbidding neighbouring
peaks — spacing them out into a regular pattern with a characteristic wavelength.
Turing patterns are now an accepted model for zebrafish pigment stripes, mouse
hair-follicle spacing, palatal ridges, digit patterning, and have been invoked
for cortical folding. This project builds the smallest faithful version of that
mechanism so you can watch a pattern crystallize from noise.

## 2. The math

We use the **Gierer–Meinhardt** activator–inhibitor model (the textbook Turing
system). Let `a(x,y,t) ≥ 0` be the activator and `h(x,y,t) ≥ 0` the inhibitor on a
2-D domain. The governing reaction–diffusion PDEs are:

```
∂a/∂t = Da ∇²a + ρ a²/h − μ_a a + ρ_a
∂h/∂t = Dh ∇²h + ρ a²     − μ_h h
```

Symbols (all non-negative), with the units used in the code (dimensionless grid
spacing `Δx = 1`, dimensionless time):

| symbol | meaning | sample value |
|--------|---------|--------------|
| `Da`   | activator diffusion coefficient (small) | `0.02` |
| `Dh`   | inhibitor diffusion coefficient (large) | `0.5`  |
| `ρ`    | reaction strength (autocatalysis + inhibitor production) | `0.05` |
| `μ_a`  | activator linear decay rate | `0.1`  |
| `μ_h`  | inhibitor linear decay rate | `0.14` |
| `ρ_a`  | basal (constant) activator source | `0.0` |
| `∇²`   | Laplacian (diffusion operator) | — |

Reading the reaction terms:

- `ρ a²/h` — the activator **autocatalyses** (the `a²` is positive feedback) but is
  **damped by the inhibitor** (divide by `h`). *Short-range activation.*
- `ρ a²` in the `h` equation — the activator **produces the inhibitor**, which then
  diffuses away quickly (`Dh ≫ Da`). *Long-range inhibition.*
- `−μ_a a`, `−μ_h h` — linear decay of each species.
- `ρ_a` — a small basal activator source (optional; keeps `a` from collapsing).

**The homogeneous steady state.** Setting `∂/∂t = 0` and `∇² = 0`:

```
0 = ρ a*² − μ_h h*        ⇒  h* = (ρ/μ_h) a*²
0 = ρ a*²/h* − μ_a a* + ρ_a
```

Substituting `h*` into the second equation, `ρ a*² / ((ρ/μ_h) a*²) = μ_h`, so it
collapses to `μ_h − μ_a a* + ρ_a = 0`, giving the **exact** fixed point

```
a* = (μ_h + ρ_a) / μ_a          h* = (ρ/μ_h) a*²
```

For the sample: `a* = 0.14/0.1 = 1.4`, `h* = 0.05·1.4²/0.14 = 0.7`. The code seeds
the fields at `(a*, h*)` plus tiny noise (`src/turing.h::tu_baseline_activator`,
`tu_baseline_inhibitor`; `src/reference_cpu.cpp::init_fields`).

**Turing instability (dispersion relation).** Linearize about `(a*, h*)` with a
perturbation `∝ e^{λt} cos(k·x)`. The reaction Jacobian is

```
J = [ f_a  f_h ] = [ 2ρa*/h* − μ_a     −ρa*²/h*² ]
    [ g_a  g_h ]   [   2ρa*              −μ_h      ]
```

Adding diffusion shifts the diagonal by `−D k²` for a mode of wavenumber `k`, so
the growth rate `λ(k²)` is the larger eigenvalue of

```
M(k²) = [ f_a − Da k²      f_h        ]
        [   g_a          g_h − Dh k²  ]
```

i.e. `λ = tr/2 + √((tr/2)² − det)`. A **Turing instability** requires the uniform
mode `k=0` to be **stable** (`tr(J) < 0` and `det(J) > 0`) while `λ(k²) > 0` for
some band of `k² > 0` — which needs `Dh/Da` large enough. The **fastest-growing
mode** `k*` sets the pattern wavelength `2π/k*`. For the sample, theory predicts
`Turing regime = YES`, `λ_max ≈ 0.040` at `k* ≈ 1.14` → wavelength `≈ 5.5` cells;
`src/main.cu::turing_growth_rate` computes exactly this and the simulation
reproduces it.

**Discretization.** The Laplacian uses the standard 5-point stencil on a unit grid
with **periodic** boundaries (a torus — no edges, matching an animal's flank):

```
∇²f[x,y] ≈ f[x−1,y] + f[x+1,y] + f[x,y−1] + f[x,y+1] − 4 f[x,y]
```

and time advances by **explicit (forward) Euler**: `f_new = f + Δt·(D ∇²f +
reaction)`. See `src/turing.h::tu_laplacian` and `tu_update`.

## 3. The algorithm

```
load parameters (nx, ny, Da, Dh, ρ, μ_a, μ_h, ρ_a, Δt, steps, seed)
seed a[] = a* + tiny_deterministic_noise(x,y,seed);  h[] = h*
repeat `steps` times:
    for every cell (x,y):
        la = laplacian(a, x, y);  lh = laplacian(h, x, y)
        a_new = a + Δt·(Da·la + ρ a²/h − μ_a a + ρ_a)
        h_new = h + Δt·(Dh·lh + ρ a²     − μ_h h)
    swap (a, a_new) and (h, h_new)          # ping-pong
report pattern metrics + analytic dispersion-relation prediction
```

**Complexity.** Each cell update is `O(1)` (4 neighbour reads + a handful of
flops). One sweep is `O(N)` for `N = nx·ny` cells; the whole run is
`O(steps · N)`. Serial cost for the sample is `3000 · 4096 ≈ 1.2·10⁷` cell
updates. **Arithmetic intensity is low** — a few flops per ~5 memory reads — so the
kernel is **memory-bandwidth bound**, the typical stencil story. The parallel
*work* is the same `O(steps·N)`, but the parallel *depth* is `O(steps)`: within a
step all `N` cells are independent, so an ideal machine does each sweep in `O(1)`
depth. That independence is exactly what the GPU exploits.

## 4. The GPU mapping

**Thread-to-data mapping.** One thread owns one grid cell:

```
x = blockIdx.x·blockDim.x + threadIdx.x    (column)
y = blockIdx.y·blockDim.y + threadIdx.y    (row)
cell index i = y·nx + x   (row-major)
```

**Launch configuration.** A 2-D block of `16×16 = 256` threads (a warp multiple; 8
warps give the scheduler latency to hide; a square tile keeps a cell's 4 neighbours
spatially close so they hit the same cache lines). The grid is
`(⌈nx/16⌉, ⌈ny/16⌉)` blocks; border blocks have threads with `x≥nx`/`y≥ny` that
return early (the ragged-edge guard).

```
      grid of 16x16 blocks over the nx x ny domain
   +--------+--------+--------+ ...
   | block  | block  | block  |     each block = 16x16 threads
   | (0,0)  | (1,0)  | (2,0)  |     each thread = one cell (x,y)
   +--------+--------+--------+ ...  reads its 4 neighbours from
   | block  | block  | ...          the "source" buffer, writes
   | (0,1)  | (1,1)  |              its own cell in the "dest" buffer
   +--------+--------+ ...
```

**Ping-pong double buffering.** The PDE update is *simultaneous*: every cell must
read the **frozen previous** state. We keep two device buffer pairs
(`a_src/a_dst`, `h_src/h_dst`); the kernel reads `src`, writes `dst`; the host
swaps the pointers between launches (a swap moves *no* data). Because each cell is
written by exactly one thread and neighbours are read-only within a step, there are
**no data races and no atomics** — the cleanest possible parallel pattern (contrast
the Monte-Carlo/k-means projects that *do* need atomics). See
`src/kernels.cu::simulate_gpu`.

**Memory hierarchy.** This teaching version reads neighbours straight from
**global** memory and relies on the L1/L2 cache for neighbour reuse. The catalog's
suggested optimizations — **texture memory** for the read-only species arrays and
**shared-memory tiling** (load a 16×16 tile + 1-cell halo into `__shared__`, so the
`4×` neighbour reuse hits shared memory instead of global) — are real speed-ups on
large grids; we leave them as exercises so the core stencil stays legible. There is
no CUDA library call here: the stencil is hand-written (the catalog mentions Thrust
only for an optional global mass-conservation reduction, which we compute serially
in `main.cu` for a tiny grid).

**Timing.** We time only the stepping loop with CUDA events (`util/timer.cuh`),
excluding the one-time host↔device copies. On the sample the GPU is a few times
faster than the CPU; its edge grows with grid size and step count. This is a
**teaching artifact, not a benchmark** (CLAUDE.md §12).

## 5. Numerical considerations

- **Precision: FP64 (`double`) throughout.** Turing patterns are sensitive to the
  small growth rates near the instability threshold, and double precision keeps the
  CPU and GPU results close enough to verify tightly. (FP32 would still form
  patterns but drift faster; trying it is a good exercise.)
- **Explicit-Euler stability.** Forward Euler on the diffusion term is stable only
  if `Δt · D · 4 ≲ 1` (the 2-D diffusion CFL condition; the `4` is the stencil's
  centre weight). The binding constraint is the *fast* inhibitor: `Δt·Dh·4 =
  0.4·0.5·4 = 0.8 < 1` for the sample. A larger `Δt` (or `Dh`) makes the field
  oscillate and blow up to `NaN` — try it to see the failure mode.
- **Determinism.** There are **no atomics and no floating-point reductions inside
  the time loop**, so each cell's update is a fixed sequence of operations — the
  GPU result is bit-reproducible run to run (verified: two runs produce identical
  stdout). The *only* CPU↔GPU divergence comes from the GPU fusing `a*b+c` into one
  FMA while the host does two rounded operations; this is `~1e-13` per step.
- **Reproducible seeding.** The initial noise is a **hash** of `(x, y, seed)`
  (splitmix-style bit mixer in `tu_perturbation`), not `rand()`, so the initial
  condition is byte-identical on every machine and on both the CPU and GPU paths —
  essential for an exact comparison and a stable `expected_output.txt`
  (PATTERNS.md §3).

## 6. How we verify correctness

Two independent checks (PATTERNS.md §4):

1. **GPU vs CPU reference.** `src/reference_cpu.cpp` runs the identical
   `tu_update()` from `src/turing.h` in plain serial C++ — the same arithmetic in
   the same order — using the same deterministic seed. `main.cu` compares the final
   `a` and `h` fields cell-by-cell and asserts the worst absolute difference is
   `≤ 1e-6`. Why `1e-6` and not `0`? Over 3000 nonlinear steps the GPU's FMA and the
   host compiler's separate multiply+add diverge by `~1e-13` per step, which
   accumulates; the field values are `O(1)`, so `1e-6` is physically negligible
   (far below "same pattern") yet strict enough to catch a real indexing or physics
   bug. Observed worst diff: `~4e-12` — comfortably inside tolerance.
2. **Simulation vs analytic theory.** Independently of the sim, `turing_growth_rate`
   computes the dispersion relation `λ(k²)` and reports whether the parameters lie
   in the pattern-forming regime and the predicted wavelength. For the sample it
   says `Turing regime = YES`, `k* ≈ 1.14`; the sim produces a high-contrast pattern
   (contrast `≈ 7.13`, far from flat). Theory and simulation agreeing validates the
   *science*, not merely that two implementations match.

Edge cases guarded: divide-by-`h` is floored (`h_safe`), `μ_a`/`μ_h`→0 are floored
in the steady-state helpers, the loader rejects non-positive `Δt`, `nx,ny ≤ 2`, and
negative rates.

## 7. Where this sits in the real world

Production reaction-diffusion / morphogenesis tools go far beyond this stencil:

- **VCell** (vcell.org) solves reaction-diffusion PDEs on **realistic cell
  geometry** (membranes, compartments, flux boundary conditions), with both
  deterministic and **spatial-stochastic (Smoldyn/Gillespie)** solvers, and
  implicit/adaptive time-stepping for stiff kinetics — none of which this explicit
  fixed-`Δt` teaching version has.
- **MOOSE** (BhallaLab) does compartmental spatial simulation on **unstructured
  meshes** (neuronal morphologies), vs our regular grid.
- **GillesPy2** simulates **stochastic Turing patterns** via the reaction-diffusion
  master equation — the regime where molecular noise, not deterministic dynamics,
  *selects* which pattern appears; our deterministic sim omits this entirely.
- Real biology also involves **3-D surfaces** (level-set reaction-diffusion on a
  curved tissue), **domain growth** (the pattern rescales as the embryo grows),
  and **parameter sweeps** across the `(Dh/Da, ρ, μ)` space to map pattern-forming
  regions — the GPU's real payoff, since each parameter point is an independent
  simulation (an embarrassingly-parallel ensemble, cf. flagships `9.02`/`13.02`).

The catalog's full brief (3-D stencils, texture memory, shared-memory 7-point
tiles, Thrust mass-conservation reduction) is the natural extension path; this
project deliberately ships the smallest correct 2-D version and points at the rest.

---

## References

- **A. M. Turing, "The Chemical Basis of Morphogenesis," Phil. Trans. R. Soc. B
  (1952).** The founding paper; the diffusion-driven-instability argument.
- **A. Gierer & H. Meinhardt, "A theory of biological pattern formation,"
  Kybernetik (1972).** The activator–inhibitor model implemented here.
- **J. D. Murray, *Mathematical Biology II* (Springer).** The standard textbook
  derivation of the Turing conditions and dispersion relation.
- **S. Kondo & T. Miura, "Reaction-Diffusion Model as a Framework for
  Understanding Biological Pattern Formation," Science (2010).** Modern biological
  evidence (zebrafish, etc.).
- **NVIDIA cuda-samples** — hand-written finite-difference/stencil kernels and
  shared-memory tiling; the reference for the exercise-2 optimization.
- **VCell / MOOSE / GillesPy2** — the production tools contrasted in §7.
