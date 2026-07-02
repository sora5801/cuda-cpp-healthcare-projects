# THEORY — 6.17 Purkinje System & Conduction System Modeling

> A didactic deep-dive: the science of the cardiac conduction system, the cable
> equation, the numerical scheme, how it maps onto the GPU, how we verify it, and
> how production tools do it differently. This project ships a **reduced-scope
> teaching model** (see "Where this sits in the real world"); the code is written
> to be read, not to be clinically accurate.
>
> _Educational only — not for clinical use._

---

## The science

The heart's pump is choreographed by electricity. A specialised **conduction
system** carries the activation wavefront so that the ventricles contract as a
coordinated unit rather than a quivering bag of cells:

```
   SA node  --->  atria  --->  AV node  --->  His bundle
                                                  |
                                    +-------------+-------------+
                                    |                           |
                            left bundle branch          right bundle branch
                                    |                           |
                            Purkinje fascicles          Purkinje fascicles
                                    |                           |
                            Purkinje-muscle             Purkinje-muscle
                            junctions (PMJs)            junctions (PMJs)
```

- The **sinoatrial (SA) node** is the natural pacemaker.
- The **atrioventricular (AV) node** delays the impulse (~100 ms) so the atria
  finish emptying before the ventricles fire.
- The **His bundle** then hands off to the **bundle branches** and the fractal
  **Purkinje fibre network** — a fast highway (conduction velocity ~2–4 m/s, vs
  ~0.3–1 m/s in ordinary myocardium) that spreads activation to the whole
  endocardium in a few tens of milliseconds.
- Each **Purkinje-muscle junction (PMJ)** is where a Purkinje fibre couples into
  the working myocardium through gap-junction conductance.

Disease of this system is common and consequential: **bundle-branch block**
widens the QRS complex; **His-Purkinje conduction disease** causes syncope and
can require a pacemaker; abnormal Purkinje activity seeds **re-entry
arrhythmias** and some forms of ventricular tachycardia. Being able to simulate
how fast a wave travels down each fibre, and when each PMJ fires, is the
computational backbone of studying these pathologies.

**What we model here.** A Purkinje fibre behaves like a 1-D excitable **cable**:
inject enough current at one end and a self-sustaining action-potential wave
ignites and propagates at a fixed **conduction velocity (CV)** set by the fibre's
passive properties (diameter → axial coupling) and its active membrane kinetics.
We simulate an ensemble of such cables and a small branching graph that connects
them, then read off each cable's CV and the total time to activate the tree.

---

## The math

### The monodomain cable equation

For a thin fibre, the transmembrane potential `V(x,t)` obeys a
reaction-diffusion PDE — the **monodomain** (single-domain) approximation of
cardiac electrophysiology:

```
    dV/dt  =  D * d2V/dx2  +  f(V, w)          (0 <= x <= L)
    dw/dt  =  g(V, w)
```

- `x` is arc length along the fibre (mm), `t` is time (ms).
- `D` (mm²/ms) is the **diffusion / axial coupling coefficient**. Physically
  `D = a / (2 R_i C_m)` where `a` is fibre radius, `R_i` intracellular
  resistivity, `C_m` membrane capacitance — so **thicker fibres have larger `D`
  and conduct faster**. This is the one parameter we vary across the ensemble.
- `f(V,w)` is the **active membrane current** (the "reaction"): the ionic
  machinery that produces the fast upstroke, plateau, and repolarisation of the
  action potential.
- `w` is a slow **recovery** variable; `g(V,w)` governs its evolution.

**Boundary conditions.** We use **zero-flux (Neumann)** ends,
`dV/dx = 0` at `x = 0, L`, modelling a *sealed cable* — no current leaks out the
tips (implemented by mirroring the ghost node onto its interior neighbour).

### The reaction term (Aliev-Panfilov)

Detailed cardiac ionic models (e.g. **Stewart-Zhang**, the standard Purkinje
model) track ~20 state variables (many ion channels, pumps, and calcium
handling). For a *teaching* cable we use the compact two-variable
**Aliev-Panfilov** excitable model, which reproduces the qualitative AP shape
with just `V` and `w`:

```
    f(V,w) = -k V (V - a)(V - 1) - V w
    g(V,w) = ( eps0 + mu1 w / (V + mu2) ) * ( -w - k V (V - a - 1) )
```

with dimensionless `a = 0.15`, `k = 8`, `eps0 = 0.002`, `mu1 = 0.2`,
`mu2 = 0.3`. `V` is nondimensional in `[0,1]` (0 = rest, ~1 = excited). The cubic
`-kV(V-a)(V-1)` is bistable: below threshold `a` it decays back to rest, above it
it flips to the excited branch — that threshold behaviour is what makes the wave
*propagate* rather than diffuse away.

### Conduction velocity & tree timing

CV is measured, not prescribed: we timestamp when the proximal end (node 0) and
the distal end (node `n-1`) first cross an activation threshold, and

```
    CV = L / (t_out - t_in)              [mm/ms = m/s]
```

The tree is a rooted graph. Each cable `i` has a local traversal delay
`Δ_i = t_out,i − t_in,i` and a fixed junction delay `delay_ms` to enter it. Then
the absolute PMJ activation times satisfy the recurrence

```
    t_in[root]   = delay_ms[root]
    t_in[child]  = t_out[parent] + delay_ms[child]
    t_out[i]     = t_in[i] + Δ_i
```

and the **total ventricular activation time** is `max_i t_out[i]`.

---

## The algorithm

**Per cable (the PDE solve)** — explicit finite differences, forward Euler:

1. Discretise the fibre into `n` nodes, spacing `dx = L/(n-1)`.
2. Two voltage buffers `Vcur`, `Vnew` (ping-pong) + one recovery buffer `w`.
3. For each time step `s = 0 … n_steps-1`:
   - For each node `i`: 3-point Laplacian
     `d2V/dx2 ≈ (V[i-1] - 2V[i] + V[i+1]) / dx²` (mirror at the sealed ends);
     evaluate `f,g`; add pacing stimulus at the proximal nodes during the window;
     `Vnew[i] = V[i] + dt (D·lap + f + stim)`, `w[i] += dt·g`.
   - Record first threshold crossings at node 0 and node `n-1`.
   - Swap `Vcur ↔ Vnew`.
4. CV = `L / ((step_out - step_in) · dt)`.

**Complexity.** One cable is `O(n_steps · n)`. The whole ensemble of `N` cables
is `O(N · n_steps · n)` — but every cable is *independent* during the solve.

**Across cables (the graph pass)** — a single `O(N)` forward sweep resolves the
recurrence above, because the input file guarantees `parent < index`
(topological order). Serial-vs-parallel: the *cables* parallelise trivially; the
graph pass is cheap and left on the host.

---

## GPU mapping

This is the **"same solver for many members"** pattern (docs/PATTERNS.md §1,
exemplified by flagship 9.02), applied to a spatial PDE instead of an ODE.

- **Thread ↔ cable.** Thread `i = blockIdx.x*blockDim.x + threadIdx.x` runs the
  *entire* space×time PDE loop for cable `i` and writes one `CableResult`. No
  inter-thread communication → embarrassingly parallel across cables.
- **Launch config.** `block = 128` threads; `grid = ceil(N/128)` blocks. We use
  128 (not 256) because each thread carries three `PK_MAX_NODES`-double scratch
  arrays; a smaller block eases the per-SM local-memory / register pressure.
- **Memory hierarchy.** The two ping-pong voltage buffers and the recovery buffer
  are declared as **per-thread local arrays** (local memory, backed by global +
  L1). `CableParams` are read from global memory once per thread. There are **no
  atomics and no shared memory** — the parallelism is over fully independent
  jobs, so none are needed.
- **Divergence.** All cables run the same `n_steps`, so control flow is uniform;
  only the threshold-crossing bookkeeping branches, which is mild, warp-local
  divergence.
- **Shared host/device core.** `pk_simulate_cable()` lives in `purkinje.h` marked
  `__host__ __device__` (the `PK_HD` idiom, PATTERNS.md §2), so the CPU reference
  and the GPU kernel execute *identical* arithmetic — verification is exact.

**The alternative production mapping (why we didn't).** A high-performance solver
(MonoAlg3D_C) puts **one thread per NODE** and solves the diffusion term
implicitly with a **batched tridiagonal (Thomas) solve** (cuSPARSE), staging the
tridiagonal coefficients in **shared memory** per segment, and wraps the
recurring per-beat launch sequence in a **CUDA graph**. That is faster and scales
to ~10⁵ segments, but it hides the physics behind library calls; our
one-thread-per-cable version keeps the cable equation legible. See "real world".

---

## Numerical considerations

- **Precision.** We use **double precision** throughout. The reaction term is
  stiff-ish and the CV depends on sub-millisecond front timing, so FP64 avoids
  accumulation error dominating the activation-step counts.
- **Explicit-Euler stability.** The diffusion term imposes the classic
  Courant-type limit for an explicit scheme:

  ```
      dt  <=  dx^2 / (2 D)
  ```

  In the sample, the tightest cable has `dx ≈ 0.234 mm`, `D = 2.5` →
  `dt_max ≈ 0.011 ms`; we use `dt = 0.01 ms`, safely under the bound. Exceed it
  and `V` blows up (NaNs) — a good exercise to try. The reaction term adds its own
  (looser) explicit-stability constraint via `k` and `dt`.
- **Determinism.** The headline outputs are **integer** activation-step indices;
  CV is `L / (integer·dt)`. Integers are order-independent, so the GPU (any thread
  order) and the CPU (serial) produce **bit-identical** step counts. There are no
  floating-point reductions across threads (no `atomicAdd`), so nothing depends on
  scheduling — the stdout is reproducible run to run (PATTERNS.md §3).
- **Threshold aliasing.** The front-crossing time is quantised to `dt`; a finer
  `dt` gives a smoother CV. This is a discretisation artefact, not randomness.

---

## How we verify correctness

Two independent checks:

1. **CPU ≡ GPU (exact).** `simulate_cpu()` and `simulate_kernel()` call the same
   `pk_simulate_cable()`. `main.cu` asserts the per-cable `activate_step_in`,
   `activate_step_out`, and `captured` flags match **exactly** (integers), and the
   conduction velocities agree within `1e-9` mm/ms (a token tolerance for the last
   double round-off bit in `L / tof`). In the demo the worst CV diff is exactly
   `0.0` and there are `0` step mismatches (PATTERNS.md §4: "exact" tolerance class
   because the same integer arithmetic runs on both sides).

2. **Physical plausibility.** The measured CVs (~1.8–2.6 mm/ms = 1.8–2.6 m/s) sit
   in the physiological Purkinje range (2–4 m/s), and the **CV increases with `D`**
   monotonically across the tree — the expected diameter→velocity relationship. A
   deliberately thin branch (`D=1.5`) is visibly the slowest. The total tree
   activation (~30 ms) is a plausible order of magnitude for endocardial spread.

Edge cases handled: a cable whose distal end never crosses threshold is reported
as `BLOCK` (conduction block), and its children inherit the block in the graph
pass rather than propagating a garbage time.

---

## Where this sits in the real world

This is a **reduced-scope teaching version** (CLAUDE.md §13). A research-grade
His-Purkinje simulation differs in several load-bearing ways:

| Aspect | This project (teaching) | Production (openCARP, MonoAlg3D_C, Cardioid) |
|---|---|---|
| Ionic model | 2-variable Aliev-Panfilov | Stewart-Zhang Purkinje (~20 ODEs), full ion channels |
| Geometry | 7 hand-made cables | fractal tree of ~10⁵ segments (L-system / rule-based growth) |
| Coupling | cables solved independently; delays assembled on a graph | true **PMJ** coupling to a 3-D ventricular monodomain/bidomain mesh via gap-junction conductance |
| Diffusion solve | explicit forward Euler | implicit / operator-split; **batched tridiagonal (Thomas)** via cuSPARSE |
| GPU mapping | one thread per cable | one thread per node; shared-memory tridiagonal coefficients; **CUDA graphs** for the per-beat pattern |
| Calibration | none (illustrative D values) | CV calibrated to measured His-Purkinje velocities |

**Prior art to study** (see `README.md` "Prior art & further reading"):
MonoAlg3D_C (GPU monodomain + integrated Purkinje network and PMJ calibration),
openCARP (Purkinje cable coupling), Cardioid/LLNL (Purkinje conduction modelling),
and Chaste (1-D cable infrastructure). The natural next steps — a real ionic
model, true PMJ coupling, and the batched-tridiagonal implicit solve — are listed
as exercises.

> **Not for clinical use.** Every number here comes from synthetic, illustrative
> parameters. This models the *shape* of conduction-system behaviour to teach the
> CUDA pattern, not to make any medical statement.
