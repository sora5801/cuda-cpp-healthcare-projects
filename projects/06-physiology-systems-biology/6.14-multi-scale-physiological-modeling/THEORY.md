# THEORY — 6.14 Multi-Scale Physiological Modeling

> A teaching-scale realization of the Virtual Physiological Human (VPH) idea:
> couple a **cell-scale** ODE (ion-channel-like kinetics) to a **tissue-scale**
> PDE (electrical propagation) and solve them together on the GPU. We do it on a
> 1-D cardiac "cable" so the whole thing runs on a laptop and every line is
> legible.
>
> _Educational only — not for clinical use._

---

## The science

A heartbeat is an **electrical wave**. Each patch of cardiac muscle is an
*excitable* medium: at rest it sits quietly, but a supra-threshold electrical
kick triggers a stereotyped **action potential (AP)** — a fast depolarization
(voltage jumps up) followed by a slower recovery (repolarization) — after which
the patch is briefly refractory and then resets. Crucially, neighbouring patches
are electrically **coupled** (through gap junctions), so an AP in one patch pushes
its neighbours over threshold: the excitation *propagates* as a traveling wave.
That wave, sweeping across the heart, is what coordinates the mechanical
contraction of every beat.

This is inherently **multi-scale** (the catalog deep dive):

| Scale | What lives here | Time scale |
|---|---|---|
| molecular / cell | ion channel & gating kinetics → membrane voltage (an ODE) | µs–ms |
| tissue | electrotonic coupling → wave propagation (a PDE) | ms–s |
| organ / system | chamber mechanics, circulation | heartbeat / minutes |

Production VPH stacks (OpenCMISS, Chaste, simcardems) solve detailed cell models
(ten-Tusscher, O'Hara-Rudy) at **every quadrature point of a 3-D finite-element
mesh** — *millions* of coupled ODEs per time step. We keep the **structure** of
that problem (a fine sub-grid cell ODE at every coarse mesh node, coupled by
diffusion) but shrink the cell model to the 2-variable **FitzHugh-Nagumo (FHN)**
caricature and the mesh to a 1-D strand.

---

## The math

**Cell scale (FitzHugh-Nagumo).** State `(v, w)` per node, where `v` is a
voltage-like excitation variable and `w` a slow recovery variable:

```
dv/dt = v(v - a)(1 - v) - w          (fast, cubic → excitable)
dw/dt = eps (v - b w)                (slow, eps << 1 → time-scale separation)
```

The cubic `f(v,w) = v(v-a)(1-v) - w` gives the all-or-none spike; `eps << 1`
makes `w` evolve much more slowly than `v` (the µs-vs-ms split, in miniature).

**Tissue scale (monodomain cable equation).** Coupling adds a diffusion term to
the voltage equation only (current spreads; the gating variable does not):

```
dv/dt = D ∂²v/∂x² + f(v, w)          (reaction–diffusion)
dw/dt =              g(v, w)
```

`D` is the tissue diffusion coefficient (conductivity / capacitance). With a
zero-flux (Neumann) boundary `∂v/∂x = 0` at both sealed ends, a localized
stimulus launches a wave whose speed — the **conduction velocity** — scales like
`√D` times a reaction-rate factor. Discretizing space on a uniform grid of `n`
nodes with spacing `dx`, the Laplacian becomes the 3-point stencil
`∂²v/∂x²|ᵢ ≈ (vᵢ₋₁ − 2vᵢ + vᵢ₊₁)/dx²`.

---

## The algorithm

We advance the coupled system by **operator splitting** (the workhorse of the
heterogeneous multiscale method, HMM). Within one global step `dt`:

1. **Reaction sub-step (fine scale).** At every node, advance the *local* cell
   ODE `(v,w)` using the reaction terms only, with one classical **RK4** step
   (4 stage evaluations, O(dt⁴) accuracy). Nodes are independent here.
2. **Diffusion sub-step (coarse scale).** Apply the tissue coupling with one
   explicit forward-Euler step of the Laplacian: `vᵢ ← vᵢ + dt·D·(vᵢ₋₁ − 2vᵢ +
   vᵢ₊₁)/dx²`, reading a **snapshot** of the whole `v` field (a Jacobi update)
   so the result does not depend on node order.
3. **Record** the first time each node's `v` crosses the activation threshold
   (0.5) → the **activation map**; its spatial slope is the conduction velocity.

**Complexity.** Per global step the work is `O(n)` (each node does a constant
amount of reaction + stencil work); over `S` steps the total is `O(n·S)`. Serial
time is `O(n·S)`; the parallel *span* (critical path) is `O(S)` because all `n`
nodes within a step are independent — that is exactly the parallelism the GPU
exploits.

```
 global step s:
   [ reaction  ]  node 0  node 1  ...  node n-1   (all independent → parallel)
   [ diffusion ]  read old v-field, write new v-field via 3-pt stencil (Jacobi)
   [ record    ]  first-crossing activation times
   repeat for S steps →  a wave crawls from the left end to the right end
```

---

## The GPU mapping

The catalog prescribes **two-level parallelism: a CUDA grid over mesh elements,
threads over the per-element ODE RHS.** On the 1-D cable that collapses to the
clean idiom **one GPU thread per node** (`kernels.cu`):

- **`react_kernel`** — thread `i = blockIdx.x*blockDim.x + threadIdx.x` owns node
  `i`; it pulls `(v[i], w[i])` into registers, runs the shared RK4 step, writes
  back. No neighbour reads, no shared memory, no atomics — embarrassingly
  parallel. This is the GPU form of "batch-solve the sub-grid cell ODEs"
  (SUNDIALS batch-CVODE's job in production; hand-rolled RK4 here so nothing is a
  black box — CLAUDE.md §6.1.6).
- **`diffuse_kernel`** — the stencil. Because each node reads its neighbours, an
  in-place update would let one thread read a value another already overwrote. We
  use **ping-pong buffers** (`d_v` → `d_v2`, then swap): read the *old* field,
  write the *new* one. This makes it a Jacobi sweep — matching the CPU's
  snapshot, so results agree. (Same pattern as flagships 6.04 / 14.02.)
- **`activation_kernel`** — each thread writes only `act[i]`, so recording
  first-crossing times is race-free.

**Memory hierarchy.** State fields (`d_v`, `d_v2`, `d_w`, `d_act`) live in global
memory; per-node RK4 scratch lives in registers. The stencil's neighbour loads
are the main global-memory traffic; on the small cable here they are served from
the L2/L1 caches. **Launch config:** 128 threads/block (a multiple of the 32-lane
warp; four warps give the scheduler latency-hiding slack). **Occupancy vs. size:**
this is a *launch-bound* problem on a short cable — three tiny kernel launches per
step, thousands of steps — so the GPU can be *slower* than the CPU here (see
timing note). The GPU's advantage grows with node count (organ-scale 3-D meshes
have 10⁶–10⁷ nodes, where per-step parallelism dominates launch overhead).

---

## Numerical considerations

- **Precision (FP64).** All state and arithmetic are `double`. The FHN cubic and
  the stencil are well-conditioned, but the *long* integration (5000 steps ×
  three sub-steps) means round-off can accumulate, so double precision keeps the
  activation times crisp.
- **Explicit-diffusion stability.** Forward Euler on the Laplacian is stable only
  when the diffusion number `r = D·dt/dx² ≤ 0.5`. The committed sample has
  `r = 2.0·0.02/0.25 = 0.16`, comfortably stable. `make_synthetic.py` prints `r`
  and warns if you push it past 0.5 (a great exercise: watch it blow up).
- **Operator-splitting error.** Godunov (first-order) splitting introduces an
  `O(dt)` splitting error on top of RK4's `O(dt⁴)` reaction accuracy; for this
  didactic wave it is negligible. Strang (second-order) splitting is the standard
  production upgrade (an exercise in the README).
- **Determinism.** Every node's update is order-independent (Jacobi diffusion, no
  atomics), so stdout is **byte-identical every run** — the demo diffs it. Timing
  and the numeric field-error go to stderr (they vary), per PATTERNS.md §3.

---

## How we verify correctness

Two independent checks:

1. **CPU == GPU (the software check).** The per-node physics (FHN reaction, RK4,
   the diffusion stencil, the mirrored boundary) lives in **one shared
   `__host__ __device__` header** (`multiscale.h`). The serial reference and the
   GPU kernels call the *same* functions, so they run identical arithmetic. We
   compare the final `v` field, the final `w` field, **and** the whole activation
   map; `main.cu` reports the worst absolute difference. Tolerance is `1e-6`
   (documented as a small *physical* tolerance for a long iterative solver where
   the GPU's fused multiply-add and the host compiler can diverge by ~1e-13 per
   op — PATTERNS.md §4). In practice they agree to **~1e-16** here, because the
   split arithmetic is genuinely the same on both sides.
2. **The physics check (the science).** The synthetic problem is engineered to
   recover a **known** answer: a stimulus at the left end must produce a *single
   traveling wave*. Success looks like an activation map whose times **increase
   monotonically** with position (0.0 → 17.6 → … → 90.2), *all* nodes activating,
   and a well-defined, roughly constant **conduction velocity** (~0.68
   space/time). A stalled or reflected wave would show up immediately as a
   non-monotone map or a low `nodes activated` count.

---

## Where this sits in the real world

- **Cell models.** Real cardiac EP uses biophysically detailed ionic models
  (ten-Tusscher–Panfilov, O'Hara–Rudy) with 10–40 state variables and stiff
  dynamics, solved with adaptive implicit integrators — this is precisely what
  **SUNDIALS batch-CVODE on the GPU** provides (one CVODE instance per quadrature
  point). FHN is the textbook caricature that keeps the *structure* while fitting
  on a slide.
- **Tissue models.** The real coupling is the **bidomain** (or monodomain)
  equation on an anisotropic 3-D **finite-element** mesh with fiber orientation,
  assembled and solved with sparse linear algebra (**cuSPARSE**/AMG) — not a
  simple 1-D explicit stencil. Our 1-D explicit cable is the didactic reduction.
- **Scale coupling.** Production stacks (**OpenCMISS**, **Chaste**,
  **simcardems**) formalize inter-scale coupling (electromechanics, circulation
  via Windkessel/1-D vessel networks) and standardize it with **FMI**
  co-simulation. Operator splitting, as used here, is the same fundamental idea.
- **What we simplified (honesty).** 1-D not 3-D; FHN not a real ionic model;
  explicit forward-Euler diffusion (stability-limited) not an implicit/FEM solve;
  Godunov not Strang splitting; no mechanics, no circulation, no fibers. The
  numbers are illustrative and have **no clinical meaning** (CLAUDE.md §8).
