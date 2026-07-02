# THEORY — 6.19 Defibrillation & High-Voltage Shock Simulation

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._
>
> **Scope note.** This is a deliberately **reduced-scope** teaching model: a 1-D
> monodomain FitzHugh-Nagumo cable, not the full 3-D bidomain system. §7 explains
> exactly what the real problem adds and where the catalog's cuSPARSE CG solve fits.

---

## 1. The science

Every heartbeat is an electrical wave. Cardiac cells are **excitable**: at rest
their transmembrane voltage `V` sits low, but a stimulus that pushes `V` past a
threshold triggers a fast, self-amplifying upstroke (an *action potential*),
followed by a slow recovery back to rest during which the cell is *refractory*
(cannot be re-excited). Neighbouring cells are electrically coupled through
gap junctions, so an action potential in one cell ignites its neighbours: the
excitation **propagates** as a travelling wave. In a healthy heart, one organised
wave per beat sweeps the ventricles and pumps blood.

In **ventricular fibrillation (VF)** that organised wave breaks up into multiple
self-sustaining **re-entrant** circuits — electrical chaos. The heart quivers
instead of pumping; VF is fatal within minutes. A **defibrillator** interrupts it
by delivering a brief, strong electric field across the whole heart. If the shock
is strong enough it drives essentially all tissue past threshold at once, leaving
the entire heart depolarised and then uniformly refractory. With no excitable gap
for a re-entrant wave to advance into, the chaos dies, and the heart's natural
pacemaker can restart a normal rhythm.

The subtlety that makes shock modelling interesting is **virtual electrode
polarization (VEP)**. An extracellular field does not depolarise tissue uniformly:
near one electrode the tissue is **depolarised** (pushed up), and near the other
it is **hyperpolarised** (pushed *down*, below rest). These adjacent regions of
opposite polarity — "virtual electrodes" — are what actually reset the wave
pattern. The engineering question is the **defibrillation threshold (DFT)**: the
smallest shock that reliably terminates the arrhythmia. Too weak fails; too strong
damages tissue. Every defibrillator and ICD is designed around the DFT.

This project answers a small version of that question by simulation: *given a
tissue with ongoing activity, how strong must the shock be to stop it?*

## 2. The math

We model a 1-D cable of excitable tissue with the **monodomain reaction-diffusion**
equation for the transmembrane voltage `V(x,t)`, coupled to a slow **recovery
variable** `w(x,t)` (the FitzHugh-Nagumo, "FHN", ionic model):

```
∂V/∂t = D ∂²V/∂x²  +  f(V, w)  +  I_stim(x, t)          (voltage)
∂w/∂t = ε (V − γ w)                                     (recovery)
f(V, w) = V (V − a)(1 − V) − w                          (FHN reaction / ionic current)
```

Symbols (all **dimensionless** — the FHN model is a qualitative caricature; §7
maps them to physical bidomain quantities):

| Symbol | Meaning | Sample value |
|---|---|---|
| `V` | transmembrane voltage, ~0 at rest, ~1 fully excited | state |
| `w` | recovery variable (lumped slow repolarising gates) | state |
| `D` | diffusion / gap-junction coupling coefficient | 0.6 |
| `a` | excitation threshold, `0 < a < 1` (V must exceed `a` to fire) | 0.13 |
| `ε` | recovery rate; small ⇒ slow recovery, long refractory period | 0.008 |
| `γ` | recovery coupling (sets the resting equilibrium) | 1.0 |
| `I_stim` | externally applied stimulus current (here: the shock) | — |
| `x, t` | space, time (grid `dx`, step `dt`) | dx=1, dt=0.1 |

**Boundary conditions.** Zero-flux (Neumann) at both ends: `∂V/∂x = 0`. No current
leaves the cable, so a wave reflects rather than leaking out.

**The shock term `I_stim`** models VEP. During the shock window
`[shock_start, shock_start+shock_len)` we apply a signed current: `+amp` to the
left half of the cable (depolarising virtual electrode) and `−amp` to the right
half (hyperpolarising virtual electrode). A **biphasic** protocol additionally
flips the sign at the temporal midpoint of the window. Outside the window
`I_stim = 0`.

**Objective.** For each shock amplitude `amp` in an ascending ladder, integrate
the system forward `nsteps` steps from an initial condition that contains an
ongoing wave, then measure the **residual activity**
`A = (1/N) Σ_i max(V_i − 0.1, 0)·[V_i>0.1]` (mean depolarised voltage). A shock
**succeeds** if `A < success_thresh`. The **DFT** is the smallest `amp` that
succeeds.

## 3. The algorithm

**Operator-split forward Euler** advances the whole cable one step at a time. Each
step, for every cell `i`, does three things in order (all in `cable_step()` in
`src/defib.h`):

1. **Diffusion (stencil).** Approximate `∂²V/∂x²` with the 3-point Laplacian
   `(V[i−1] − 2V[i] + V[i+1]) / dx²`, using ghost cells `V[i]` at the ends for the
   zero-flux boundary.
2. **Reaction + shock.** Add the FHN ionic current `f(V,w)` and `I_stim`, then
   `V_new = V + dt·(D·lap + f + I_stim)`.
3. **Recovery.** `w_new = w + dt·ε(V − γ w)`.

We use **two buffers** (`V_old→V_new`, `w_old→w_new`) and swap after each step, so
a cell's update reads only the *old* field — no in-place hazards.

**Initial condition.** The left `initial_excited` cells start at `V=1` (a
depolarised patch); the rest at rest. This launches a travelling wave — our
stand-in for the ongoing activity a shock must terminate.

**The sweep.** Repeat the whole simulation for each amplitude, then scan the
residuals from weakest to strongest for the first success (the DFT).

**Complexity.** One simulation is `O(nsteps · N)` work (N = cells). A sweep of `M`
amplitudes is `O(M · nsteps · N)`. Serial depth is `O(nsteps)` per simulation
(steps are sequential); the parallelism is across the `M` amplitudes and, within
a step, across the `N` cells. Arithmetic intensity is modest — a handful of
flops per cell per step against a few neighbour reads — so the kernel is
memory-/latency-bound, not compute-bound (§5, §7).

## 4. The GPU mapping

The sweep is an **ensemble of independent trajectories** (PATTERNS.md §1, like
9.02 SEIR and 13.02 PBPK). We assign **one thread per shock amplitude**: thread
`k` runs the entire cable simulation for `amps[k]` and writes one residual number.

- **Thread-to-data mapping:** `k = blockIdx.x * blockDim.x + threadIdx.x` selects
  the shock amplitude; the thread loops all `nsteps` internally.
- **Launch configuration:** `block = 128` threads, `grid = ceil(M / 128)`. Each
  thread does a lot of work (a full time-stepped cable) and streams its own
  scratch, so we are latency-bound; 128 gives 4 warps per block to hide that
  latency without over-subscribing registers. `M` is small here (10), so a single
  block runs — the pattern scales to thousands of amplitudes unchanged.
- **Memory hierarchy:** each thread needs four length-`N` buffers (`Va,Vb,wa,wb`,
  double-buffered). Those are too big and too runtime-sized for registers/stack,
  so the host allocates **one global slab** of `4·M·N` doubles and each thread
  takes four disjoint slices by index arithmetic. Disjoint slices ⇒ **no data
  races, no atomics**. `FhnParams` is passed **by value**, so it rides in the
  kernel's parameter/constant space and is broadcast to every thread.
- **No CUDA library** is used in this reduced version: the physics is a hand-rolled
  stencil + ODE, which is the whole point (no black box). §7 explains where the
  catalog's **cuSPARSE conjugate gradient** enters the *full* bidomain problem.

```
 amplitudes (M threads)            each thread's private cable (N cells, ping-pong)
 ┌───────────────────────┐        ┌──────────────────────────────────────────┐
 │ t0  amp=0.00 ──────────┼──────▶ │ Va[N] ─step→ Vb[N] ─step→ Va[N] ... → A0  │
 │ t1  amp=0.05 ──────────┼──────▶ │ Va[N] ─step→ Vb[N] ─step→ ...      → A1  │
 │ ...                    │        │                                          │
 │ tk  amp=amps[k] ───────┼──────▶ │ 3-point Laplacian + FHN + shock per step │
 └───────────────────────┘        └──────────────────────────────────────────┘
   grid.x = ceil(M/128)              d_scratch = 4·M·N doubles (Va,Vb,wa,wb slices)
```

Alternative mapping (an exercise): **one thread per cell** on a 2-D grid
(amplitude × cell) with a block barrier per step. That wins for a *single huge*
cable; thread-per-trajectory wins for *many small* cables and needs no cross-block
sync across thousands of steps.

## 5. Numerical considerations

- **Precision: FP64.** The FHN cubic and the long forward-Euler integration are
  sensitive to rounding; double precision keeps the CPU and GPU trajectories
  essentially identical (see §6) and keeps the diffusion stable.
- **Stability (CFL-like limit).** Explicit forward-Euler diffusion is stable only
  if `dt ≤ dx²/(2D)`. With `dx=1, D=0.6` that limit is `0.833`; the sample `dt=0.1`
  is well inside it. `load_sweep()` and `make_synthetic.py` both **enforce** this,
  so you cannot accidentally run an unstable, garbage simulation.
- **No atomics, fully deterministic.** Threads write disjoint memory and each
  residual is a single-thread sequential reduction, so there is no floating-point
  reordering. stdout is byte-identical every run (PATTERNS.md §3); timings go to
  stderr.
- **The biphasic artifact (honest caveat).** In this 2-variable model a *biphasic*
  shock has a **higher** DFT than monophasic — the opposite of the clinic. The
  clinical biphasic advantage comes from the polarity reversal recharging fast
  sodium-channel availability; FHN has no fast Na⁺ gate to recharge, so the second
  phase merely partially undoes the first. We surface this rather than hide it
  (CLAUDE.md §1). Capturing it correctly needs a multi-gate ionic model (§7).

## 6. How we verify correctness

The GPU kernel and the CPU reference (`src/reference_cpu.cpp`) call the **same**
`__host__ __device__` functions in `src/defib.h` (the shared-core idiom,
PATTERNS.md §2). So for each amplitude they perform the identical sequence of
double-precision operations; the only possible divergence is the compiler's
freedom to fuse a multiply-add (FMA) differently on host vs. device.

`main.cu` runs both sweeps and takes the max `|residual_cpu − residual_gpu|` over
all amplitudes. We verify to **`1e-6`** — tight enough to catch any real bug, and
honest about FMA (PATTERNS.md §4). In practice the observed difference is `~1e-17`
(machine epsilon), i.e. effectively bit-identical, which is strong evidence the
GPU implementation is correct: an independent serial baseline and the parallel
kernel agree to the last representable digit.

A second, *scientific* check beyond CPU==GPU: the DFT curve is **monotone and
all-or-nothing** — weak shocks leave `A ≈ 0.052`, and above threshold `A = 0`
exactly. That step-like transition is the qualitative signature of a real DFT,
so the model reproduces the *phenomenon*, not just internal self-consistency.

## 7. Where this sits in the real world

Production defibrillation studies (openCARP, Cardioid, MonoAlg3D, Chaste) solve
the full **bidomain** equations on 3-D patient-specific meshes:

```
∇·(σ_i ∇φ_i) = χ (C_m ∂V/∂t + I_ion)                 (intracellular)
∇·(σ_i ∇φ_i) + ∇·(σ_e ∇φ_e) = −I_e                    (extracellular, elliptic)
V = φ_i − φ_e
```

Key differences from this teaching version:

- **Two potentials, one elliptic solve.** The bidomain has separate intra- and
  extracellular potentials. The extracellular equation is an **elliptic** PDE
  solved every time step — this is exactly where the catalog's **cuSPARSE
  conjugate gradient** goes (a large sparse linear solve `A φ_e = b`). Our
  monodomain reduction collapses this to a single diffusion term and skips the
  solve entirely; that is the biggest simplification here.
- **Realistic ionic models.** Ten-Tusscher, O'Hara-Rudy, etc. have 10–40 gating
  variables integrated with the **Rush-Larsen** method (which treats the gate
  ODEs semi-analytically for stability at large `dt`). FHN's two variables are a
  caricature — hence the biphasic artifact (§5).
- **3-D geometry + fine resolution.** ~0.1 mm meshes over a whole heart mean tens
  of millions of cells; the ionic ODE per cell is the per-cell kernel, and the
  **dual-grid** approach (fine heart on GPU, coarse torso on CPU, coupled at the
  interface) plus **Unified Memory** are what the catalog names.
- **Real VEP.** True virtual electrodes emerge from the bidomain's unequal
  anisotropy ratios (`σ_i` vs `σ_e`) via the "activating function"; we impose the
  dipole structure by hand as a signed current.
- **DFT protocols.** Clinically the DFT is estimated statistically (dose-response,
  up-down methods) over shock timing, waveform, and electrode placement — a huge
  parameter space, which is precisely why GPUs matter.

The teaching value that survives the reduction: excitable propagation, the
stencil+ODE operator split, virtual-electrode polarity, the all-or-nothing DFT,
and the ensemble-parallel GPU pattern that a real DFT sweep also uses.

---

## References

- **FitzHugh (1961) / Nagumo et al. (1962)** — the FHN excitable model used here;
  read for the phase-plane intuition behind the cubic reaction term.
- **Keener & Sneyd, *Mathematical Physiology*** — monodomain/bidomain derivations
  and the cable equation; the standard textbook treatment.
- **Trayanova (2011), "Whole-heart modeling," *Circ. Res.*** — how defibrillation
  and VEP are modelled at whole-heart scale; the real version of this project.
- **openCARP** (https://git.opencarp.org/openCARP/openCARP) — study the bidomain
  solver and its extracellular-stimulus (defibrillation) setup.
- **MonoAlg3D_C** (https://github.com/rsachetto/MonoAlg3D_C) — a GPU
  monodomain/bidomain solver; see how the per-cell ODE and linear solve are laid
  out on the device.
- **Cardioid/LLNL** (https://github.com/llnl/cardioid) — HPC cardiac EP + shock.
- **Chaste** (https://github.com/Chaste/Chaste) — bidomain with proper electrode
  boundary conditions.
- **Rush & Larsen (1978)** — the stable ionic-gate integration method the catalog
  cites; the reason real ionic models can take reasonable time steps.
