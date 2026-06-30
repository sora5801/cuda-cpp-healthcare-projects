# THEORY — 2.32 Protein Folding Pathway Extraction (Transition Path Sampling)

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._
>
> **Scope.** This is a **reduced-scope teaching version** (CLAUDE.md §13). We
> implement TPS on a 1-D model so the *method* is visible end to end. The full
> all-atom version is described in the last section.

---

## The science

### Folding is a rare event

A small protein in water explores an astronomically large configuration space,
yet at equilibrium it is almost always found in one of two **metastable states**:
the **unfolded** ensemble (a fluctuating coil) or the **folded** native
structure. Between them lies a **free-energy barrier**: intermediate
configurations are high in free energy (few microstates, unfavorable), so the
molecule rarely occupies them. The consequence is dramatic timescale separation:
the molecule *dwells* in a basin for microseconds to milliseconds, then crosses
the barrier in just nanoseconds. A crossing — a **transition path** — is the
event we care about (it tells us the folding *mechanism*: which contacts form
first, what the transition-state ensemble looks like), but it is a vanishing
fraction of the total time.

Brute-force molecular dynamics (MD) is therefore catastrophically inefficient for
mechanism: > 99.99 % of the compute is spent jiggling inside a basin, waiting.

### The TPS idea

**Transition Path Sampling** (Bolhuis, Chandler, Dellago, Geissler, ~1998)
changes the question. Instead of sampling *configurations* weighted by their
Boltzmann probability (what MD does), it samples **whole trajectories** weighted
by the probability that the dynamics would produce them *and* by the constraint
that they are **reactive** (they start in basin A and end in basin B). You never
simulate the long waiting time — you only ever look at the short, interesting
crossing segments. The engine that generates new reactive paths from old ones is
the **shooting move**: perturb a path at one time slice, re-integrate the
dynamics forward and backward, and accept the new path if it still connects the
basins.

### The committor — the *true* reaction coordinate

How do we know which configurations are "the transition state"? The rigorous
answer is the **committor** `p_B(x)`: the probability that a trajectory launched
from configuration `x` (with random thermal velocities) reaches basin **B**
before basin **A**. By definition `p_B = 0` deep in A and `p_B = 1` deep in B.
The **transition-state ensemble** is the isosurface where a configuration is
equally likely to fall either way:

```
        p_B(x) = 1/2          <-- the transition state
```

Committor analysis is the gold standard precisely because it is defined purely by
the dynamics, not by a guessed coordinate. This project computes `p_B` directly
and shows it crossing 1/2 right at the barrier top.

---

## The math

### The reduced 1-D model

We collapse the 3N-dimensional protein onto **one reaction coordinate** `x` (a
"folding order parameter" — think *fraction of native contacts*, rescaled to run
over roughly `[0, 1]`). The folding free-energy profile along `x` is modeled as a
symmetric **quartic double well**:

```
V(x) = barrier * ( ((x - x0)/w)^2 - 1 )^2
```

- minima (the basins) at `x = x0 - w` (basin **A**, unfolded) and `x = x0 + w`
  (basin **B**, folded), where `V = 0`;
- a maximum (the **barrier**) at `x = x0`, where `V = barrier` (in units of `kT`).

The force is `F(x) = -dV/dx = -(4*barrier/w) * q * (q^2 - 1)`, with
`q = (x - x0)/w`. (Both `V` and `F` are coded once in `tps_physics.h` as shared
`__host__ __device__` functions.)

### Overdamped Langevin (Brownian) dynamics

On a free-energy surface the natural dynamics is **overdamped Langevin** (the
high-friction limit appropriate for a slow collective coordinate in a viscous
solvent). The Euler–Maruyama discretization is:

```
x_{t+1} = x_t + (D*dt / kT) * F(x_t) + sqrt(2*D*dt) * η,   η ~ N(0,1)
```

with diffusion constant `D`, timestep `dt`, and `kT = 1` in our reduced units
(energies already measured in `kT`). The first term is deterministic drift
downhill; the second is the **thermal kick**. The kick is the *only* mechanism
that can climb the barrier — set it to zero and the bead just slides into the
nearest basin and stays there. The mean first-passage time over the barrier
scales like Kramers' law, `~ exp(barrier/kT)`, which is exactly why a 5 kT
barrier already makes spontaneous crossings rare (`e^5 ≈ 148×` slower than the
in-basin relaxation).

### The committor as a Monte-Carlo estimate

We cannot solve the committor PDE in closed form for a general landscape, so we
**estimate** it: from a configuration `x`, launch `N` independent trajectories
with fresh random kicks and count how many reach B first:

```
p_B(x) ≈ (# shots from x that hit B before A) / (total shots from x)
```

In this project each shooting point contributes one such shot, and we *bin*
shooting points along `x`, so `committed_per_bin[b] / shots_per_bin[b]` is a
Monte-Carlo committor estimate for bin `b`.

---

## The algorithm

### One shooting move (`run_shot` in `tps_physics.h`)

```
input : shooter index i, parameters P
1.  seed an independent RNG stream from (P.seed, i)          # reproducible
2.  choose a shooting point x_sp near the barrier            # spread across [A,B]
3.  fwd = run_leg(x_sp):  integrate BD forward until x enters basin A or B
4.  bwd = run_leg(x_sp):  integrate an independent BD leg (the backward shot)
5.  is_transition = (one of {fwd,bwd} is A and the other is B)   # connects basins
6.  committed_B   = (fwd reached B)                              # committor sample
7.  return (fwd, bwd, is_transition, committed_B, bin(x_sp))
```

`run_leg` is the inner BD loop: it steps `x_{t+1} = bd_step(x_t)` until
`basin_of(x)` returns A or B (an absorbing boundary), or a `max_steps` budget is
hit (the rare-event safety net). Reflective walls just beyond the basins keep a
huge kick from sending `x` to infinity.

### Aimless shooting vs. what we do

Real **aimless shooting** draws the shooting point from an *existing accepted
path* and runs a Metropolis chain in path space (each new path is shot from the
last accepted one). We instead **spread shooting points deterministically across
the transition region** (shooter `i` at fraction `(i+0.5)/n` from A to B, plus a
small reproducible jitter). This trades the path-space Markov chain for a direct
sweep — which is *better* for a single teaching run, because one run populates the
*entire* committor curve instead of wandering. The acceptance rule and committor
definition are identical to the real method.

### Complexity

- **Serial (CPU):** `O(n_shooters × max_steps)` — each of `n_shooters`
  independent shots integrates up to `max_steps` steps per leg (two legs).
- **Parallel (GPU):** the same total work, but spread across `P` threads, so the
  wall time is `O(n_shooters × max_steps / P)` plus launch overhead — because the
  shots are independent there is **no communication** and **no synchronization**
  between them (only the final tally uses atomics). This is the best case for a
  GPU: linear speedup until you run out of shooters to fill the machine.

---

## The GPU mapping

### Threads, blocks, grid

Each **shooter → one thread**. We launch a fixed grid (1024 blocks ×
256 threads) and use a **grid-stride loop**: thread `t = blockIdx.x*blockDim.x +
threadIdx.x` handles shooters `t, t+stride, t+2*stride, …` with
`stride = blockDim.x*gridDim.x`. This covers any `n_shooters` with one fixed
launch and gives the scheduler far more resident warps than SMs, so the long
latency of the BD inner loop (each step calls `sqrt`, `log`, `cos` for the
Gaussian) is hidden behind other warps' work.

```
grid:   [ block 0 ][ block 1 ] ... [ block 1023 ]      (1024 blocks)
block:  256 threads = 8 warps of 32 lanes
thread: shooter t, then t+256*1024, then t+2*256*1024, ...   (grid-stride)
```

### Memory hierarchy

- **Registers:** the entire per-shooter state (`x`, the RNG word, the
  `ShotResult`) lives in registers — there are no per-thread arrays, which keeps
  register pressure low and occupancy high.
- **Constant-ish parameters:** `SimParams` is passed **by value** to the kernel,
  so it lands in the constant/parameter bank and is broadcast to every thread.
- **Global memory:** only the four tally targets (two scalars + two `n_bins`
  histograms) live in global memory, written via `atomicAdd`.
- **No shared memory** is needed: shooters share nothing during the simulation;
  the only cross-thread interaction is the final tally.

### Why integer atomics (the determinism lesson)

Many threads add into the same counters and histogram bins simultaneously, so we
serialize those updates with `atomicAdd`. The subtle, important point
(PATTERNS.md §3): **floating-point addition is not associative**, so if we tallied
*floats*, the nondeterministic order in which atomics land would change the
low-order bits and the GPU result would not be bit-reproducible — nor would it
match the CPU. Our tallies are **integers** (a shot either is or isn't a
transition; a leg either did or didn't reach B), and integer addition *does*
commute, so the final counts are identical regardless of order. That is what lets
us verify **exactly** (`== 0` mismatches), and it is why the demo's stdout is
byte-stable.

---

## Numerical considerations

- **Precision.** The dynamics run in `double`. Brownian integration accumulates
  thermal noise, so single precision would be tolerable for the *physics*, but we
  keep `double` so the shared CPU/GPU core produces bit-identical reals and the
  integer tallies match exactly. The committor curve and acceptance count are
  invariant to precision within Monte-Carlo error.
- **Identical RNG on host and device.** We use a **counter-based splitmix64**
  stream seeded from `(seed, shooter)`, not a stateful generator. Counter-based
  RNG is the standard choice for massively parallel Monte Carlo: thread `i` can
  reconstruct its *own* stream with no shared state and no warm-up race. The exact
  same code path runs on the CPU, so shooter `i` draws the exact same numbers on
  both — the precondition for exact verification.
- **Box–Muller, single branch.** Each Gaussian uses only the cosine branch of
  Box–Muller (discarding the sine partner) so that the **RNG-draw count per step
  is fixed**; caching the second value would add per-thread state that complicates
  host/device parity. The cost is ~2× the RNG draws; the benefit is exact
  reproducibility.
- **Absorbing/reflecting boundaries.** A basin is "reached" within `basin_tol` of
  its minimum (an absorbing boundary that ends a leg). Reflective walls at
  `x0 ± 2w` prevent a rare huge kick from escaping to infinity; without them a
  leg could run to `max_steps` for the wrong reason.
- **Determinism of atomics.** Integer `atomicAdd` only (see GPU-mapping above) —
  no floating-point reduction anywhere on the device.

---

## How we verify correctness

Two independent checks:

1. **Exact CPU↔GPU agreement (the regression guarantee).** `main.cu` runs
   `tps_cpu` and `tps_gpu` on the same parameters and compares **every** integer
   counter: `n_transitions`, `n_fwd_to_B`, and all `2 × n_bins` histogram
   entries. Because both call the identical `run_shot()` with the identical RNG
   and the tallies are integer, the correct result is **zero mismatches**. The
   tolerance is therefore **exact** (`== 0`), not a floating-point epsilon — any
   difference would signal a real bug (RNG divergence, an atomics error, a
   host/device code-path split), which is exactly what we want a test to catch.

2. **Known-answer physics check (the science guarantee).** The committor `p_B`
   must rise monotonically from 0 (unfolded basin) to 1 (folded basin) and cross
   **1/2 at the barrier top** `x = x0`. In the sample run the transition-state bin
   is **10 of 20**, i.e. `x ≈ 0.5 = x0` — the analytic barrier position. So the
   simulation recovers the correct transition state, validating the *method*, not
   just CPU==GPU consistency (PATTERNS.md §4: pair an exact agreement test with an
   analytic check).

---

## Where this sits in the real world

A research-grade TPS / pathway study differs from this teaching model in every
heavy dimension — and understanding the gap is the point:

- **Full all-atom MD, not 1-D BD.** Each "shot" is a real molecular-dynamics
  simulation of the protein in explicit solvent (tens of thousands of atoms),
  integrating Newton's equations with a force field (AMBER, CHARMM) on the GPU via
  **OpenMM**. The 1-D reaction coordinate here stands in for that 3N-dimensional
  trajectory; the *parallel structure* (many independent shots) is what carries
  over to the GPU verbatim.
- **Rigorous shooting with momentum reversal.** Real shooting perturbs the
  **momenta** at a shooting point and integrates the backward leg with
  *time-reversed* velocities, with a Metropolis acceptance derived from the path
  action. Our backward leg is an independent forward Brownian leg — valid for
  overdamped 1-D dynamics, but a deliberate simplification.
- **A Markov chain in path space.** TPS proper runs a chain where each new path is
  shot from the last *accepted* path (aimless shooting, one-way shooting, spring
  shooting…). We sweep shooting points directly so one run covers the whole
  committor.
- **Smarter rare-event machinery.** Production toolchains layer on **path
  collective variables (PathCV)**, **weighted ensemble** (WESTPA: split/merge
  walkers across progress bins, coordinated with NCCL across GPUs), **adaptive
  sampling + Markov State Models** (HTMD), and **τRAMD** for unbinding kinetics.
- **AIMMD — the AI-augmented frontier.** A neural network is trained on the
  shots to predict the committor `p_B` directly; TPS then shoots preferentially
  from the learned `p_B = 1/2` isosurface, converging the path ensemble far
  faster. That GPU neural-committor inference is the modern research direction the
  catalog points at — and the committor histogram we build here is the quantity it
  learns to approximate.

The honest summary: this project teaches the **anatomy** of Transition Path
Sampling — shooting, acceptance, the committor, and the `p_B = 1/2` transition
state — and the **embarrassingly-parallel GPU pattern** that makes TPS scale. It
is not, and does not pretend to be, a quantitative folding-kinetics engine.
