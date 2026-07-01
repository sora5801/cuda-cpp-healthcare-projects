# THEORY — 6.11 Stochastic (Gillespie) Biochemical Simulation

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

Inside a single cell, a gene may be transcribed into just a **handful of mRNA
molecules**, and a signaling pathway may hinge on **tens of protein copies**.
At those counts the usual picture of chemistry — smooth concentrations changing
continuously — breaks down. Reactions are *discrete events*: at some random
instant, one specific molecule degrades, or one new transcript appears. The copy
number is a small integer that jumps up and down by ±1 (or ±2 for dimerization),
and repeating the same experiment gives a *different* trajectory each time. This
randomness is not measurement error — it is real **intrinsic biochemical noise**,
and it has biological consequences: cell-to-cell variability, stochastic
gene-expression bursts, probabilistic cell-fate decisions.

The deterministic mass-action ODE (concentrations evolving by
`d[X]/dt = production − degradation`) is the *mean-field* limit — accurate only
when molecule counts are large. When they are small, we need a model of the
**probability distribution over integer states**, and a way to sample it. That is
the Chemical Master Equation and Gillespie's algorithm.

**Our worked model — constitutive gene expression (birth–death).**
A single species, mRNA count `M`, with two reactions:

```
  R1:  ∅  ── k_prod ──►  M      (transcription: a constant "source")
  R2:  M  ── k_deg  ──►  ∅      (degradation: each mRNA decays independently)
```

This is the simplest interesting stochastic network, and — crucially for
teaching — its stationary distribution is known in closed form (§2), so we can
check that our simulator recovers the right physics, not merely that CPU==GPU.

---

## 2. The math

### 2.1 The Chemical Master Equation (CME)

Let `P(n, t)` be the probability that `M = n` at time `t`. Reactions change `n`
by their stoichiometry and fire at rates given by the **propensities** `a_j(n)`
(defined so that `a_j(n)·dt` is the probability reaction `j` fires in `[t, t+dt)`):

- R1 (source): `a₁ = k_prod` (independent of `n`).
- R2 (degradation): `a₂ = k_deg · n` (each of the `n` molecules can decay).

The CME is the balance equation for `P(n, t)`:

```
dP(n,t)/dt =  k_prod·P(n-1,t)              (enter n from n-1 via R1)
           +  k_deg·(n+1)·P(n+1,t)         (enter n from n+1 via R2)
           - (k_prod + k_deg·n)·P(n,t)     (leave n via either reaction)
```

### 2.2 The stationary distribution (our analytic check)

Setting `dP/dt = 0` and solving the detailed-balance recursion gives a **Poisson
distribution**:

```
P_stationary(n) = e^(−λ) · λ^n / n! ,     with   λ = k_prod / k_deg
```

So both the stationary **mean** and **variance** equal `λ = k_prod/k_deg`. For the
sample (`k_prod = 10`, `k_deg = 0.5`), `λ = 20`. The demo prints the ensemble mean
next to this analytic `20` — a direct test that the simulation samples the CME
correctly.

### 2.3 Mass-action propensities in general

For a general well-mixed network, the propensity counts the number of distinct
reactant combinations available (statistical mechanics of collisions):

| Reaction order | Example        | Propensity `a`                |
|----------------|----------------|-------------------------------|
| 0 (source)     | `∅ → X`        | `k`                           |
| 1              | `A → …`        | `k · x_A`                     |
| 2 (hetero)     | `A + B → …`    | `k · x_A · x_B`               |
| 2 (homodimer)  | `A + A → …`    | `k · x_A·(x_A−1)/2`           |

The homodimer form uses `n(n−1)/2` = the number of *unordered pairs* of identical
molecules — the correct discrete count, not `n²`. `propensity()` in
[`src/ssa.h`](src/ssa.h) implements exactly this table.

---

## 3. The algorithm — Gillespie's Direct Method

Gillespie (1976, 1977) proved that the following procedure generates **statistically
exact** trajectories of the CME — no time discretization, no approximation:

```
t ← 0,  x ← x0
while t < t_end:
    1. compute a_j = propensity(j, x) for every reaction j
       a0 = Σ_j a_j                                  # total event rate
    2. if a0 == 0: system is frozen → stop
    3. τ  = −ln(u1) / a0        with u1 ~ Uniform(0,1)   # time to next event
    4. k  = smallest index with  Σ_{j≤k} a_j  >  u2·a0   # which reaction fires
    5. x ← x + ν_k             # apply stoichiometry
       t ← t + τ
```

**Why this is exact.** Given the current state, the time to the *next* event of
*any* kind is Exponential(`a0`) (the minimum of independent exponential clocks,
one per reaction), and the probability that the event is reaction `k` is
`a_k/a0`. Steps 3–4 sample exactly those two facts. There is no `Δt` to shrink —
every step is a real reaction at a correctly distributed time.

### 3.1 Sampling the two random draws

- **Exponential waiting time:** if `u ~ Uniform(0,1)` then `−ln(u)/a0 ~
  Exponential(a0)` (inverse-CDF sampling). We use `1−u` so we never take `ln(0)`.
- **Reaction selection:** draw `u2·a0`, then walk the cumulative propensities
  until the running sum exceeds it (linear "roulette wheel"). For a large network
  a prefix-sum + binary search is `O(log R)` instead of `O(R)`; with `R = 2` the
  linear scan wins outright.

### 3.2 The observable we record

For each trajectory we record the **time-averaged count**

```
  ⟨M⟩_traj = (1/T) ∫₀ᵀ M(t) dt
```

Because `M(t)` is a **step function** (constant between events), this integral is
an *exact finite sum* `Σ M·(time spent at that value)` — no quadrature error.
Averaging `⟨M⟩_traj` over the ensemble estimates the stationary mean `λ` (§2.2).

### 3.3 Complexity

- **Per trajectory:** `O(E · R)` where `E` = number of events fired and
  `R` = reactions. `E` grows with copy numbers and `t_end`; for the sample each
  trajectory fires ~1000 events. There is **no** `1/Δt` factor — a key advantage
  over fixed-step methods when events are rare.
- **Ensemble of `N` trajectories, serial:** `O(N · E · R)`.
- **On the GPU:** the `N` trajectories run concurrently, so wall-clock ≈
  `O(E · R)` × (divergence + occupancy factors). The GPU's edge grows with `N`.

---

## 4. The GPU mapping

```
        ensemble of N independent SSA trajectories
   ┌──────────┬──────────┬──────────┬─────────────┐
   │ traj 0   │ traj 1   │ traj 2   │ ...  traj N-1│
   └────┬─────┴────┬─────┴────┬─────┴──────┬───────┘
        │          │          │            │
     thread 0   thread 1   thread 2 ...  thread N-1     (one thread = one traj)
        │          │          │            │
   [own RNG]   [own RNG]  [own RNG]    [own RNG]        (splitmix64, seed=idx)
        │          │          │            │
   full SSA    full SSA   full SSA     full SSA         (event loop in registers)
   time loop   time loop  time loop    time loop
        │          │          │            │
     out[0]     out[1]     out[2]  ...  out[N-1]        (one write, no races)
```

- **Thread-to-data map:** `idx = blockIdx.x*blockDim.x + threadIdx.x` owns
  trajectory `idx` (guard `idx < N` for the ragged last block). See
  [`src/kernels.cu`](src/kernels.cu).
- **Memory hierarchy.** Each thread keeps its entire working set — the RNG state,
  the small fixed-size molecule-count array, and the running time-integral — in
  **registers / local memory**. The reaction network is a POD struct passed **by
  value** in the kernel's parameter space (it rides in constant/param memory,
  broadcast to all threads), so there is *no* `cudaMalloc` for the network and no
  global-memory traffic in the inner loop. The only global write is the single
  `out[idx] = result` at the end.
- **No atomics, no shared memory, no `__syncthreads()`.** Trajectories never
  interact. This is the "embarrassingly parallel Monte-Carlo" ideal — contrast
  with `5.01` (Monte-Carlo dose) where many histories *tally into shared bins* and
  therefore need atomics.
- **Occupancy / block size.** We use 128 threads/block. Each thread here is
  *heavy* (a whole SSA run) rather than a one-line SAXPY, so register pressure per
  SM matters; a smaller block keeps enough blocks resident while still giving the
  scheduler several warps to hide latency.
- **Divergence — the real cost.** Different trajectories fire different numbers of
  events, so within a warp the fast lanes idle until the slowest finishes its
  `while` loop. This is intrinsic to *exact* SSA. Tau-leaping (fixed number of
  sub-steps) is the standard remedy that trades exactness for warp-uniform work.

---

## 5. Numerical considerations

- **Precision.** Molecule counts are exact `uint64` integers — no rounding. The
  waiting-time and time-integral math is `double` (FP64).
- **Determinism & the RNG.** The whole verification strategy rests on the CPU and
  GPU drawing the **same** random numbers. We therefore use a **shared
  `__host__ __device__` splitmix64** counter-based RNG (integer ops + shifts only,
  identical on both), seeded per trajectory from `(base_seed, idx)`. Production
  GPU-SSA uses cuRAND, whose device streams cannot be reproduced bit-for-bit on
  the host — so it is verified statistically, not exactly (see §7).
- **Why integers are the trick.** Because the *state transitions* are integer
  additions and the *reaction choice* is an integer index, the two sides take
  literally the same branches in the same order. Contrast floating-point atomic
  sums, which are non-associative and non-deterministic (`5.01`, `11.09` avoid this
  with integer/fixed-point tallies).
- **The one non-exact number.** The time-average `⟨M⟩ = Σ M·τ / T` is a
  floating-point sum. On the GPU the compiler fuses `M*τ + acc` into a single FMA
  (one rounding) while the host may do a separate multiply then add (two
  roundings). Over ~1000 events this makes the CPU and GPU time-averages differ by
  `~1e-15` — real, tiny, and worth teaching. We verify to `1e-9` and *say so*
  rather than pretending the doubles are bit-identical (PATTERNS.md §4).
- **Runaway guard.** A hard cap (`MAX_EVENTS = 5,000,000`) bounds every thread so
  a mis-specified explosive network cannot hang the GPU.

---

## 6. How we verify correctness

Two independent checks:

1. **CPU==GPU, per trajectory.** Same shared SSA core + same seed ⇒ the integer
   outputs (final count, event count) match **exactly**, and the time-average
   matches to `~1e-15` (§5). `main.cu` reports the worst difference; the demo gates
   on it.
2. **Recovering the physics (analytic).** The ensemble mean of `⟨M⟩_traj` must
   approach the closed-form stationary mean `λ = k_prod/k_deg = 20`. With 256 short
   trajectories the demo recovers ≈19.2 — the gap is finite-sample Monte-Carlo
   error, which shrinks like `1/√N` (Exercise 1). This validates that we sample the
   *correct* distribution, not just that two implementations agree.

Edge cases handled: `a0 = 0` (frozen system → fast-forward to `t_end`); an event
that would overshoot `t_end` (integrate the remaining constant window and stop);
stoichiometry that would drive a count below zero (clamped, guarding `uint64`
underflow).

---

## 7. Where this sits in the real world

- **cuRAND, not splitmix64.** Real GPU-SSA (e.g. cuTauLeaping, GillesPy2's GPU
  backend) gives each thread a cuRAND state (`Philox`/`XORWOW`) for high-quality,
  fast device randoms. You gain speed and statistical quality; you lose bit-exact
  CPU reproducibility, so you validate against analytic moments or a large-sample
  reference distribution instead (Exercise 3).
- **Thrust prefix-sum for reaction selection.** With hundreds of reactions,
  `O(R)` roulette dominates; a parallel prefix-sum over propensities + a binary
  search (`O(log R)`) is the standard fix. For our two-reaction model it would be
  pure overhead.
- **Tau-leaping and hybrids.** At high copy numbers the exact SSA fires an
  astronomical number of events. Tau-leaping fires `Poisson(a_j·τ)` events of each
  reaction per fixed `τ`, trading exactness for speed and *uniform* per-thread work
  (fixing GPU divergence). Hybrid methods run abundant species as ODE/Langevin and
  only rare species with exact SSA.
- **Spatial stochastic (RDME).** When "well-mixed" fails, the domain is divided
  into voxels with diffusion as extra reactions (the Reaction-Diffusion Master
  Equation) — combining this project's pattern with the stencil pattern of `14.02`.
- **Model provenance.** Production studies pull curated networks from the
  **BioModels Database** as SBML; our single-species model is a hand-built,
  analytically-checkable teaching stand-in.

---

## 8. Summary

| Aspect | This project |
|---|---|
| Problem | Exact stochastic simulation of a small well-mixed reaction network |
| Method | Gillespie SSA, direct method (exact CME sampling) |
| Model | Birth–death gene expression, `λ = k_prod/k_deg` (Poisson stationary) |
| GPU pattern | One thread per trajectory; per-thread RNG; no atomics/shared/sync |
| Shared core | `__host__ __device__` RNG + SSA loop ⇒ CPU==GPU exact (integers) |
| Verification | Per-trajectory CPU==GPU **and** ensemble mean ≈ analytic `λ` |
| Honest caveat | Time-average diverges `~1e-15` (FMA); divergent warps; not cuRAND |
