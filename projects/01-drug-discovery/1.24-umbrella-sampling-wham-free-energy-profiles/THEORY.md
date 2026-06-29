# THEORY — 1.24 Umbrella Sampling / WHAM Free Energy Profiles

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use. The landscape here is synthetic._

---

## 1. The science

Whether a drug binds, whether an ion crosses a channel, whether a protein changes
shape — all of these are governed not by *energy* alone but by **free energy**,
which folds in both energy and entropy at a temperature `T`. If we pick a single
geometric quantity that tracks the process — a **reaction coordinate** `ξ` (Greek
"xi"), such as the distance between a ligand and its pocket, or an ion's depth in a
pore — then the function we care about is the **potential of mean force (PMF)**:

```
F(ξ) = −kT · ln p(ξ)
```

where `p(ξ)` is the equilibrium probability of finding the system at coordinate
value `ξ`, `k` is Boltzmann's constant, and `kT` is the thermal energy scale. The
PMF is the effective free-energy landscape *along* `ξ`: its minima are stable
states (bound / unbound), and the **height of the barrier** between them sets the
rate and the binding strength.

**The sampling problem.** To estimate `p(ξ)` you simulate the system and build a
histogram of `ξ`. But if `F(ξ)` has a barrier of height `ΔF`, the system spends a
factor `exp(−ΔF/kT)` *less* time at the barrier top than in the wells. A 10 kT
barrier means the top is visited `~e^−10 ≈ 1/22000` as often as a well — so a
feasible simulation **never samples it**, and `F(ξ)` there is simply unknown.

**Umbrella sampling** (Torrie & Valleau, 1977) is the classic cure. Add a stiff
**harmonic restraint** ("umbrella") that holds the system near a chosen value
`x0_k`. Now the simulation is *forced* to sample a small neighbourhood around
`x0_k` — even if that sits on top of the barrier. Run a whole row of such
**windows** with centers `x0_k` marching across the coordinate, collect one
histogram per window, and then mathematically remove the bias and merge the
windows into the single unbiased PMF. The merge step is **WHAM**.

This project models the simplest system that *has* a barrier: a **symmetric
double-well** along a 1-D coordinate — the canonical picture of a two-state process
(think "bound" at `x=−b` vs. "unbound" at `x=+b`, with a transition barrier at
`x=0`). Because the landscape is known analytically, we can grade WHAM's answer.

> *Synthetic, educational.* A real study restrains a collective variable of an
> all-atom system and runs molecular dynamics; we run a 1-D toy with the same
> statistical mechanics so the ideas stay visible.

## 2. The math

**The true landscape.** A quartic double-well, in reduced units where `kT = 1`:

```
U(x) = A · (x² − b²)² / b⁴
```

- `A` — barrier height at `x = 0`, in units of kT.
- `b` — half-separation of the minima (the two wells sit at `x = ±b`, where `U=0`).

Its force (used by the dynamics) is `F_U(x) = −dU/dx = −4A·x·(x²−b²)/b⁴`.

**The umbrella bias.** Window `k` adds a harmonic restraint about its center
`x0_k`:

```
w_k(x) = ½ · k_spring · (x − x0_k)²
```

Window `k` therefore samples the **biased** Boltzmann distribution

```
p_k^b(x) ∝ exp( −[U(x) + w_k(x)] / kT ).
```

**The goal.** Recover the *unbiased* `p(x) ∝ exp(−U(x)/kT)` (and hence
`F(x) = −kT ln p(x)`) from the set of biased histograms `{N_{k,i}}`, where
`N_{k,i}` is window `k`'s count in bin `i`.

**WHAM.** The optimal (minimum-variance) unbiased estimate of the probability in
bin `i`, pooling all windows, is the pair of coupled equations

```
                 Σ_k N_{k,i}
   p_i  =  ───────────────────────────────────────────      (1)
            Σ_k  Ntot_k · exp( −[ w_k(x_i) − f_k ] / kT )

   exp(−f_k / kT)  =  Σ_i  p_i · exp( −w_k(x_i) / kT )       (2)
```

where `Ntot_k = Σ_i N_{k,i}` is window `k`'s total count, `x_i` is bin `i`'s
center, and `f_k` is a per-window **free-energy offset** (the free energy of
turning on window `k`'s bias). Equations (1)–(2) are the WHAM equations; the PMF is
`F_i = −kT ln p_i`, defined only up to an additive constant (we shift `min F = 0`).

Symbols: `i` indexes bins (`0..nbins−1`), `k` indexes windows (`0..n_windows−1`),
`x_i = x_min + (i+½)·Δx`, `Δx = (x_max−x_min)/nbins`.

## 3. The algorithm

Two stages, with very different costs.

**Stage A — generate biased samples (the expensive part).** For each window `k`,
integrate overdamped (Brownian) **Langevin dynamics** under the biased force and
histogram the coordinate. Overdamped Langevin is the high-friction limit of
Newton's equations, appropriate for a coordinate immersed in solvent:

```
x_{n+1} = x_n + (D/kT)·F_total(x_n)·dt + sqrt(2·D·dt)·ζ_n,   ζ_n ~ N(0,1)
```

with `F_total = F_U(x) + F_bias(x)`, `F_bias(x) = −k_spring·(x − x0_k)`, and `D` the
diffusion constant. The first term is deterministic drift down the biased gradient;
the second is the random thermal kick whose variance `2·D·dt` is fixed by the
**fluctuation–dissipation theorem** so that the stationary distribution is exactly
`p_k^b(x)` above. We discard `n_equil` warm-up steps (to forget the artificial
start) then histogram `n_sample` steps.

- Cost per window: `O(n_equil + n_sample)` steps, each `O(1)` work.
- Total Stage A: `O(n_windows · (n_equil + n_sample))`.
- **Parallel structure:** windows are independent → fully parallel across windows;
  each window is sequential in time (depth `= n_equil + n_sample`).

**Stage B — WHAM (the cheap part).** Iterate (1)↔(2) to a fixed point: start
`f_k = 0`, update all `p_i` via (1), update all `f_k` via (2), repeat `iters`
times.

- Cost per sweep: `O(n_windows · nbins)`. Total: `O(iters · n_windows · nbins)`.
- For the demo (`27 × 32`, 200 sweeps) this is microseconds — hence "WHAM on the
  CPU", matching the catalog pattern.

Then `F_i = −kT ln p_i`, shifted so the minimum is 0.

## 4. The GPU mapping

This is the **ensemble pattern** (`docs/PATTERNS.md` §1: "the same simulation for
many independent parameter sets → one thread per simulation"), the same shape as
flagships `9.02` (SEIR ensemble) and `13.02` (PBPK ensemble).

**Thread-to-data mapping.** One thread owns one window:

```
k = blockIdx.x · blockDim.x + threadIdx.x        // this thread's window index
```

Thread `k` reads window `k`'s center via `window_center()`, runs the *entire*
biased Langevin trajectory in registers/local memory, and writes its histogram
into the disjoint slice `hist[k·nbins .. k·nbins + nbins − 1]`.

**Launch configuration.** `block = 128` threads (each thread is register-heavy
because it carries a whole trajectory's state, so a smaller block keeps register
pressure down while still giving the scheduler 4 warps/block); `grid =
ceil(n_windows / 128)` blocks. A boundary guard `if (k >= n_windows) return;`
protects the ragged last block.

**Memory hierarchy & why.**
- **Registers / local memory:** the entire per-window state (`x`, the RNG word,
  loop counters) lives here — no global traffic during the inner loop, which is
  exactly why this kernel is compute-bound rather than bandwidth-bound.
- **Global memory:** only the output histogram. Crucially, **each thread writes a
  private slice**, so there is *no contention and no atomics* — a cleaner cousin of
  the histogram-accumulation pattern. (Contrast `5.01` Monte-Carlo dose, where many
  threads deposit into *shared* bins and therefore must use `atomicAdd`.)
- **Constant/shared:** not needed at this scale; the config is a small POD passed
  by value.

**No CUDA library is used.** The work is a hand-written integrator and histogram —
there is no FFT/GEMM/sort step to delegate, so there is no black box to explain
(CLAUDE.md §6.1.6). WHAM runs on the host with plain `std::exp`/`std::log`.

```
grid (one thread per window)
┌──────────────────────────────────────────────┐
│ thread 0    thread 1    ...    thread 26       │   27 windows
│   │           │                  │             │
│  run win 0   run win 1         run win 26      │   each: equilibrate + sample
│   │           │                  │             │
│   ▼           ▼                  ▼             │
│ hist[0..31] hist[32..63] ... hist[832..863]    │   disjoint slices (no atomics)
└──────────────────────────────────────────────┘
                     │
                     ▼   (copy histograms to host)
              WHAM on CPU  →  PMF F(ξ)
```

## 5. Numerical considerations

- **Precision: FP64 throughout.** Both the dynamics and WHAM use `double`. The
  Langevin update accumulates millions of steps; FP64 keeps the trajectory
  statistics clean and, with the shared core, makes CPU/GPU agreement *exact*.
- **The shared `__host__ __device__` core (`umbrella.h`).** RNG, potential,
  Langevin step, and binning are defined *once* and compiled for both host and
  device (the `US_HD` macro). This guarantees the CPU reference and the GPU kernel
  run *byte-identical* simulations — the basis of the exact histogram check.
- **Deterministic by construction.** The histogram is **integer counts**, and
  integer addition commutes, so the result does not depend on thread scheduling
  (`docs/PATTERNS.md` §3). Each window seeds an *independent* counter-based RNG
  stream (splitmix64 → Box–Muller Gaussian) from `(seed, window_index)`, so every
  run — and every GPU/CPU pairing — produces the same histograms.
- **No race conditions.** Because each thread owns a private histogram slice, there
  is no shared-write hazard and `atomicAdd` is unnecessary. (If you ever collapsed
  all windows into one shared histogram, you would need integer atomics — still
  deterministic, because integer atomics commute.)
- **Floating-point in WHAM.** The fixed-point iteration multiplies and sums
  `exp(±·)` terms; for the demo's moderate barriers plain doubles are fine.
  Production codes use the **log-sum-exp** trick to avoid overflow/underflow when
  barriers are tens of kT — noted as an exercise.

## 6. How we verify correctness

Two independent checks, by design (see `src/main.cu`):

1. **Exact GPU == CPU histograms (`tolerance = 0`).** Integer counts produced by
   identical physics *must* match bit-for-bit; we count mismatching bins and
   require zero. This is the strongest possible agreement and catches any
   divergence between the two code paths (`docs/PATTERNS.md` §4, "exact when the
   same exact operations run on both sides").

2. **WHAM PMF recovers the known landscape (`tolerance = 0.30 kT`).** Because the
   true `U(x)` is known, we compare WHAM's `F_i` to `U(x_i)` (both shifted to a
   common zero). This is a **sampling** comparison, not a round-off one: even with
   identical histograms, the WHAM estimate differs from `U` by finite-sampling
   noise that shrinks like `1/√N`. For the committed sample (1.62M samples) the
   worst interior error is ~0.2 kT, and the barrier is recovered as ~4.00 kT vs. a
   true 4.0 kT. `0.30 kT` is a generous, honest bound, documented as such.

   We judge the PMF on the **interior** of the scan (excluding a margin near
   `win_min`/`win_max`). This is not cherry-picking: the outermost windows have a
   *one-sided* harmonic well (no neighbour beyond the edge), so their histograms
   are genuinely noisier — a real property of every umbrella scan. The full sweep
   is still printed so the learner sees the edges.

Why this is convincing: an independent, obviously-correct serial implementation and
a parallel one, written to different constraints, agreeing exactly on the
histograms — *and* the end-to-end pipeline reproducing a landscape it was never
told — is strong evidence the implementation is right.

## 7. Where this sits in the real world

Production umbrella sampling differs from this toy in scale, not in spirit:

- **Full molecular dynamics per window.** Instead of a 1-D Langevin step, each
  window runs all-atom MD (thousands–millions of atoms, a force field, PME
  electrostatics) on the GPU. The reaction coordinate is a **collective variable**
  (a distance, angle, RMSD, coordination number) restrained via a plugin like
  **PLUMED** or the MD engine's own pull code. This is the "full MD per window on
  GPU" the catalog calls out, and it is where the GPU's advantage is overwhelming —
  each step is a huge parallel force evaluation, not a single scalar update.
- **Launching the window array.** With hundreds of windows, runs are distributed
  across GPUs/nodes with **MPI + NCCL** — the "window array" launch. Our 27 threads
  are the single-GPU shadow of that.
- **WHAM at scale / MBAR.** `gmx wham` (GROMACS) and **alchemlyb** implement WHAM
  with overlap diagnostics and error bars. The modern successor is **MBAR**
  (multistate Bennett acceptance ratio): binless, uses every sample's energy in
  every state, and has lower variance — the preferred estimator today.
- **Alternatives to umbrella sampling.** **Steered MD + the Jarzynski equality**
  extracts free energies from nonequilibrium pulling; **metadynamics** and
  **adaptive biasing force** build the bias *adaptively* instead of via fixed
  windows. All target the same `F(ξ)`; THEORY-level mentions only here.

The honest gap: this project teaches the *statistics* (biased sampling → WHAM →
PMF) exactly, while standing in a 1-D analytic potential for the *physics* (the
force field + atoms) that a real free-energy calculation spends its FLOPs on.

---

## References

- **Torrie & Valleau (1977)**, *Nonphysical sampling distributions in Monte Carlo
  free-energy estimation: umbrella sampling*, J. Comput. Phys. 23:187 — the origin
  of the method.
- **Kumar, Bouzida, Swendsen, Kollman, Rosenberg (1992)**, *The weighted histogram
  analysis method...*, J. Comput. Chem. 13:1011 — the WHAM equations used here.
- **Shirts & Chodera (2008)**, *Statistically optimal analysis of samples from
  multiple equilibrium states* (MBAR), J. Chem. Phys. 129:124105 — the modern
  successor to WHAM.
- **Hub, de Groot, van der Spoel (2010)**, *g_wham — A Free Weighted Histogram
  Analysis Implementation...*, J. Chem. Theory Comput. 6:3713 — GROMACS' production
  WHAM; study its overlap/error diagnostics.
- **GROMACS umbrella-sampling tutorial** — <https://tutorials.gromacs.org> — a
  hands-on worked example end to end.
- **PLUMED** — <https://github.com/plumed/plumed2> — collective variables and
  biases (umbrella, metadynamics) as a portable plugin.
- **alchemlyb** — <https://github.com/alchemistry/alchemlyb> — clean Python
  MBAR/WHAM post-processing to compare against.
