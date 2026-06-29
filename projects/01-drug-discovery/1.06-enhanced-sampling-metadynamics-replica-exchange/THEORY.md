# THEORY — 1.6 Enhanced Sampling — Metadynamics & Replica Exchange

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See
> [README.md](README.md) for the quick tour and build steps.
>
> _Educational only — not for clinical use. This project ships a clearly-labeled
> **reduced-scope teaching version** (a 1-D analytic model). The full method, as
> implemented in PLUMED + GROMACS on real molecular systems, is described in
> "Where this sits in the real world" at the end._

---

## The science

### The rare-event problem in molecular dynamics

Molecular dynamics (MD) integrates Newton's (or Langevin's) equations for every
atom in a biomolecule, taking femtosecond steps. To watch a drug molecule unbind
from a protein, or a peptide fold, you must cross a **free-energy barrier** — a
high-energy transition state separating two stable conformational basins (e.g.
"ligand bound" vs "ligand unbound"). The rate of crossing scales like the
Arrhenius factor `exp(-ΔF / kT)`: a barrier of 15 kT is crossed roughly once
every `e^15 ≈ 3×10⁶` characteristic times. At MD timescales that can mean
**milliseconds to seconds of simulated time** — months of wall-clock even on a
GPU. Plain MD simply never sees the event. This is the **rare-event problem**.

### Enhanced sampling: bias the dynamics, then correct for it

Enhanced-sampling methods deliberately distort the dynamics to cross barriers
faster, in a way that can be **rigorously undone** afterwards so the underlying
thermodynamics is still recovered. Two big families:

- **Metadynamics** (the focus here): pick one or a few **collective variables**
  (CVs) `s` — low-dimensional descriptors of the slow motion (a torsion angle, a
  ligand–pocket distance). Periodically add a small repulsive **Gaussian "hill"**
  of bias potential `V_bias(s)` at the walker's current CV value. The bias fills
  up whichever basin the system sits in, like sand poured into a valley, until
  the system is pushed over the barrier. Remarkably, the accumulated bias also
  **reconstructs the free-energy surface** (FES): once converged,
  `F(s) ≈ -V_bias(s)` (up to a constant and a temper factor).
- **Replica exchange (REMD / parallel tempering)**: run many copies ("replicas")
  at different temperatures; high-temperature replicas cross barriers easily, and
  a Metropolis swap move occasionally exchanges configurations between replicas,
  letting the cold replica inherit a barrier-crossed structure. **HREX / REST2**
  scale the *Hamiltonian* (e.g. solute–solute interactions) instead of the global
  temperature. We describe REMD in "real world"; the runnable code implements
  **well-tempered metadynamics with multiple walkers**, which is the more direct
  fit to the GPU "ensemble of trajectories" pattern.

### The teaching model: a 1-D double well

A real FES is unknown — discovering it is the whole point. To make the method
**verifiable**, we use a synthetic landscape whose FES we know exactly: a
particle diffusing on the quartic **double well**

```
F0(s) = A (s² − 1)²        minima at s = ±1 (F0 = 0),  barrier A at s = 0.
```

This is the canonical stand-in for a molecular CV with two metastable states
(think of the two φ/ψ basins of alanine dipeptide, or a bound vs unbound ligand
pose projected onto one reaction coordinate). Because we *know* `F0`, we can
check that metadynamics reconstructs it — a luxury you never have in production,
which is exactly why it is the right teaching example.

```
F0(s)
  A |‾‾\          /‾‾        barrier (transition state) at s=0, height A
    |   \        /
    |    \      /
  0 |_____\____/______  s
        -1    +1            two metastable basins (the wells)
```

---

## The math

### Biased Langevin dynamics

The walker's CV `s` obeys the **Langevin equation** (a stochastic ODE that models
contact with a heat bath at temperature `T`):

```
m s̈ = f_total(s) − m γ_L ṡ + √(2 m γ_L kT) · ξ(t),
```

where `m` is the (effective) mass, `γ_L` the friction, `ξ(t)` Gaussian white
noise with `⟨ξ(t)ξ(t')⟩ = δ(t−t')`, and the total force is the conservative force
plus the metadynamics bias force:

```
f_total(s) = −dF0/ds − dV_bias/ds
           = −4 A s (s² − 1) − dV_bias/ds.
```

The fluctuation–dissipation theorem fixes the noise amplitude `√(2 m γ_L kT)` so
that *without* bias the walker samples the Boltzmann distribution
`p(s) ∝ exp(−F0(s)/kT)`.

### Well-tempered metadynamics

Standard metadynamics deposits, every `τ` steps, a Gaussian hill of fixed height
`w` and width `σ` at the current position `s*`:

```
V_bias(s) ← V_bias(s) + w · exp(−(s − s*)² / (2σ²)).
```

The problem: the bias never stops growing, so it overfills the wells and the FES
estimate oscillates. **Well-tempered** metadynamics (Barducci, Bussi, Parrinello,
2008) fixes this by shrinking each new hill in proportion to the bias *already*
deposited where it lands:

```
w_eff = w · exp( −V_bias(s*) / (kT (γ − 1)) ),          γ = bias factor > 1.
```

As `V_bias` grows, `w_eff → 0`, so the bias **converges**. At convergence the
recovered free energy is

```
F(s) = −(γ / (γ − 1)) · V_bias(s) + const
     = −(1 + 1/(γ−1)) · V_bias(s) + const.
```

`γ → ∞` recovers standard metadynamics (`F = −V_bias`); `γ → 1` recovers plain MD
(no bias). The bias factor `γ` controls how much the effective temperature of the
CV is raised: `T_eff = γ·T` along `s`.

### Why the recovered bias *is* (minus) the free energy

Intuition: at convergence the biased system samples `s` almost uniformly (the bias
has flattened the landscape). For the *biased* distribution
`p_b(s) ∝ exp(−(F0(s) + V_bias(s))/kT)` to be flat, we need
`F0(s) + V_bias(s) = const`, i.e. `V_bias(s) = −F0(s) + const`. The well-tempered
factor `γ/(γ−1)` is the exact correction for the fact that the bias only partially
(not fully) flattens the surface. This is the magic of metadynamics: a *sampler*
that simultaneously **measures** the thing it is sampling.

---

## The algorithm

### Per-walker integration (the inner loop)

For one walker, starting at `s0` with `v = 0` and an empty bias grid:

```
for step = 0 .. N-1:
    langevin_step(s, v)               # one BAOAB-style symmetric Langevin step
    if step mod τ == 0:               # the metadynamics "pace"
        deposit_hill(bias, s)         # add a well-tempered Gaussian at s
```

`langevin_step` is a symmetric splitting (thermostat half-kick → force half-kick →
drift → force half-kick → thermostat half-kick) — accurate and stable for these
smooth dynamics. The bias enters through the force `−dV_bias/ds`, read from the
grid by linear interpolation.

### The bias grid (avoiding O(#hills) per step)

Naively, evaluating `V_bias(s)` sums every hill ever deposited — O(#hills), which
grows without bound. Instead we store `V_bias` on a **uniform grid** of `nbins`
points over `[s_lo, s_hi]`. Depositing a hill adds the Gaussian to every grid
point (O(nbins), but only every `τ` steps); reading the bias/force is O(1) by
linear interpolation. This is "grid metadynamics", the standard production choice.

### Complexity

| Quantity | Serial (CPU) | Parallel (GPU) |
|---|---|---|
| One walker, `N` steps | `O(N)` force evals + `O((N/τ)·nbins)` deposits | same, on one thread |
| `M` walkers | `O(M · N)` (loop over walkers) | `O(N)` **wall-clock** (M threads at once) |
| Memory | one `nbins` grid reused | `M · nbins` doubles (one grid per thread) |

The serial cost is `M·N` force evaluations done one after another; the GPU runs
all `M` walkers **concurrently**, so the wall-clock is set by a single walker's
`N` steps (modulo occupancy). That is the entire speed-up argument for
multi-walker enhanced sampling.

---

## The GPU mapping

### Pattern: ensemble of independent trajectories (thread per walker)

This is the **ensemble / thread-per-trajectory** pattern from
[`docs/PATTERNS.md`](../../../docs/PATTERNS.md) §1 — the same shape as flagships
**9.02** (SEIR ensembles) and **13.02** (PBPK virtual patients). Each walker's
trajectory is sequential in time but **independent** of every other walker, so:

```
thread id = blockIdx.x * blockDim.x + threadIdx.x   →   walker id
```

Thread `id` runs the *entire* `run_walker()` loop (Langevin + deposition) and
writes back its private bias grid and a small summary. No inter-thread
communication, no atomics, no synchronization. Embarrassingly parallel.

```
ensemble of M walkers, one per GPU thread:

  block 0          block 1                 block ⌈M/128⌉-1
 ┌──────────┐    ┌──────────┐             ┌──────────┐
 │t0 t1 …t127│   │t0 t1 …t127│   …         │ … tM-1   │   each thread:
 └─┬─────────┘   └──────────┘             └──────────┘     run_walker()
   │ owns bias slice  d_bias[id*nbins : (id+1)*nbins]      → its own FES
```

### Memory hierarchy

- **Global memory**: the big `d_bias` buffer of `M·nbins` doubles. Thread `id`
  reads/writes only its contiguous slice `[id*nbins, (id+1)*nbins)`. Each walker
  touches its grid every step (read for the force) and every `τ` steps (write for
  a deposit). The accesses are *not* coalesced across threads (each walker reads a
  different bin), which is the main inefficiency of this simple version — see
  Exercises.
- **Registers**: the walker state (`s`, `v`), loop counters, and the RNG counter
  live in registers — the inner loop is register-resident and global-memory-light
  apart from the bias grid.
- **Constant memory**: not used here (the `Model` struct is passed by value as a
  kernel argument, which the compiler places in constant/param memory
  automatically). The Tanimoto flagship (1.12) shows the explicit `__constant__`
  idiom.

### Why not shared memory / a shared hill list?

Production multi-walker metadynamics has all walkers deposit into **one shared**
bias, so they cooperate (a barrier crossed by one walker helps all). That needs
atomic updates to a shared grid and periodic synchronization — real but more
involved. We keep walkers **independent** here so the teaching kernel needs no
atomics; the shared-bias version is the headline Exercise. Each independent walker
still produces a valid FES; we average them for a smoother estimate.

### CUDA-library note

This project links only `cudart` (the runtime). No cuBLAS/cuFFT/etc. is needed:
the work is a per-thread time-stepping loop, not a dense linear-algebra or
transform problem. (Compare 8.03, which genuinely needs cuFFT.) The catalog's
"GPU-MetaD" reference uses libraries for on-the-fly CV evaluation on large atomic
systems — out of scope for a 1-D model.

---

## Numerical considerations

### Precision

Everything is **double precision** (`double`). The Langevin integrator,
Box–Muller transform, and bias accumulation all benefit: single precision would
lose the small force differences that distinguish nearby grid points.

### Determinism and the counter-based RNG

For the CPU reference and the GPU kernel to be comparable at all, they must draw
the **same** Gaussian noise sequence, independent of thread scheduling. We use a
**counter-based RNG**: the `n`-th normal sample is a pure hash of
`(seed, walker_id, step)` — no hidden evolving state — via a SplitMix64 finalizer
and a Box–Muller transform (`metad.h §4`). Same inputs → same bits, on host and
device, in any order. This is the same philosophy as cuRAND's Philox generator.

### Chaos: why individual trajectories do NOT match across CPU and GPU

This is the most important — and most teachable — numerical point in the project.
Even with identical RNG and `--fmad=false` (which forces the device to compute
`a*b+c` as a separate multiply-then-add, matching the host), the CPU and GPU
trajectories **diverge by O(1)** after a few thousand steps. Why?

1. The device and host **transcendental libraries** (`exp`, `log`, `sin`, `cos`,
   `sqrt`) are correctly rounded to slightly different results — they differ by
   about **1 unit in the last place (ULP)**, ~10⁻¹⁶.
2. The biased Langevin dynamics on a double well are **chaotic**: nearby
   trajectories separate exponentially (positive Lyapunov exponent). A 10⁻¹⁶ seed
   of difference doubles every so-many steps and reaches O(1) within the run.

So `final_s`, the exact crossing count, etc. are **not reproducible across
platforms**. This is *not a bug* — it is a fundamental property of chaotic
systems (the same reason weather is unpredictable past ~2 weeks). We therefore
print those per-walker numbers only to **stderr** as machine-local diagnostics.

### What IS reproducible: the ensemble free-energy surface

The **statistical observable** — the ensemble-averaged recovered FES — is robust.
Averaging over `M` walkers, plus the self-flattening (`w_eff → 0`) nature of
well-tempered metadynamics, makes the FES converge to nearly the same surface on
both platforms even though the trajectories that produced it differ. This is the
deep lesson: in statistical mechanics you verify **distributions and free
energies**, never individual chaotic trajectories.

---

## How we verify correctness

Two complementary checks (see `main.cu` and `docs/PATTERNS.md` §4):

1. **CPU vs GPU agree on the recovered FES** (the robust observable). We recover
   `F(s)` from each platform's ensemble-mean bias and compare over the
   well-sampled core `s ∈ [−1.3, 1.3]` (the grid edges are barely visited, so
   their bias is near-zero noise we deliberately exclude — standard MetaD
   practice). Tolerance **0.25 kT** (a quarter of thermal energy): a generous,
   honest "the two platforms recover the SAME surface" bound. Observed: ~0.17 kT.
   We do **not** assert bit-identity — that would be dishonest for chaotic
   dynamics (see above).
2. **Science check against the known landscape.** The recovered barrier height
   `F_est(0)` must match the analytic barrier `A` within **0.35 kT**. This
   validates the *physics* (metadynamics actually recovered the free energy), not
   just that two codes agree. Observed: barrier err ~0.02 kT (5.0 kT recovered vs
   5.0 kT true).

`expected_output.txt` is **captured from a real run**, and stdout prints the FES
rounded to 0.1 kT — coarse enough that the statistical estimate is reproducible
run-to-run, so the demo's byte-diff is stable. Both the Release and Debug builds
produce byte-identical stdout (verified).

Edge cases handled: ragged last thread block (`if (id >= M) return`), divide-by-
zero guards in the bias normalization, `log(0)` guard in Box–Muller, and grid
clamping for walkers that wander past `[s_lo, s_hi]`.

---

## Where this sits in the real world

This is a **reduced-scope teaching version** (CLAUDE.md §13). A production
enhanced-sampling study differs in every dimension of scale, not in concept:

| Aspect | This project | Production (PLUMED + GROMACS, etc.) |
|---|---|---|
| System | 1 particle, 1 analytic CV | 10⁴–10⁶ atoms; CVs computed from atomic coordinates on the fly |
| FES | known double well (so we can verify) | unknown — the quantity being measured |
| Dynamics | 1-D Langevin, our own integrator | full force-field MD on the GPU (NVIDIA-accelerated GROMACS/OpenMM/NAMD) |
| Bias | per-walker grid, no sharing | multi-walker shared bias via MPI; PLUMED computes bias on CPU with negligible overhead |
| Methods | well-tempered MetaD | + funnel MetaD, HREX, T-REMD, REST2, infrequent MetaD for rates |
| Reweighting | direct `F = −(γ/(γ−1))V` | block-averaging, time-independent free-energy estimators, error bars |

**Replica exchange (not coded here):** run `R` replicas at temperatures
`T_1 < … < T_R`. Each step, propose swapping configurations of adjacent replicas
`i, i+1` and accept with Metropolis probability
`min(1, exp((1/kT_i − 1/kT_{i+1})(U_i − U_{i+1})))`. Hot replicas cross barriers;
swaps funnel barrier-crossed structures down to the cold replica of interest.
**HREX/REST2** scale only the solute Hamiltonian, which keeps the needed replica
count modest. On a GPU each replica is — again — one trajectory in an ensemble,
so the same thread-per-replica mapping applies; the swap step adds a small
reduction over replica energies.

**Tools to study** (catalog "Starter repos"):
[PLUMED](https://github.com/plumed/plumed2) (the CV + bias engine that plugs into
every major MD code), [GROMACS](https://github.com/gromacs/gromacs) (GPU MD),
[OpenPathSampling](https://github.com/openpathsampling/openpathsampling)
(transition-path sampling), and
[HTMD](https://github.com/Acellera/htmd) (adaptive sampling on GPU clusters).
Benchmark FES test systems: **alanine dipeptide** and **chignolin**, with
published inputs on [PLUMED-NEST](https://www.plumed-nest.org).

---

### See also

- [`README.md`](README.md) — quick tour, build, run, exercises.
- [`src/metad.h`](src/metad.h) — the shared host+device physics core (start here).
- [`docs/PATTERNS.md`](../../../docs/PATTERNS.md) §1 (ensemble), §2 (HD core),
  §3 (determinism), §4 (tolerance) — the idioms this project follows.
