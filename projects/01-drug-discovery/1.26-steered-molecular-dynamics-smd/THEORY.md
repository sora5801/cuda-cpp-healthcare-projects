# THEORY — 1.26 Steered Molecular Dynamics (SMD)

> Read this after the [README](README.md). It walks from the biophysics of pulling
> a ligand out of a pocket, to the stochastic equations we integrate, to
> Jarzynski's equality, to how the whole thing maps onto GPU threads — and finally
> how this teaching model differs from production SMD.
>
> _Educational only — not for clinical use._

---

## The science

When a drug binds a protein, it sits in a free-energy **well**: the bound state.
Unbinding means climbing out of that well and over any barriers between it and the
solvent. Two quantities matter to medicinal chemists:

- **Binding free energy ΔG** — how favorable binding is at equilibrium (related to
  the dissociation constant `K_d` via `ΔG = kT ln K_d`).
- **Residence time / k_off** — how *long* the drug stays bound (a kinetic, not
  thermodynamic, property, increasingly seen as a better efficacy predictor).

You cannot watch a real unbinding event in a plain MD simulation: it is a **rare
event** that might take milliseconds, while MD steps are femtoseconds. **Steered
Molecular Dynamics** forces the issue: attach a virtual spring to the ligand (or
to a chosen **reaction coordinate** ξ, e.g. the ligand–pocket distance) and pull,
just like an atomic-force-microscope tip yanking on a molecule. The pull drives
the system over the barrier in nanoseconds.

But pulling does work *irreversibly* — you dump energy into friction and heat — so
the measured work `W` over-estimates the true ΔG. The breakthrough is
**Jarzynski's equality** (1997): even though each pull is far from equilibrium, a
particular *average* over many pulls recovers the **equilibrium** free energy
exactly. That turns a non-equilibrium, GPU-friendly experiment (many independent
fast pulls) into an equilibrium answer.

This project models the *method* on a single reaction coordinate moving in a fixed
free-energy landscape. The biology is abstracted to one number, ξ, so the
algorithm and the statistics stand out cleanly.

---

## The math

### State and dynamics

The reaction coordinate `ξ(t)` (nm) evolves under **overdamped Langevin
dynamics** — the high-friction limit of Newtonian MD, where inertia is negligible
and position responds directly to force:

```
dξ/dt = (1/γ) · F_total(ξ, t) + sqrt(2 kT / γ) · η(t)
```

- `γ` — friction coefficient ((kJ/mol)·ps/nm²); models the drag of the solvent
  and the rest of the protein.
- `kT` — thermal energy (kB·T; 2.4943 kJ/mol at 300 K).
- `η(t)` — Gaussian white noise, `⟨η(t)η(t')⟩ = δ(t−t')`. Its amplitude
  `sqrt(2kT/γ)` is fixed by the **fluctuation–dissipation theorem**, so that with
  no pulling the coordinate samples the Boltzmann distribution `∝ exp(−U(ξ)/kT)`.

The total force is the landscape force plus the SMD spring:

```
F_total(ξ, t) = −dU/dξ  +  k · (center(t) − ξ)
center(t) = ξ0 + v·t                 (the dummy atom moves at constant velocity v)
```

`U(ξ)` is the **potential of mean force** (PMF) — the free-energy profile along ξ.
We use a tilted symmetric double well with minima at `xa` and `xb`:

```
U(ξ) = A · [ (ξ − xa)(ξ − xb) / (xb − xa) ]²  +  slope · ξ
```

The quartic term is a barrier between two wells (it vanishes at `ξ = xa` and
`ξ = xb`); `slope` tilts the whole landscape. Because the pull runs well-to-well
(`ξ0 = xa → ξ_end = xb`), the **true end-to-end free energy is exact and known**:

```
ΔG_true = U(ξ_end) − U(ξ0) = slope · (ξ_end − ξ0)
```

(with `slope = −12 kJ/mol/nm` over a 1 nm pull, `ΔG_true = −12 kJ/mol`). This is
the ground truth we test Jarzynski against.

### Work and Jarzynski's equality

The **external work** done by the moving spring along one pull is the spring force
integrated over the displacement of the *spring center* (the protocol parameter),
the Crooks/Jarzynski convention:

```
W = ∫ F_spring · d(center) = Σ_n  k·(center_n − ξ_n) · v · dt
```

Jarzynski's equality states, **exactly**, for any pulling speed:

```
⟨ exp(−W / kT) ⟩  =  exp(−ΔG / kT)        ⟹        ΔG = −kT · ln ⟨ exp(−W/kT) ⟩
```

where `⟨·⟩` averages over the distribution of `W` from infinitely many pulls
started in equilibrium. Two consequences we will *see* in the demo:

- **Jensen's inequality** gives `⟨W⟩ ≥ ΔG`: the naive mean work is biased toward
  larger values. The gap `⟨W⟩ − ΔG` is the **dissipated work** (≈ 10 kJ/mol here).
- The exponential average is dominated by the **rare low-work tail** — pulls that,
  by luck of the thermal noise, cost less than average. Sampling that tail is why
  you need many trajectories, and why the GPU matters.

---

## The algorithm

Per trajectory `i` (all in `smd_core.h::run_trajectory`):

```
seed an independent RNG stream from (base_seed, i)
ξ ← ξ0 ;  center ← ξ0 ;  W ← 0
repeat `steps` times:
    f_pmf    ← −dU/dξ                        # pmf_force()
    f_spring ← k·(center − ξ)
    W       += f_spring · v · dt             # work BEFORE the coordinate moves
    ξ       += (dt/γ)·(f_pmf + f_spring) + sqrt(2kT·dt/γ)·N(0,1)   # Euler–Maruyama
    center  += v · dt
return W
```

Then on the host, over the whole ensemble (`smd_core.h::jarzynski_dg`):

```
Wmin ← min_i W_i                              # shift for numerical stability
ΔG  ← Wmin − kT · ln( (1/N) Σ_i exp(−(W_i − Wmin)/kT) )
```

**Complexity.** One trajectory is `O(steps)` time, `O(1)` memory (the whole state
lives in a handful of scalars). The ensemble is `O(N · steps)` work and `O(N)`
output. Serial: a double loop. Parallel: the outer loop over `N` trajectories is
fully independent, so it collapses to **one GPU launch** of `N` threads, each
doing `O(steps)` work — wall time `O(steps)` given enough threads. The Jarzynski
reduction is `O(N)` and runs once on the host.

---

## The GPU mapping

**One thread = one trajectory.** Thread `i = blockIdx.x·blockDim.x + threadIdx.x`
runs the entire `run_trajectory(p, i)` and writes `work[i]`. This is the
ensemble-ODE pattern (flagship `9.02` SEIR, `13.02` PBPK) fused with a per-thread
reproducible RNG (`5.01` Monte-Carlo dose). See [docs/PATTERNS.md](../../../docs/PATTERNS.md) §1–§2.

- **Memory hierarchy.** The per-trajectory state (`ξ`, `center`, `W`, the 64-bit
  RNG state, the constant params) lives entirely in **registers** / local memory.
  The *only* global-memory traffic is the single `double` write `work[i]` at the
  end. There is **no shared memory and no atomics** — trajectories never
  communicate. That is the cleanest possible GPU workload: pure compute, minimal
  bandwidth.
- **Why no library RNG (cuRAND)?** We need the CPU reference and the GPU kernel to
  draw the *same* random numbers so their works match (for an exact-ish
  verification). cuRAND's device generators have no bit-identical host twin, so we
  use a tiny **splitmix64** counter-based stream that compiles identically under
  nvcc and the host compiler (the `SMD_HD` macro). Each trajectory seeds its own
  stream from `(base, i)`, so streams are independent yet reproducible.
- **Launch config.** `THREADS_PER_BLOCK = 128`, `blocks = ceil(N/128)`. 128 is a
  warp multiple that balances latency hiding against the register pressure of a
  state-heavy per-thread integrator (each thread holds the full Langevin state, so
  too-large blocks could cap occupancy). The last block is guarded (`if (i ≥ N)
  return;`).
- **Divergence.** Every thread runs the same `steps` iterations; the only
  data-dependent control flow is inside `log`/`cos`/`sqrt` in the Gaussian draw,
  so warps stay coherent. This is *not* like Monte-Carlo particle transport
  (`5.01`), where histories take different numbers of steps and diverge.
- **Occupancy vs. registers.** Because everything is in registers, register count
  is the occupancy limiter — a good Nsight Compute exercise (README Exercise 5).
- **Where streams would come in.** The catalog mentions CUDA *streams* for
  "multiple independent pulling trajectories". In full-atom SMD each trajectory is
  its own expensive MD simulation, so you overlap them across streams/GPUs; here a
  trajectory is cheap enough that one big launch already saturates the device, so
  we keep it to a single stream for clarity.

**Why the GPU wins here.** The naive `⟨W⟩` estimate barely needs samples, but the
*Jarzynski* estimate's accuracy is gated by how well you sample the low-work tail
— i.e. by `N`. The GPU lets you push `N` up by orders of magnitude in the same
wall time, which is exactly the lever this method needs.

```
ensemble of N independent pulls            one GPU launch, N threads
  traj 0:  ξ0 →→→→→→→→ ξ_end   W_0          thread 0  ─┐
  traj 1:  ξ0 →→→→→→→→ ξ_end   W_1          thread 1   ├─ each runs the full
  ...                                       ...        │   Langevin loop in
  traj N-1:ξ0 →→→→→→→→ ξ_end   W_{N-1}      thread N-1 ─┘   registers → work[i]
                                            host: ΔG = −kT·ln⟨exp(−W/kT)⟩
```

---

## Numerical considerations

- **Precision: FP64 throughout.** Work values are differences of similar-magnitude
  energies and then exponentiated; single precision would lose the tail. All state
  and accumulation is `double`.
- **The log-sum-exp shift.** `exp(−W/kT)` overflows/underflows for `W` far from 0.
  We subtract `Wmin` first so every exponential is in `(0, 1]`, then add `Wmin`
  back in the log — algebraically exact, numerically safe. The sum runs in a
  **fixed index order**, so the reduction is bit-reproducible run to run
  (PATTERNS.md §3).
- **Deterministic stdout.** Results go to **stdout** and are byte-identical every
  run (integer-seeded RNG + fixed-order FP reduction). Timings and the run-varying
  GPU/CPU residual go to **stderr** (shown by the demo, not diffed).
- **RNG draw count is fixed.** Box–Muller consumes exactly **two** uniforms per
  step on both CPU and GPU, so the two streams stay perfectly in lockstep — a
  prerequisite for the works to match.
- **Euler–Maruyama, not RK4.** Stochastic differential equations need
  noise-aware integrators; the naive Euler–Maruyama scheme here is `O(dt)` weak
  order, which is why `dt` is small. Higher-order SDE integrators exist but would
  obscure the teaching point.

---

## How we verify correctness

Two independent checks, both in `main.cu`:

1. **GPU == CPU per-trajectory work.** Both sides call the *same*
   `run_trajectory()` with the same `(seed, i)`, so they should produce the same
   `W_i`. They agree to **~1e-13 kJ/mol**, not exactly: over 25000 double-precision
   steps with `log`/`cos`/`sqrt`, device and host `libm`/FMA differ at the last
   bit and the difference accumulates. We verify to `1e-6` kJ/mol and **say so**,
   rather than claiming bit-exactness — the long-iterative-solver case in
   [PATTERNS.md](../../../docs/PATTERNS.md) §4 (cf. flagship `10.02`). The residue
   is ~14 orders of magnitude below the ~30 kJ/mol work scale, i.e. physically
   meaningless.
2. **Jarzynski recovers the known ΔG.** The PMF is engineered so the true
   end-to-end free energy is exactly `slope·(ξ_end−ξ0) = −12 kJ/mol`. The
   Jarzynski estimate from 8192 pulls lands within ~0.7 kJ/mol (tolerance 1.5
   kJ/mol). This is the **science** check: it validates that the work bookkeeping
   and the exponential average are right, not merely that CPU and GPU agree. The
   demo also prints the **naive ⟨W⟩** (biased) and the **dissipation** so the
   learner sees *why* Jarzynski is needed.

Edge cases handled: the loader rejects non-positive counts/dt/γ/kT/k and a
degenerate PMF (`xa == xb`); the RNG returns uniforms in `(0,1]` so `log` never
sees 0.

---

## Where this sits in the real world

- **Production SMD is full-atom MD.** NAMD, GROMACS, and OpenMM integrate Newton's
  equations for every atom (tens of thousands to millions) in an explicit force
  field with explicit solvent, on the GPU. The reaction coordinate is a
  center-of-mass distance, an RMSD, or a more elaborate collective variable
  (PLUMED). Our 1-D overdamped model collapses all of that into one coordinate
  plus a friction/noise term — faithful to the *method*, not the *system*.
- **The PMF is the unknown.** Here we *hand* the simulation the PMF so we have a
  ground truth. In practice the PMF is exactly what you are trying to learn; SMD +
  Jarzynski (or its cousins) is one way to estimate it.
- **Better estimators.** Raw Jarzynski is biased for finite samples and high
  dissipation. Production work uses the **Crooks fluctuation theorem** with
  bidirectional pulls and **BAR**, the **Hummer–Szabo** PMF reconstruction, or
  switches to **umbrella sampling / metadynamics** for slow degrees of freedom.
  `alchemlyb` implements several. README Exercises 2–3 add the second-cumulant
  correction and a Hummer–Szabo PMF.
- **Kinetics, not just thermodynamics.** Rupture-force and residence-time studies
  (ion channels, motor proteins, drug `k_off`) often use **constant-force** SMD
  and analyze first-passage times — a natural extension (README Exercise 4).
- **Scale.** Real campaigns run trajectories across many GPUs and streams; the
  GPU's role is identical to here — independent trajectories in parallel — just
  with each trajectory being a full MD run rather than a 25000-step scalar loop.

> **Not for clinical or design use.** This is a didactic model with synthetic
> parameters; nothing here estimates a real binding free energy.
