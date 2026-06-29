# THEORY — 1.25 Gaussian-Accelerated MD (GaMD)

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use. This is a **reduced-scope teaching
> version**: it implements the GaMD algorithm (boost + cumulant reweighting) on a
> 1-D model potential, not a full all-atom force field. See §7._

---

## 1. The science

**The problem: rare events.** A drug binds; a protein opens a cryptic pocket; a
loop flips between two conformations. These are the events that matter
pharmacologically, and they are **rare**: the system spends almost all its time
sitting in a free-energy minimum (a "state"), separated from other states by an
energy **barrier**. Ordinary molecular dynamics (MD) integrates Newton's equations
one ~2-femtosecond step at a time; to *see* a transition that happens once per
microsecond you must take ~10⁹ steps. That is the timescale problem that makes
straightforward MD too slow to study binding pathways or allostery.

**Enhanced sampling** speeds this up by biasing the simulation to cross barriers
more often, then mathematically *removing* the bias to recover the true,
unbiased free-energy landscape. The classic methods (umbrella sampling,
metadynamics, steered MD) require you to pick a **reaction coordinate** ahead of
time — "pull along *this* distance / *that* dihedral." Choosing wrong gives wrong
answers, and for an unknown mechanism you often *don't know* the right coordinate.

**Gaussian-accelerated MD (GaMD)** (Miao, Feher & McCammon, 2015) removes that
requirement. It watches the system's **total potential energy** `V` and, whenever
`V` dips into a deep well (`V < E`, a threshold), adds a smooth **boost potential**
`ΔV` that lifts the well toward `E` — flattening barriers *everywhere at once*,
with **no predefined reaction coordinate**. Because the boost is constructed to be
nearly **Gaussian-distributed**, the unbiased free energy can be recovered cheaply
by a **cumulant expansion** of `ΔV` (§2). GaMD ships in AMBER (`pmemd.cuda`),
NAMD, and OpenMM, and has been used to map GPCR drug-binding pathways and
allosteric mechanisms on practical timescales.

**What this project teaches.** We keep the *algorithm* — the boost potential, its
energy threshold and force constant, and the 2nd-order cumulant reweighting — and
apply it to the simplest system that has a barrier: a **1-D double well**, the
textbook model of a two-state conformational switch. We sample it with overdamped
Langevin dynamics (a minimal thermostat), boost it with GaMD, and show that
reweighting recovers the **known** double-well free-energy profile. Everything you
learn here is exactly what `pmemd.cuda` does — minus the 3N-dimensional force
field (§7).

## 2. The math

**The model potential.** A symmetric double well in one coordinate `x`:

```
U(x) = u_barrier · (x² − 1)²
```

with minima ("states") at `x = ±1` where `U = 0`, and a barrier at `x = 0` of
height `U(0) = u_barrier` (in units of the thermal energy `kT`). Its gradient is
`U'(x) = 4·u_barrier·x·(x²−1)`.

**The thermostat: overdamped Langevin (Brownian) dynamics.** The simplest dynamics
that samples the Boltzmann distribution `p(x) ∝ exp(−U(x)/kT)`:

```
x_{t+1} = x_t − (dt/γ)·U'(x_t) + √(2·dt·kT/γ)·ξ_t ,   ξ_t ~ N(0,1)
```

`γ` is friction, `dt` the timestep, `ξ_t` independent standard-normal noise. The
deterministic drift `−U'/γ` pulls downhill; the random kick supplies thermal
energy. At low temperature (barrier ≫ kT) a walker is **trapped** in one well for
a very long time — the rare-event problem in miniature.

**The GaMD boost.** GaMD adds a harmonic-in-energy boost wherever the potential is
below a threshold `E`:

```
ΔV(x) = ½·k·(E − U(x))²     if U(x) < E
        0                    otherwise
```

The force constant is `k = k0 / (V_max − V_min)` with a dimensionless knob
`0 < k0 ≤ 1`. This specific form is not arbitrary: Miao et al. derive it from two
requirements — (i) the boosted surface must keep the **same ordering** of states
(the boost may not create new minima or reorder old ones), and (ii) the
distribution of `ΔV` should be near-**Gaussian** so the reweighting below
converges. Condition (i) is what bounds `k` by the energy span; `k0→1` is the most
aggressive boost still satisfying it. The **boosted potential** is
`V*(x) = U(x) + ΔV(x)`, and the **boosted force** (what actually moves the walker)
is, by the chain rule,

```
−dV*/dx = −(1 + k·(E − U))·U'(x)     for U < E
```

i.e. the real force, *scaled down* near the wells so barriers shrink.

**Reweighting: recover the true free energy.** The boosted run samples the
*biased* distribution `p*(x) ∝ exp(−V*/kT)`. The true (unbiased) probability is
recovered by reweighting each sample by `exp(+βΔV)` (β = 1/kT). The exact
free energy (potential of mean force, PMF) of a histogram bin `b` is

```
F(b) = −kT·[ ln p*(b) + ln ⟨ e^{βΔV} ⟩_b ]      (up to an additive constant)
```

where `p*(b)` is the boosted occupancy of bin `b` and `⟨·⟩_b` averages over the
samples that landed in `b`. Evaluating `⟨e^{βΔV}⟩` directly is noisy (it is
dominated by rare large-`ΔV` samples). GaMD's signature trick is the **cumulant
expansion** of its logarithm, truncated at **2nd order**:

```
ln ⟨ e^{βΔV} ⟩_b  ≈  β·⟨ΔV⟩_b  +  (β²/2)·( ⟨ΔV²⟩_b − ⟨ΔV⟩_b² )
                      └─ 1st ──┘    └──────── 2nd cumulant = variance ───────┘
```

This is exact **iff** `ΔV` is Gaussian in each bin (higher cumulants vanish) — the
reason GaMD is engineered to make `ΔV` near-Gaussian. So per bin we only need
three accumulators: the count `n_b`, the sum of `ΔV`, and the sum of `ΔV²`.

**Inputs:** the 15-field config (data/README.md). **Output:** `F(b)` over the bins,
shifted so `min F = 0`, compared against the analytic `U(x)`.

## 3. The algorithm

```
for each walker w (independent):                 # the ENSEMBLE
    x ← starting well (±1, alternating by w)
    for s in 0 .. steps-1:                        # sequential time loop
        F* ← boosted_force(x)                     # GaMD bias folded in
        ξ  ← deterministic N(0,1) from hash(seed,w,s)
        x  ← x + (dt/γ)·F* + √(2 dt kT/γ)·ξ       # Langevin step
        if s ≥ equil:                             # skip burn-in
            b  ← bin_of(x)
            ΔV ← boost_dV(U(x))
            tally[b] += (1, ΔV, ΔV²)              # fixed-point integer add
# post-processing (host):
for each bin b:  F(b) ← reweight via 2nd-order cumulant   # §2
shift so min F = 0
```

**Complexity.** With `W` walkers and `S` steps, the simulation is `O(W·S)` work:
each step is a handful of flops (one polynomial gradient, one RNG, one tally). The
reweighting post-pass is `O(B)` over `B` bins — negligible. The per-step work is
tiny and the loop is **memory-light** (a walker's whole state is a couple of
registers), so the computation is dominated by the sheer **number** of
walker-steps. That is exactly the regime where a GPU wins: thousands of
independent walkers in flight.

**Data-access pattern.** Within a walker there is a strict sequential dependency
(step `s+1` needs `x` from step `s`) — you cannot parallelize *within* a
trajectory. But walkers are **mutually independent**, so the parallelism is across
the ensemble. The only shared state is the histogram, touched by all walkers
(handled with atomics, §4–5).

## 4. The GPU mapping

**Pattern: ensemble of independent trajectories — one thread per walker**
(PATTERNS.md §1, exemplars `9.02` SEIR and `13.02` PBPK; plus the per-thread RNG of
`5.01` Monte-Carlo dose). Each GPU thread runs one walker's *entire* time loop in
registers and deposits into a shared histogram.

- **Thread-to-data map:** `idx = blockIdx.x·blockDim.x + threadIdx.x` owns walker
  `idx`. A guard `if (idx ≥ n_walkers) return;` handles the ragged last block.
- **Launch config:** `block = 128` threads (a warp multiple; enough warps to hide
  latency; modest register pressure for a register-heavy integrator so occupancy
  stays healthy on sm_75–sm_89). `grid = ceil(n_walkers / 128)`.
- **Memory hierarchy:**
  - **Registers / local:** the walker state (`x`, RNG counters, the running
    Langevin constants) lives entirely in registers — no global traffic per step.
  - **Global:** only two touches — the histogram deposits (atomic, §5) and the
    final-position write. The histogram is `3·n_bins` 64-bit integers = a few KB.
  - **No shared memory / no constant memory** needed here: there is no halo to
    stage and no broadcast-read constant (contrast `1.12`, which puts its query in
    constant memory). The config struct is passed **by value** into the kernel, so
    it rides in the kernel's parameter space — the same trick `9.02`/`13.02` use.
- **CUDA libraries:** **none.** We deliberately hand-roll a counter-based RNG
  (splitmix64) instead of calling **cuRAND** — not because cuRAND is a black box,
  but because we need the device and host to draw the **identical** stream for
  exact verification (§6). cuRAND would give a different (and not host-matching)
  sequence. Writing the RNG by hand is ~10 lines and makes the determinism
  story explicit (this is the "explain what the library does / what hand-rolling
  takes" rule, CLAUDE.md §6.1.6, applied by *choosing* to hand-roll).

```
                 ensemble of W walkers  →  W GPU threads
   grid:  [ block 0 ][ block 1 ] ... [ block ceil(W/128)-1 ]
   block: 128 threads, thread t = walker (blk*128 + t)

   thread:  x in registers ──step loop (S steps)──▶ atomicAdd into
                                                    global histogram
            ┌─────────────────────────────────────────────┐
   global:  │ count[0..B-1] │ S1[0..B-1] │ S2[0..B-1] │  (int64, fixed-point)
            └─────────────────────────────────────────────┘
                       ▲ all threads atomicAdd here
```

The per-walker physics (potential, boost, RNG, the step loop, the tally) lives in
**one** `__host__ __device__` header, [`src/gamd.h`](src/gamd.h), so the CPU
reference and this kernel run **byte-identical** math (PATTERNS.md §2).

## 5. Numerical considerations

- **Precision: FP64 (double) throughout.** The Langevin step, the boost, and the
  RNG are all double. Enhanced sampling and reweighting are sensitive to small
  energy differences, and `double` keeps the host and device arithmetic in lock-
  step. The cost is acceptable for a teaching ensemble.
- **The race condition: the shared histogram.** Many threads add into the same bin
  simultaneously, so the deposits **must** be atomic, or counts are lost.
- **Determinism — the key teaching point (PATTERNS.md §3 rule 2).** A *float*
  `atomicAdd` is **not** reproducible: floating-point addition is not associative,
  so the sum depends on the (nondeterministic) order threads happen to arrive — the
  GPU result would wobble run-to-run and never exactly match the serial CPU. We
  avoid this entirely by accumulating in **fixed-point integers**: multiply `ΔV`
  by `2²⁰` and add as `int64`. Integer addition *is* associative and commutative,
  so the tally is **order-independent** — bit-identical every run and bit-identical
  to the CPU. (We add signed `int64` through CUDA's `unsigned long long` atomic;
  two's-complement wraparound makes the bit pattern of the sum correct.) The RNG is
  also deterministic: each `ξ` is a pure hash of `(seed, walker, step)`, so there
  is no shared RNG state and no order dependence.
- **The cumulant truncation is an *approximation*, by design.** The 2nd-order
  cumulant is exact only if `ΔV` is Gaussian per bin. For a **gentle** boost it
  nearly is; for a **strong** boost (large `k0`) the boost distribution is broad
  and skewed, and the truncation **systematically overestimates** the barrier (we
  observe ~9.5 kT recovered for a true 5 kT barrier at `k0=1`, vs. ~3.1 for a true
  3 kT barrier at `k0=0.15`). This is a *real, taught* property of GaMD — the
  method trades reweighting accuracy for acceleration, which is why production GaMD
  keeps the boost mild and/or uses higher-order/replica corrections.

## 6. How we verify correctness

Two independent checks (PATTERNS.md §4):

1. **GPU == CPU, exactly (tolerance 0).** `src/reference_cpu.cpp` runs the *same*
   `run_walker()` loop serially with a plain integer `+=` adder. Because the RNG,
   the physics, and the fixed-point accumulators are all deterministic integers,
   the GPU's atomic tally must equal the CPU's tally **bit-for-bit**. `main.cu`
   asserts the worst per-slot difference is `0`. A nonzero difference would mean a
   genuine bug (a lost atomic, a host/device math divergence), so `0` is a
   meaningful gate, not a rubber stamp. This is *stronger* than the float
   tolerances used by long iterative solvers (`10.02`, `14.02`) — here we can
   demand exactness because we engineered the reduction to be integer.
2. **Recovered barrier ≈ known answer (a physical tolerance).** The whole point is
   the *science*, so we also check the reweighted PMF against the analytic double
   well: the recovered barrier height must land within ~0.6 kT (= 0.2·barrier) of
   the true value. With the committed gentle boost (`k0=0.15`) the recovery is
   ~3.26 kT for a 3.00 kT barrier — within band. We use a **physical** tolerance,
   not machine precision, and say why (finite sampling + the cumulant bias of §5).

**Edge cases handled:** ragged last thread block (guarded); walkers that wander
outside the histogram range (binned as "out", not tallied); never-visited bins
(left at +∞ and printed `n/a`); divide-by-zero in occupancy/union (guarded).

## 7. Where this sits in the real world

This teaching version differs from production GaMD in **scope, not in method**:

| Aspect | This project | Production GaMD (AMBER `pmemd.cuda`, NAMD, OpenMM) |
|---|---|---|
| System | 1-D double well | full all-atom force field (3N coordinates: bonds, angles, torsions, Lennard-Jones, PME electrostatics) |
| Dynamics | overdamped Langevin | velocity-Verlet + Langevin/Berendsen thermostat, periodic boundaries, constraints (SHAKE) |
| Boost target | total `U(x)` | **dual boost**: dihedral energy *and* total potential, separately |
| Boost stats | fixed `E`, `k0` from config | running mean/variance of `V` collected on-the-fly, with `σ0` thresholds bounding the boost's standard deviation |
| Reweighting | 2nd-order cumulant | cumulant to 2nd order **and** exponential-average / Maclaurin variants; PCA of boosted trajectories |
| Scale | 512 walkers × 8k steps | one (or replica-exchanged) trajectory of millions of atoms over µs, multi-GPU |
| Variants | — | **LiGaMD** (ligand-selective boost for binding ΔG), Pep-GaMD, GLOW |

The pieces this version *omits* but a full implementation adds: the force field and
neighbor lists (the bulk of the compute, itself a huge GPU topic), the **on-the-fly
boost-parameter estimation** (you don't know `E`/`σ` in advance; GaMD measures them
during an equilibration phase using memory-efficient running statistics — the
"variance threshold" in the catalog), the dual dihedral+total boost, and replica
parallelism across multiple GPUs. The boost form and the cumulant reweighting you
see here are **identical** to what those tools use; that is the transferable lesson.

---

## References

- **Miao, Feher & McCammon (2015)**, *J. Chem. Theory Comput.* 11, 3584 — the GaMD
  method paper: the boost form, the harmonic-boundary constraints on `k`, and the
  cumulant reweighting derived here in §2.
- **Miao, Sinko, Pierce, … McCammon (2014)**, *JCTC* 10, 2677 — the cumulant-
  expansion reweighting and why 2nd order works when the bias is near-Gaussian.
- **AMBER GaMD tutorials** — https://www.med.unc.edu/pharm/miaolab/resources/gamd/ :
  the reference `pmemd.cuda` workflow; map its `iE`, `sigma0P`, `sigma0D` keywords
  onto this project's `E` and `k0`.
- **MiaoLab GaMD analysis scripts** — https://github.com/MiaoLab20/GaMD :
  post-processing and reweighting (PyReweighting) — the production version of §2's
  reconstruction.
- **AMBER** (https://ambermd.org) / **NAMD**
  (https://www.ks.uiuc.edu/Research/namd/) — the two reference GPU GaMD engines;
  study how the boost is folded into the force evaluation.
- **Zwanzig (1954)** — the free-energy perturbation identity `ΔF = −kT ln⟨e^{−βΔV}⟩`
  that all reweighting (including GaMD's) descends from.
