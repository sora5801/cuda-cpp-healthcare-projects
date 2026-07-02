# THEORY — 6.10 Systems-Biology ODE/SDE Network Solver

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

A **gene regulatory network (GRN)** is a set of genes whose protein products
switch each other on and off. Transcription factors bind promoters and either
activate or repress the production of other proteins, forming feedback loops.
These loops are the "wetware logic" of the cell: they make decisions, keep time,
and maintain homeostasis. Systems biology models them — together with signalling
cascades and metabolism — as systems of coupled nonlinear ordinary differential
equations (ODEs), or stochastic differential equations (SDEs) when molecule
counts are low enough that noise matters.

This project studies the canonical engineered GRN: the **repressilator** (Elowitz
& Leibler, *Nature* 2000). Three genes are wired in a ring so that each represses
the next:

```
        represses            represses            represses
 gene0  ---------->|  gene1  ---------->|  gene2  ---------->|  (back to gene0)
   ^                                                            |
   +------------------------------------------------------------+
```

A ring of an **odd** number of repressors is a negative-feedback loop that cannot
find a stable "all consistent" state. With enough cooperativity it therefore does
not settle — it **oscillates**, producing regular pulses of each protein out of
phase with its neighbours. Elowitz & Leibler built this circuit in living *E.
coli* and watched cells blink under a microscope: the first synthetic genetic
clock. The scientific question this code answers is: **for which parameter values
does the circuit oscillate, and how strong is the oscillation?** Answering it
requires solving the ODE not once but across a whole grid of parameters — the
GPU batch problem.

Real studies push this much further: uncertainty quantification (which parameters
does the behaviour depend on?), sensitivity analysis, and multi-cell simulations
where each of thousands of cells carries slightly different parameters. All of
them reduce to "solve this small ODE system a huge number of times."

## 2. The math

State (dimensionless, all quantities ≥ 0):

- `m_i` — concentration of mRNA for gene *i*, `i = 0,1,2`.
- `p_i` — concentration of the protein for gene *i*.

The flat state vector used in code is `s = [m0 m1 m2 p0 p1 p2]` (`STATE_DIM = 6`).

Governing ODEs (one common non-dimensionalisation, time measured in mRNA
lifetimes):

```
dm_i/dt = -m_i + alpha * f(p_{i-1}) + alpha0
dp_i/dt = -beta * (p_i - m_i)
```

with the ring index `i-1` taken mod 3 (so gene 0 is repressed by protein 2), and
the **Hill repression function**

```
f(p) = 1 / (1 + p^n).
```

Symbols (units, ranges):

| Symbol | Meaning | Units / range |
|---|---|---|
| `alpha` | max transcription rate (leak-subtracted) | in mRNA-decay units; swept `[10,260]` |
| `alpha0` | leaky/basal transcription (promoter floor) | mRNA-decay units; `1.0` here |
| `beta` | protein-decay / mRNA-decay rate ratio | dimensionless; `5.0` here |
| `n` | Hill coefficient (repression cooperativity) | dimensionless; swept `[1,3]` |
| `t` | time | mRNA lifetimes |

`f(p)` decreases from 1 (repressor absent → promoter fully ON) toward 0
(repressor abundant → promoter OFF); larger `n` makes the switch sharper. The
term `-m_i` is first-order mRNA decay; `-beta(p_i - m_i)` is protein production
proportional to mRNA minus first-order protein decay, written so `beta` sets the
relative protein/mRNA timescale.

**Objective (what we compute):** for each `(alpha, n)` in the sweep, integrate the
IVP from a fixed initial state and report, for the readout protein `p2`: its final
value, its min/max over the second half of the run, a level-crossing count, and a
boolean "oscillates".

**Why it oscillates (sketch).** The system has a single symmetric fixed point
`m_i = p_i = m*` solving `m* = alpha·f(m*) + alpha0`. Linearising the ring and
requiring the Jacobian to have eigenvalues cross into the right half-plane gives a
Hopf-bifurcation condition of the form (Elowitz & Leibler, supplementary):

```
(beta+1)^2 / beta  <  3 X^2 / (4 + 2X),   X = -alpha n f'(p*) p*/(alpha f + alpha0)
```

Intuitively: high cooperativity `n` and high `alpha` (steep, strong repression)
plus a moderate protein timescale `beta` destabilise the fixed point → sustained
oscillation. Exercise 2 in the README asks you to recover this boundary from the
computed `oscillates` grid.

## 3. The algorithm

For one member:

1. Set the 6-D state to the shared initial condition (`m0=1`, else 0 — an
   asymmetric seed so the symmetric fixed point is not a trap).
2. Advance with fixed-step **RK4** for `steps` steps of size `dt`. One RK4 step
   evaluates the RHS four times (`k1..k4`) and forms the weighted update
   `s += dt/6 (k1 + 2k2 + 2k3 + k4)`; global error is `O(dt^4)`.
3. Summarise the readout `p2` (see §6 for the two-pass, deterministic detector).

Complexity: per member, `O(steps · STATE_DIM)` work with a small constant (4 RHS
evals/step). The RHS is `O(STATE_DIM)` and includes one `pow()` per gene (the Hill
term) — the dominant cost. For `M = na·nn` members the **serial** cost is
`O(M · steps · STATE_DIM)`.

**Parallel structure:** the `M` members are independent (no data shared between
trajectories). So the *work* is unchanged but the *depth* collapses from
`O(M·steps)` to `O(steps)` — every member advances its own time loop in parallel.
This is the "same ODE for many parameter sets" row of `docs/PATTERNS.md §1`,
exemplified by flagships 9.02 (SEIR ensembles) and 13.02 (PBPK).

Arithmetic intensity is high and memory traffic is tiny: the entire per-member
working set (state + RK4 temporaries, a few dozen doubles) fits in registers, and
the only global-memory write is one `MemberResult` at the end. There is no
input array at all — the problem is described by a small config struct.

## 4. The GPU mapping

**Thread-to-data mapping.** One thread integrates one ensemble member:

```
idx = blockIdx.x * blockDim.x + threadIdx.x;   // flat member index
if (idx >= M) return;                          // guard the ragged last block
member_params(c, idx, pr);                     // decode (alpha, n) from idx
out[idx] = integrate_member(c.s0, pr, dt, steps);
```

`idx = a·nn + b` decodes to row `a` (an `alpha`) and column `b` (an `n`) of the
sweep grid.

**Launch configuration.** `block = 128` threads; `grid = ceil(M / 128)` blocks.
128 is a solid occupancy default on sm_75..sm_89: several warps per block to hide
latency, while the per-thread **register** footprint of double-precision RK4 (the
`k1..k4` and `tmp` arrays, `4·6 + 6 = 30` doubles plus scalars) stays low enough
to keep many blocks resident. If register pressure ever limits occupancy, options
are: shrink the block, use `-maxrregcount`, or spill the RK4 temporaries — all
discussed in `docs/BUILD_GUIDE.md`.

**Memory hierarchy.**
- **Registers/local:** the whole trajectory. This is why the kernel is fast — no
  global traffic in the hot loop.
- **Constant/parameter bank:** the `EnsembleConfig` is passed **by value**, so it
  rides in the kernel's constant parameter space, broadcast to every thread.
- **Global:** only the `out[M]` result array, one coalesced write per thread.
- **Shared memory / atomics:** none. Members never interact.

```
grid  ─────────────────────────────────────────────────────────────
 block0            block1                       block(⌈M/128⌉-1)
[t0 t1 ... t127] [t128 ... t255] ...            [... tM-1  (idle)]
   │  │      │
   │  │      └── member 127: RK4 loop in registers → out[127]
   │  └───────── member 1
   └──────────── member 0
```

**Why not "one block per system + shared-memory Jacobian + cuSPARSE" (the catalog
hint)?** That layout is for *large, stiff* systems where each step solves a linear
system with the Jacobian (implicit BDF/Newton). Then it pays to cooperate a whole
block on one system and store the Jacobian in shared memory / cuSPARSE. The
repressilator is tiny (6 states) and non-stiff at our `dt`, so an **explicit**
solver with **one thread per system** has less overhead and no divergence from a
per-system linear solve. THEORY §7 says when to switch.

## 5. Numerical considerations

- **Precision: FP64 throughout.** The Hopf boundary and the oscillation waveform
  are sensitive to accumulated error over thousands of steps; double precision
  keeps the CPU and GPU trajectories agreeing to ~`1e-13` (see §6). FP32 would
  drift enough to flip borderline members' classification.
- **The Hill `pow()` domain.** `p^n` is undefined for negative `p` with
  non-integer `n`. A tiny negative round-off excursion could produce a NaN, so
  `hill_repress` clamps `p` to ≥ 0 before `pow`.
- **No atomics, no reordering.** Because each thread owns a private trajectory and
  writes one independent result, there is **no parallel reduction** and hence no
  floating-point-reassociation nondeterminism (contrast Monte-Carlo tallies in
  5.01 / k-means in 11.09, which must accumulate in integers). Every thread's
  arithmetic is the same closed-form RK4 sequence the CPU runs.
- **The FMA subtlety (why the detector is two-pass).** An optimised `-O2` GPU/host
  build may contract `a*b+c` into a fused multiply-add, changing the last bit
  versus an unoptimised build. Over thousands of steps this makes a *flat* steady
  state wiggle at the `1e-14` level. A naïve "count crossings of the running mean"
  detector would count different numbers of spurious crossings in Debug vs
  Release, breaking the byte-identical-stdout contract. The two-pass hysteretic
  detector (§6) uses a deadband proportional to the observed amplitude, so a flat
  member scores **exactly 0** crossings on every build.

## 6. How we verify correctness

The CPU reference (`src/reference_cpu.cpp`) integrates every member in a plain
serial loop, calling the **same** `integrate_member()` from `src/grn.h` that the
GPU thread calls. Because the per-element physics lives in one
`__host__ __device__` header (the CPU/GPU-parity idiom, `docs/PATTERNS.md §2`),
both paths run identical arithmetic.

`main.cu` then checks, over all `M` members:

- **Continuous observables** (`p2_final`, `p2_min`, `p2_max`): worst absolute
  difference ≤ `TOLERANCE = 1e-9`. Both sides run the same double-precision RK4,
  so they agree to ~`1e-13`–`1e-14`; `1e-9` is a comfortable, honest margin for
  "same computation, FMA-level differences" (`PATTERNS.md §4`, the
  ~machine-precision case). The observed worst diff on the sample is `5.95e-14`.
- **The oscillation flag** (`oscillates`): must match **exactly**. It is an integer
  classification; any disagreement is a real bug, not round-off.

**The two-pass oscillation detector** (in `integrate_member`):

1. *Pass 1* integrates the trajectory and records `min`, `max`, and the final `p2`
   over the **second half** (after transients decay). From these, form a **fixed**
   reference level `mid = (min+max)/2`, amplitude `amp = max-min`, and a hysteresis
   deadband `band = 0.25·amp`.
2. *Pass 2* re-integrates and counts **upward** crossings of `mid` using a two-state
   hysteresis machine: the signal must drop below `mid-band` and then rise above
   `mid+band` to score one crossing (two crossings = one full cycle).

Using a *fixed* level + deadband (not a running mean) is what makes the count
deterministic across build configs. A member "oscillates" iff the relative swing
`amp/mid > 5%` **and** there are ≥ 2 crossings. A steady state has `amp ≈ 0`
(so `band ≈ 0`, the signal never travels it) → 0 crossings → not oscillating,
which is exactly what the demo shows for the `n = 1.0` members.

**A second, physical check** beyond CPU==GPU: the demo output is *scientifically*
correct — low cooperativity settles, high cooperativity oscillates — matching the
repressilator's known Hopf behaviour (PATTERNS.md §4 "compare against a known
result").

## 7. Where this sits in the real world

Production systems-biology solvers differ from this teaching version in three big
ways:

- **Arbitrary models via SBML.** Tools like **libRoadRunner** / **Tellurium** parse
  SBML (an XML standard for biochemical models) and **JIT-compile** the RHS with
  LLVM, so any of BioModels' 1000+ curated models runs without hand-writing
  `grn_deriv`. We hard-code one circuit to keep the focus on the GPU pattern.
- **Stiff, adaptive integration.** Metabolic and signalling models span timescales
  from milliseconds to hours (stiff). **SUNDIALS/CVODE** uses adaptive-order,
  adaptive-step **BDF** with an implicit Newton solve each step (needing the
  Jacobian and a linear solve — where the catalog's shared-memory/cuSPARSE layout
  and CVODE's CUDA NVector come in). Our fixed-step explicit RK4 is fine for the
  non-stiff repressilator but would be inefficient or unstable on a stiff model.
- **True stochasticity.** At low molecule counts, deterministic ODEs are wrong;
  one uses Gillespie SSA, tau-leaping, or the **Chemical Langevin Equation**
  (**GillesPy2**). We include a CLE step (`grn_cle_step`) for teaching but keep it
  off the verified path, because an SDE's random increments cannot be made
  bit-identical across independent CPU/GPU RNG streams.

The one thing this project shares with all of them is the load-bearing idea:
**the ensemble is embarrassingly parallel, so map one member to one thread.**

---

## References

- Elowitz M.B., Leibler S. (2000). *A synthetic oscillatory network of
  transcriptional regulators.* Nature 403:335–338. — the repressilator; source of
  the model and the Hopf condition.
- Hindmarsh et al., **SUNDIALS** (https://github.com/LLNL/sundials) — study the
  CUDA NVector and batch-CVODE for how a production batch-ODE solver is structured.
- **libRoadRunner** (https://github.com/sys-bio/roadrunner) — SBML → JIT RHS; the
  automated version of writing `grn_deriv` by hand.
- **Tellurium** (https://github.com/sys-bio/tellurium) — Antimony/roadrunner
  workflows; a fast way to prototype the repressilator and compare.
- **GillesPy2** (https://github.com/GillesPy2/GillesPy2) — SSA / tau-leaping / CLE;
  the reference for the SDE extension (README Exercise 3).
- Gillespie D.T. (2000). *The chemical Langevin equation.* J. Chem. Phys.
  113:297–306. — the SDE `grn_cle_step` approximates.
