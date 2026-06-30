# THEORY — 2.34 Biophysical Simulation of Biomolecular Condensates (Active Learning Loop)

> Catalog ID 2.34 · Structural Biology & Protein Science · Difficulty: Advanced ·
> Maturity: Frontier/Theoretical. This document is the deep didactic dive behind a
> **reduced-scope teaching version** of the project (CLAUDE.md §13). Read the
> code in this order: `src/condensate.h` (the physics + integrator) →
> `src/reference_cpu.cpp` (baseline + the active-learning step) →
> `src/kernels.cu` (the GPU twin) → `src/main.cu` (the driver).

---

## 0. What we built vs. the frontier project

The catalog entry describes a research-grade **active-learning loop**: GPU
coarse-grained molecular dynamics (CG-MD) simulates many candidate disordered-
protein (IDP) sequences; a graph neural network (GNN) learns a surrogate map
from sequence to condensate properties; and Bayesian optimization (e.g. BoTorch)
proposes the next sequence to simulate. That full pipeline is several papers'
worth of machinery and is not a single teachable CUDA kernel.

So we ship the **load-bearing GPU compute pattern** faithfully and reduce the
machine-learning scaffolding to a deterministic stand-in:

| Frontier component | This teaching version |
|---|---|
| GPU CG-MD per sequence (CALVADOS-style) | GPU Brownian-dynamics of a short bead-spring chain, one thread per replica |
| Sequence → property | sequence reduced to one scalar "stickiness" `lambda`; properties = Rg and internal mobility `D` |
| GNN surrogate of property landscape | the just-measured ensemble *is* the surrogate (zero-variance at sampled points) |
| Bayesian-optimization acquisition | deterministic argmin of `|D − target_D|` over the swept members |
| GPU MSD → diffusion coefficient | per-thread lag-MSD in the COM frame → Einstein relation |

Everything simplified is called out below under **§7 Where this sits in the real
world**.

---

## 1. The science

**Biomolecular condensates** are membraneless compartments — nucleoli, stress
granules, P-bodies — that form when proteins and RNA undergo **liquid–liquid
phase separation (LLPS)**: above a concentration threshold the solution splits
into a dense droplet phase and a dilute phase. The proteins that drive LLPS are
often **intrinsically disordered** (IDPs / IDRs): they have no fixed fold, so
their behaviour is governed by *sequence-averaged* chemistry — the density of
"sticky" residues (aromatics, charges) that mediate weak, multivalent,
transient contacts.

Two measurable properties matter for whether a condensate is functional or
pathological:

- **Compactness** — how tightly a chain folds inside the dense phase, captured by
  the **radius of gyration** `Rg`. Stickier sequences collapse more.
- **Internal mobility** — how freely molecules rearrange inside the droplet,
  captured by a **diffusion coefficient** `D`. Liquid-like droplets have high
  `D`; aged, gel-like, disease-associated condensates (FUS, TDP-43, hnRNPA1 in
  ALS/FTD) have low `D`.

Designing or understanding condensates means learning the map *sequence →
(Rg, D)* and then searching sequence space — exactly an **active-learning loop**.

---

## 2. The math

### 2.1 The coarse-grained model

We represent one IDP as a chain of `n` beads at positions `r_i ∈ ℝ³`. The energy
has two parts:

- **Chain connectivity** — a harmonic bond between consecutive beads. We use a
  per-axis spring with rest offset `r0` along x (the initial layout) and `0`
  along y, z:
  `U_bond = ½ k_bond Σ_i [ (x_i−x_{i−1} − r0)² + (y_i−y_{i−1})² + (z_i−z_{i−1})² ]`.
- **Cohesion** — a soft well pulling each bead toward the chain centre of mass,
  with a stiffness that scales with the sequence stickiness `lambda`:
  `U_coh = ½ (k_cohese · lambda) Σ_i |r_i − r_com|²`.

The cohesive force on bead `i` is `F_coh,i = −(k_cohese·lambda)(r_i − r_com)`.
Because `Σ_i (r_i − r_com) = 0`, the cohesive force exerts **zero net force on
the centre of mass** — a fact that drives the choice of observable in §2.3.

### 2.2 Overdamped Langevin (Brownian) dynamics

Inside a crowded condensate, inertia is negligible: motion is **overdamped**.
The equation of motion for each bead is the Langevin equation without the
acceleration term:

```
gamma dr/dt = F(r) + sqrt(2 gamma kT) xi(t),   <xi>=0, <xi(t)xi(t')>=delta(t-t')
```

Discretized with the **Euler–Maruyama** scheme at timestep `dt`:

```
r_{t+1} = r_t + (dt/gamma) F(r_t) + sqrt(2 kT dt / gamma) * N(0,1)
```

The first term is deterministic drift down the energy gradient; the second is the
thermal kick, a Gaussian whose **variance grows with kT** (the fluctuation–
dissipation theorem). All quantities are in reduced MD units (σ length, τ time,
kT energy), the standard non-dimensionalization that keeps the integrator
well-scaled.

### 2.3 Observables

- **Radius of gyration** (compactness), averaged over production steps:
  `Rg = sqrt( (1/n) Σ_i |r_i − r_com|² )`.
- **Internal mobility** `D`. The chain's *centre of mass* diffuses **freely**
  regardless of `lambda` (the cohesion has no net COM force, §2.1), so COM
  diffusion would not discriminate sequences. What `lambda` controls is how far a
  bead wanders **relative to the COM** before the cohesive well pulls it back. We
  therefore measure the internal mean-square displacement at a fixed lag
  `tau = lag·dt`, in the COM frame `r̃_i = r_i − r_com`:

  ```
  MSD_int(tau) = < | r̃_i(t+tau) - r̃_i(t) |² >   (averaged over beads i and origins t)
  ```

  and apply the 3-D **Einstein relation** `MSD = 6 D tau` ⇒ `D = MSD_int(tau)/(6 tau)`.
  For a bead confined in a harmonic well of stiffness `k = k_cohese·lambda`,
  `MSD_int(tau)` saturates to a plateau `≈ 2σ²(1 − e^{−tau/τ_relax})` with
  `σ² = kT/k`. Stiffer (stickier) wells ⇒ smaller plateau ⇒ smaller apparent `D`.

### 2.4 The active-learning step

Given the ensemble's measured `{(lambda_m, D_m)}`, the loop proposes the next
sequence. The acquisition is `a(m) = |D_m − target_D|` and the proposal is the
**argmin** — the candidate whose mobility is closest to the experimental target.
(A full Bayesian-optimization acquisition would add an uncertainty term from the
GNN surrogate, e.g. Upper Confidence Bound `mean − kappa·sigma`; with the
ensemble itself as the surrogate the predictive variance is ~0 at sampled points,
so the score reduces to exploitation only.)

---

## 3. The algorithm (and its complexity)

```
for each ensemble member m (candidate sequence, in parallel):
    lambda  <- cohesive_lambda(m)              # m's stickiness on the sweep grid
    r       <- straight chain along x          # deterministic initial condition
    ring    <- empty circular buffer (lag+1 COM-frame snapshots)
    for s in 0 .. steps-1:                      # SEQUENTIAL in time
        com  <- mean(r)
        for each bead i:                        # forces + thermal kick
            F   <- bond_force(i) + cohesive_force(i, com, lambda)
            xi  <- gaussian_noise(m, s, i)      # counter-based, reproducible
            r'_i <- r_i + (dt/gamma) F + sqrt(2 kT dt / gamma) xi
        r <- r'                                 # simultaneous update
        if s >= eq_steps:                       # production phase
            accumulate Rg
            if >= lag snapshots stored: accumulate MSD_int(lag) vs snapshot lag-ago
            push COM-frame snapshot into ring
    D  <- (mean MSD_int) / (6 lag dt)           # Einstein relation
    Rg <- mean Rg
# host reduction:
propose m* = argmin_m | D_m - target_D |
```

- **Per replica:** `O(steps · n)` work; `O((lag+1)·n)` extra memory for the ring
  buffer.
- **Whole ensemble:** `O(n_members · steps · n)`. The member loop is fully
  parallel; the time loop inside a replica is inherently sequential. That shape —
  *parallel across members, sequential within a member* — is what makes "one
  thread per trajectory" the right GPU mapping.

---

## 4. The GPU mapping

This is the **ensemble-integration pattern** (PATTERNS.md §1, exemplified by
flagships 9.02 SEIR and 13.02 PBPK): *the same ODE/SDE solved for many parameter
sets*.

- **Thread-to-data map:** `idx = blockIdx.x*blockDim.x + threadIdx.x` owns
  ensemble member `idx`. One thread runs that candidate's **entire** Brownian-
  dynamics trajectory.
- **Memory hierarchy:** each thread keeps its bead coordinates, the scratch
  arrays for the simultaneous update, and the lag ring buffer in
  **registers / local memory** — there is no global-memory traffic during the
  trajectory and **no shared memory** (threads never cooperate). The only global
  writes are the `n_members` `ReplicaResult` outputs at the end. The kernel
  config travels **by value** as a small read-only argument.
- **No atomics:** members are independent, so nothing is reduced on the device;
  the active-learning argmin is a tiny host-side reduction.
- **Occupancy / block size:** 128 threads per block. Each thread is register- and
  local-memory-heavy (the ring buffer is `(lag+1)·n_beads·3` doubles), so a
  modest block keeps enough warps resident to hide local-memory latency without
  exhausting per-SM resources. The grid is `ceil(n_members / 128)` blocks; the
  ragged last block is guarded by `if (idx >= ensemble_size) return;`.
- **Divergence:** mild and benign — every thread runs the same step count; only
  the production-phase / ring-fill branches differ, and they are data-independent
  across threads at the same step.

```
grid:   [ block 0 ][ block 1 ] ... ceil(n_members/128) blocks
block:  128 threads
thread: idx --> member idx --> full trajectory --> out[idx] = {lambda, D, Rg}
```

---

## 5. Numerical considerations

- **Precision:** all dynamics and reductions are **double precision (FP64)**. The
  thermal kick and the cohesive drift differ by orders of magnitude, so FP32 would
  accumulate visible bias over hundreds of steps.
- **Deterministic, stateless RNG.** The thermal force on `(replica, step, bead,
  axis)` is a **counter-based** hash of those integer indices plus the seed
  (`gaussian_noise` in `condensate.h`, Box–Muller on two SplitMix-style hashes).
  There is **no evolving RNG state**, so:
  1. the draw is independent of thread-scheduling order — bit-reproducible on the
     GPU run to run; and
  2. GPU thread `m` draws *exactly* the numbers CPU iteration `m` draws — the CPU
     and GPU integrate the identical noise realization. This is what lets us
     verify to a tiny tolerance instead of a statistical one.
- **Simultaneous update.** We compute every bead's new position from the *old*
  positions (scratch arrays), then commit. An in-place update would let bead `i+1`
  read bead `i`'s already-advanced position — a different, order-dependent
  integrator. Simultaneous update keeps the math unambiguous and order-stable.
- **Stability.** The bond stiffness `k_bond=80` with `dt=0.005` gives
  `k_bond·dt/gamma = 0.4 < 1`, comfortably inside the explicit-Euler stability
  limit for a harmonic spring, so trajectories never blow up.
- **Float summation / FMA.** Because both sides run identical operations and the
  per-thread reductions are short, CPU↔GPU agreement is essentially bit-exact
  (~`1e-15`); we still verify to `1e-6` to leave honest headroom for FMA
  scheduling differences on other GPUs (PATTERNS.md §4).

---

## 6. How we verify correctness

1. **CPU == GPU per replica.** `main.cu` runs `integrate_cpu` (serial) and
   `integrate_gpu` (one thread per member) — both call the *same*
   `integrate_replica` in `condensate.h` — and asserts the worst per-replica
   difference in `D` and `Rg` is `≤ 1e-6`. Observed: `~1.3e-15`.
2. **Physical sanity (the science, not just CPU==GPU).** The committed sample is
   engineered so the answer is interpretable:
   - `Rg` is **monotonically decreasing** in `lambda` (2.90 → 0.95) — the
     compaction trend a stickier sequence must show.
   - `D` trends downward in `lambda` — lower internal mobility for stickier
     sequences — with realistic thermal scatter.
   - the active-learning argmin recovers the **known interior optimum** `m12`
     (`lambda=4.413`), the member whose `D=0.16357` is closest to `target=0.165`.
3. **Determinism.** stdout is byte-identical across runs *and* across the Debug
   and Release builds (verified), so `demo/run_demo` can diff it against
   `demo/expected_output.txt`.

---

## 7. Where this sits in the real world

The teaching model is deliberately simple; here is what a production loop does
differently, and where to look:

- **Force field.** We use one scalar `lambda` and a harmonic cohesion well. Real
  residue-level models — **CALVADOS 2** (KULL Centre) — assign each of the 20
  amino acids its own "stickiness" `lambda_a` (from a hydropathy/Ashbaugh–Hatch
  scale) and use the Ashbaugh–Hatch potential plus Debye–Hückel electrostatics.
  Our scalar is the sequence-averaged abstraction of that per-residue table.
- **Phase behaviour.** Real studies run **slab simulations** to find the
  coexistence concentrations (dilute vs dense), and compute surface tension and
  viscosity — not just single-chain Rg and D. **LAMMPS GPU** / **OpenMM** drive
  these at scale.
- **Surrogate + search.** The `|D − target|` argmin stands in for a **GNN
  surrogate** over sequence graphs plus **Bayesian optimization** (BoTorch) that
  models predictive *uncertainty* and balances exploration against exploitation —
  the real reason the loop is "active learning" and not a grid scan.
- **Integrator.** Production CG-MD uses Langevin/Brownian integrators with
  neighbour lists and periodic boundaries; we use a tiny single-chain
  Euler–Maruyama with no periodicity.
- **Data.** Sequences and disorder annotations come from **PhaSePro**, **DisProt**
  and the **PDB** (FUS/TDP-43/hnRNPA1 LC-domain structures); see `data/README.md`.

None of the numbers here are quantitative predictions for any real protein, and
nothing here is for clinical use.
