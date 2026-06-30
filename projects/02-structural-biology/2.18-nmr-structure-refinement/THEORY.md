# THEORY — 2.18 NMR Structure Refinement

> The deep didactic explanation (the "why"). Written for a sharp student who knows
> C++ but is new to CUDA and new to this domain. See [README.md](README.md) for the
> quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

### What an NMR experiment actually measures

X-ray crystallography gives you an electron-density map you can trace into atoms.
Solution **NMR is different**: you never see the structure. You see a spectrum of
peaks, and from it you extract *relationships between atoms*:

- **NOE cross-peaks (the dominant restraint).** The Nuclear Overhauser Effect
  couples two protons through space, and the cross-peak intensity falls off as
  **1/r⁶**. A visible cross-peak therefore means "these two protons are close" —
  in practice closer than ~5–6 Å. Crucially the 1/r⁶ dependence and spin-diffusion
  make the *exact* distance unreliable, so an NOE is treated as an **upper bound**,
  not an equality: "atoms i and j are no farther apart than `upper`."
- **Dihedral restraints** from J-couplings (Karplus equation) — bounds on backbone
  torsion angles φ/ψ.
- **RDCs** (residual dipolar couplings) — orientational restraints relative to a
  molecular alignment frame.

On top of the experimental data you always have the **covalent geometry**: bond
lengths, bond angles, and the chain connectivity are known chemistry and must hold.

### From restraints to a structure

A typical protein yields thousands of NOE restraints — but they are **sparse**
(only close-in-space proton pairs) and **noisy** (loose upper bounds, occasional
mis-assignments). No closed-form inversion exists. Instead you treat refinement as
**optimisation**: define a pseudo-energy that is zero when every restraint is
satisfied and positive when restraints are violated, then search coordinate space
for a low-energy structure. The standard search is **simulated annealing**.

Because the data underdetermines the structure, a *single* annealing run is not
trusted. You run an **ensemble** of independent runs from different random starts
and keep the lowest-energy members; the spread of that ensemble *is* the reported
uncertainty of the NMR structure (this is why PDB NMR entries contain ~20 models).

### What this project models (reduced scope)

We keep the essential idea and strip the rest to stay teachable:

- The molecule is a chain of **N Cα beads** (one per residue) in 3-D — a Cα trace,
  not all atoms.
- Restraints are **NOE upper bounds** between bead pairs plus **harmonic bonds**
  between consecutive beads (ideal Cα–Cα ≈ 3.8 Å). Dihedral/RDC terms are omitted
  (discussed in §7).
- The search is **Cartesian Metropolis Monte-Carlo annealing**, not MD. The
  *ensemble* structure — many independent trajectories — is preserved exactly, and
  that is what we parallelise.

---

## 2. The math

### State and restraint energy

A structure is the coordinate vector **x** = (x₀, y₀, z₀, …, x_{N−1}, y_{N−1},
z_{N−1}) ∈ ℝ³ᴺ. The bead–bead distance is

&nbsp;&nbsp;&nbsp;&nbsp; r_{ab}(**x**) = ‖ **p**_a − **p**_b ‖₂, &nbsp; where **p**_a = (x_a, y_a, z_a).

The **NOE flat-bottom penalty** for a restraint (i, j, u) — *u* the upper bound — is

&nbsp;&nbsp;&nbsp;&nbsp; E_NOE(r; u) = 0 &nbsp; if r ≤ u, &nbsp;&nbsp; ½·k_NOE·(r − u)² &nbsp; if r > u.

"Flat-bottom" means **no reward for being closer than u** (the NOE only bounds from
above), and a smooth quadratic wall once the bound is exceeded. The **bond restraint**
between consecutive beads is an ordinary harmonic spring about the ideal length b:

&nbsp;&nbsp;&nbsp;&nbsp; E_bond(r) = ½·k_bond·(r − b)².

The **total pseudo-energy** the annealer minimises is the sum over all bonds and all
NOE restraints:

&nbsp;&nbsp;&nbsp;&nbsp; **E(x)** = Σ_{b=0}^{N−2} E_bond(r_{b,b+1}) + Σ_{(i,j,u)∈NOE} E_NOE(r_{ij}; u).

These are exactly the terms in `total_energy()` (`src/nmr_refine.h`).

### Simulated annealing as a Markov chain

SA samples structures from the Boltzmann distribution **P(x) ∝ exp(−E(x)/T)** and
slowly lowers the temperature T. The Metropolis–Hastings transition is:

1. Propose **x′** from **x** by displacing one bead by a symmetric Gaussian step.
2. Compute ΔE = E(**x′**) − E(**x**).
3. Accept with probability **A = min(1, exp(−ΔE/T))**; else keep **x**.

At high T, A ≈ 1 even for uphill moves → the chain roams freely and escapes local
minima. As T → 0, A → 0 for any uphill move → the chain descends into the nearest
basin and freezes. A slow **geometric cooling**

&nbsp;&nbsp;&nbsp;&nbsp; T(s) = T_hot · (T_cold / T_hot)^{s/(S−1)}, &nbsp; s = 0 … S−1,

interpolates smoothly from exploration to exploitation. This is the textbook
Kirkpatrick (1983) annealing schedule.

---

## 3. The algorithm (and complexity)

Pseudo-code for **one replica** (mirrors `anneal_one` in `src/nmr_refine.h`):

```
seed RNG from (base_seed, replica_index)
x <- random self-avoiding-ish chain (hops of ~bond_len)
E <- total_energy(x);  best <- (E, x)
for s in 0 .. S-1:
    T  <- T_hot * (T_cold/T_hot)^(s/(S-1))     # geometric cooling
    b  <- random bead;  save its old position
    move bead b by a Gaussian displacement (sigma)
    E' <- total_energy(x);  dE <- E' - E
    xi <- uniform()                            # drawn UNCONDITIONALLY (see §5)
    if dE <= 0 or xi < exp(-dE/T):  E <- E'; if E<best.E: best <- (E,x)
    else: restore bead b
report best.E, #restraints satisfied at best.x, #accepted moves
```

**Complexity.** Each step recomputes the full energy: `O(N + R)` for N beads and R
restraints. One replica is `O(S·(N+R))`. The ensemble of M replicas is

- **Serial (CPU):** `O(M · S · (N + R))` — one trajectory after another.
- **Parallel (GPU):** the M trajectories run concurrently; wall-clock ≈
  `O(⌈M / P⌉ · S · (N + R))` for P resident threads. The *work* is identical; the
  GPU just does M of them at once.

A standard optimisation (left as Exercise 3) computes ΔE **locally**: moving one
bead changes only the `O(degree)` terms touching it, dropping the per-step cost from
`O(N+R)` to `O(degree)`. We deliberately recompute the full energy for clarity —
correctness-you-can-read beats speed-you-can't-explain.

### Honest timing

On the committed sample the GPU is ~2× the CPU (512 replicas). That ratio is a
**teaching artifact, not a benchmark** (CLAUDE.md §12). The per-thread loop is
register/local-memory heavy and somewhat divergent (the accept/reject branch
differs per replica), so a small ensemble does not saturate the device. The GPU's
edge **grows with the replica count** — push `--replicas 4096+` and the gap widens,
which is exactly why real ensemble NMR scales on GPUs.

---

## 4. The GPU mapping

```
        replica 0   replica 1   replica 2          replica M-1
          (thread)    (thread)    (thread)   ...     (thread)
   block:  t0 t1 t2 ... t127 | t0 ... t127 | ...   (128 threads/block)
            |                                         |
            v   each thread runs the FULL anneal_one() loop in local memory
        [x[3N], xbest[3N]]  <- per-thread scratch (no sharing, no atomics)
            |
            v
        out[replica] = ReplicaResult{ best_energy, n_satisfied, accepted }
```

- **Thread-to-data map:** `replica r = blockIdx.x * blockDim.x + threadIdx.x`. Thread
  *r* owns replica *r* and writes `out[r]`. A guard `if (r >= M) return;` handles the
  ragged final block.
- **Memory hierarchy.** Each thread keeps its current and best coordinates in
  **per-thread local memory** (`double x[3*NMR_MAX_BEADS]`,
  `double xbest[3*NMR_MAX_BEADS]`). For N ≤ 64 that is 2·3·64·8 = 3 KB/thread —
  comfortable. No **shared memory** is needed (threads never cooperate) and no
  **atomics** (no shared accumulator). The whole `RefineConfig`, including the
  restraint list, is passed **by value** so it lives in the kernel's constant
  parameter bank and is broadcast-read by every thread with zero extra `cudaMalloc`.
- **Why this maps so cleanly.** The ensemble is *embarrassingly parallel*: replicas
  are independent by construction, so there is literally nothing to synchronise. This
  is the same decomposition as the 9.02 SEIR / 13.02 PBPK ensembles (PATTERNS.md §1);
  the only difference is that the per-thread body is a Monte-Carlo annealer rather
  than an RK4 integrator.
- **Occupancy & divergence.** We use 128 threads/block: the body is long and
  local-memory-heavy, so a smaller block keeps register/local pressure reasonable
  while still giving the scheduler several warps to hide latency. Branch divergence
  is *mild and inherent* — every replica runs the same `S` steps; only the
  accept/reject branch differs, which is the nature of MC.

---

## 5. Numerical considerations

- **Precision: FP64 throughout.** Energies and coordinates are `double`. The
  flat-bottom kink at r = u and the `exp(−ΔE/T)` factor are smooth enough that double
  precision is ample; FP32 would risk visibly different accept/reject decisions.
- **Determinism via a shared integer RNG.** Reproducible stdout requires that a given
  replica make the *exact same sequence of moves* on CPU and GPU. We therefore use
  **splitmix64**, a tiny integer-hash RNG with no tables and no platform floating
  point, seeded from `(base_seed, replica)`. cuRAND would be faster and statistically
  superior but is **not bit-identical** to a host RNG — which would defeat the
  CPU-vs-GPU verification. This is a deliberate teaching trade-off.
- **Identical RNG call counts.** The Metropolis test draws its uniform
  **unconditionally** — even when ΔE ≤ 0 and the draw is unused — so the host and
  device consume the same number of randoms per step and their streams never drift.
  Box–Muller likewise recomputes both deviates each call rather than caching the
  second, for the same reason.
- **FMA and float associativity.** The host compiler and nvcc may contract
  multiply-adds differently, which could nudge an energy by ~1 ULP. In principle a
  1-ULP difference in ΔE could flip a borderline accept/reject and send the two
  trajectories down different paths. In practice, with FP64 and `FastMath` **off**
  (set in the `.vcxproj`), this does not happen on the sample: the discrete counts
  match **exactly** and the best energies differ by ~`1e-14`. We still verify the
  continuous energy only to a documented physical tolerance — the honest position
  (PATTERNS.md §4).
- **No atomics ⇒ no float-summation nondeterminism.** Because each replica owns its
  own state and there is no parallel reduction inside the kernel, there is no
  order-dependent floating-point sum to worry about (contrast the 5.01 / 11.09
  flagships, which must accumulate in integers).

---

## 6. How we verify correctness

The GPU and CPU run the **same** `anneal_one()` from `src/nmr_refine.h`, so for each
replica they should produce the same trajectory. `main.cu` checks this two ways:

1. **Discrete metrics — exact.** The number of satisfied restraints
   (`n_satisfied`, with a 0.02 Å validation slack) and the number of accepted moves
   (`accepted`) are **integers**; a last-bit floating-point wobble cannot change an
   integer, so we require them to match **bit-for-bit** across all replicas. This is
   the strong check: if any accept/reject decision had diverged, `accepted` would
   differ.
2. **Continuous metric — tight tolerance.** The per-replica best energy is compared
   with `ENERGY_TOL = 1e-4`. The observed worst difference is ~`1e-14` (round-off),
   far below any structural meaning.

A **second, science-level** check is built into the synthetic data (PATTERNS.md §6):
the restraints are generated from a *known* α-helix, so a satisfying structure
provably exists. The demo confirms the ensemble actually finds it — the best replica
reaches near-zero energy with **19/19** restraints satisfied, and a few hundred of
the 512 replicas satisfy them all. That validates the *annealer*, not just CPU==GPU
agreement.

**Edge cases handled:** ragged final thread block (guarded); divide-by-zero in the
random initial direction (ε added); `log(0)` in Box–Muller (sample from (0,1]);
empty NOE list (loop simply skips); restraint indices validated at load time.

---

## 7. Where this sits in the real world

| Aspect | This teaching project | Production (XPLOR-NIH, CYANA, AMBER `pmemd.cuda`, ARIA) |
|---|---|---|
| Representation | Cα beads only | All atoms (or all heavy atoms + H) |
| Energy | flat-bottom NOE + harmonic bonds | full force field: bonds, angles, dihedrals, vdW, electrostatics + NOE/dihedral/**RDC** terms |
| Search | Cartesian Metropolis MC | restrained **MD** (Cartesian, XPLOR/AMBER) or **torsion-angle dynamics** (CYANA) |
| Restraint types | NOE upper bounds | NOE (with 1/r⁶ averaging), J-coupling dihedrals, RDCs, H-bonds, chemical-shift back-calculation |
| Ensemble | M independent replicas, one GPU thread each | hundreds of replicas via MPI+CUDA; each replica is a full GPU-MD run |
| Validation | satisfied-restraint count | restraint-violation statistics, Ramachandran, WhatCheck/PROCHECK, chemical-shift RMSD (ShiftX2) |
| RNG | shared splitmix64 (for bit-exact verify) | cuRAND / MD thermostat; validated statistically |

The catalog's framing — *"GPU-accelerated CYANA/XPLOR-NIH can run hundreds of
independent SA trajectories simultaneously — essential for ensemble NMR structure
determination"* — is exactly the pattern demonstrated here, minus the force-field and
restraint richness. The parallel decomposition (one independent trajectory per
thread/process) is **identical** to production; what scales up is the per-replica
physics, not the ensemble strategy. The biggest real-world differences are (a) MD
with gradients instead of gradient-free MC, (b) torsion-angle space to cut the
degrees of freedom, and (c) chemical-shift / RDC back-calculation to validate the
result against orthogonal data.

---

## 8. Further reading

- Kirkpatrick, Gelatt, Vecchi (1983), *Optimization by Simulated Annealing*, Science.
- Nilges, Clore, Gronenborn (1988), *Determination of three-dimensional structures
  of proteins from interproton distance data by hybrid distance geometry–dynamical
  simulated annealing*.
- Schwieters, Kuszewski, Clore — **XPLOR-NIH** documentation (NOE/dihedral/RDC terms).
- Güntert — **CYANA** torsion-angle dynamics papers.
- See README "Prior art & further reading" for tool links.
