# THEORY — 1.22 Constant-pH Molecular Dynamics

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use. This is a **reduced-scope** teaching
> model (CLAUDE.md §13); §7 explains what production CpHMD does instead._

---

## 1. The science

A protein is built from amino acids, and several of them carry **ionizable side
chains** — chemical groups that can hold or release a proton (H⁺) depending on the
acidity of the surroundings:

| Residue | Type | Protonated form | Deprotonated form | Textbook pKa |
|---|---|---|---|---|
| Aspartate (Asp) | acid | –COOH (neutral) | –COO⁻ (−1) | ~3.7–4.0 |
| Glutamate (Glu) | acid | –COOH (neutral) | –COO⁻ (−1) | ~4.2 |
| Cysteine (Cys) | acid | –SH (neutral) | –S⁻ (−1) | ~8.3 |
| Histidine (His) | base | imidazolium (+1) | imidazole (neutral) | ~6.0–6.5 |
| Lysine (Lys) | base | –NH₃⁺ (+1) | –NH₂ (neutral) | ~10.5 |

The **pKa** is the pH at which the group is half protonated. Below its pKa a group
tends to *hold* its proton; above it, to *release* it. Most molecular simulations
**fix** every protonation state at the start and never change it — which silently
assumes you already know each residue's charge. That assumption breaks exactly
where it matters most for drug design:

- A histidine in an enzyme active site may flip protonation as a ligand binds,
  switching a hydrogen bond on or off.
- An aspartate near a positively charged pocket may stay protonated far above its
  textbook pKa because the environment disfavors its negative form.
- Ion-channel gating and pH-sensitive drug binding are driven by these shifts.

The reason a residue's *real* pKa differs from the textbook value is **coupling**:
the other charged groups in the protein create an electric field that stabilizes
one protonation form over the other. **Constant-pH molecular dynamics (CpHMD)**
treats protonation as a dynamic variable, sampling conformation and protonation
*together* at a chosen pH, and so predicts these shifted pKa values.

**What this project computes.** For a small set of titratable residues at fixed
positions, we sample the coupled protonation equilibrium across a grid of pH
values and report (a) each residue's **titration curve** (fraction protonated vs
pH) and (b) its **pKa**, read off where the curve crosses 50%. The headline
teaching result is that the predicted pKa is **shifted** from the intrinsic input
by the inter-residue electrostatics — the core phenomenon CpHMD exists to capture.

---

## 2. The math

**State.** Each residue `i` has a binary protonation variable `s_i ∈ {0,1}`
(1 = protonated). A microstate of an `N`-residue system is the vector
`s = (s_0, …, s_{N−1})`, one of `2^N` configurations.

**Charge.** Residue `i` carries net charge
`q_i(s_i) = s_i · q_i^prot + (1 − s_i) · q_i^deprot` (units of `e`). For an acid
`(q^prot, q^deprot) = (0, −1)`; for a base `(+1, 0)`.

**Energy of a microstate at pH.** The constant-pH free energy of a microstate,
relative to a reference, splits into a per-residue **intrinsic** term and a
pairwise **coupling** term:

```
G(s; pH) =  Σ_i  s_i · [ −kT·ln(10)·(pKa_i − pH) ]            (intrinsic)
          + Σ_{i<j}  k · q_i(s_i) · q_j(s_j) / r_ij           (coupling)
```

Symbols (with units):

- `kT` — thermal energy `k_B·T` (kcal/mol; ≈ 0.593 at 298 K). Sets the scale of
  thermal fluctuations.
- `ln(10) ≈ 2.302585` — converts the base-10 pKa/pH difference into the natural-log
  units of the Boltzmann factor.
- `pKa_i` — the intrinsic (model, isolated) pKa of residue `i` (dimensionless).
- `pH` — the fixed acidity of this simulation (dimensionless).
- `k = 332.06 / ε` — the Coulomb prefactor (kcal·Å·mol⁻¹·e⁻²). `332.06` converts
  `e²/Å` to kcal/mol; `ε` is an effective dielectric (water ≈ 80, protein interior
  ≈ 4–20) that screens the interaction. Set `k = 0` to decouple the residues.
- `r_ij` — distance between residues `i` and `j` (Å); positions are fixed here.

The **intrinsic term** is just Henderson–Hasselbalch written as a free energy:
protonating residue `i` costs `ΔG = −kT·ln(10)·(pKa_i − pH)`, which is negative
(favorable) when `pH < pKa_i` and positive (costly) when `pH > pKa_i`.

**Equilibrium target.** At fixed pH the microstates follow the Boltzmann
distribution `P(s) ∝ exp(−G(s; pH)/kT)`. The observable we want is the
**fraction protonated** of each residue,

```
f_i(pH) = ⟨ s_i ⟩ = Σ_s s_i · P(s).
```

`f_i(pH)` is residue `i`'s titration curve; the **pKa** is the pH where
`f_i = 0.5`.

**Move energy (what the code actually evaluates).** A Monte Carlo step proposes
flipping one residue `i`, `s_i → s_i' = 1 − s_i`, holding all others fixed. The
energy change is local:

```
ΔG_i =  (intrinsic flip term, ± kT·ln(10)·(pKa_i − pH))
      +  Σ_{j ≠ i} k · Δq_i · q_j(s_j) / r_ij ,     Δq_i = q_i(s_i') − q_i(s_i).
```

This `O(N)` expression — not the full `O(N²)` energy — is all the Metropolis rule
needs (`src/cph_core.h::delta_G_flip`).

---

## 3. The algorithm

Because `2^N` enumeration is exponential, we sample with **Metropolis Monte
Carlo**, which draws microstates with the correct Boltzmann weights using only
local move energies.

**One chain** (one pH, one replica):

```
state ← all deprotonated                       # identical start on CPU and GPU
for sweep in 0 .. sweeps-1:
    for i in 0 .. N-1:                          # one sweep = attempt every residue
        ΔG ← delta_G_flip(state, i, pH)
        u  ← uniform(0,1)                        # drawn even for downhill moves*
        if ΔG ≤ 0 or u < exp(−ΔG/kT):           # Metropolis acceptance
            state[i] ← 1 − state[i]
    if sweep ≥ burn_in:                          # discard equilibration
        for i: prot_count[i] += state[i]         # integer tally
```

`*` We draw `u` even when the move is downhill so the RNG stream advances
identically on CPU and GPU regardless of branch — essential for the exact match.

**Ensemble.** Repeat the chain over a **pH grid** (`n_pH` points) and `replicas`
independent chains per pH (different seeds), then average. Fraction protonated:
`f_i(pH_k) = prot_count[k,i] / (replicas · (sweeps − burn_in))`.

**Reading the pKa.** Scan the curve for the bracket where `f_i` crosses 0.5 and
linearly interpolate the crossing pH (`src/reference_cpu.cpp::estimate_pKa`).

**Complexity.** One chain is `O(sweeps · N²)` (the `N` factor is the coupling sum
inside each of the `N` flips per sweep). The whole run is
`O(n_pH · replicas · sweeps · N²)`. Crucially the `n_pH · replicas` chains are
**completely independent** — no shared state, no communication — so the only thing
that scales with parallel hardware is launching more chains at once. That
independence is the entire reason this maps cleanly onto the GPU.

---

## 4. The GPU mapping

**One thread = one chain.** The ensemble has `M = n_pH · replicas` independent
chains. We launch a fixed grid and use a **grid-stride loop** so any `M` is
covered; global thread `g` decodes its work as

```
k = g / replicas      # which pH grid point
r = g % replicas      # which replica
pH = pH_min + (pH_max - pH_min) * k / (n_pH - 1)
```

Consecutive threads in a warp share a pH `k` and differ only by replica `r`, so
they run structurally similar chains — keeping warp divergence modest.

```
ensemble of chains (n_pH × replicas)            GPU grid (1-D, grid-stride)
   pH0:  r0 r1 r2 ... r7                         block 0   block 1   ...
   pH1:  r0 r1 r2 ... r7        ───►        [t0 t1 t2 ... t255][t0 ...]   each thread:
   ...                                        │   │                       seed RNG(seed, chain_id(k,r))
   pH14: r0 r1 r2 ... r7                       └─► run_chain() in registers
                                                   atomicAdd integer counts
                                            d_prot[n_pH × N]  ◄── atomic tally
```

**Memory hierarchy and why:**

- **Parameters in constant/parameter memory.** `CphSystem sys` is passed *by
  value*; the driver places it where every thread reads it cheaply (broadcast). It
  is read-only for the whole launch — ideal.
- **Chain state in registers / local memory.** `state[N]` and `chain_counts[N]`
  with `N ≤ 16` live in per-thread registers/local memory. A chain runs thousands
  of sweeps **without touching global memory at all** — the kernel is compute- and
  divergence-bound, not bandwidth-bound.
- **Integer tally in global memory via atomics.** Only at the end does each thread
  `atomicAdd` its integer per-residue counts into the shared `d_prot[n_pH·N]`
  array. The tally is tiny (tens to hundreds of `uint64`), so atomic contention is
  negligible.

**Launch config.** Block = 256 threads (multiple of the 32-lane warp, eight warps
to hide latency). Grid = 1024 blocks, far more than the chip's resident-warp
capacity, so the scheduler always has work to hide the per-chain divergence; the
grid-stride loop folds the rest of the ensemble onto those threads.

**No CUDA library is used.** The RNG is a hand-written splitmix64 (cph_core.h)
**on purpose**: a tiny, deterministic, host-reproducible generator is what makes
verification an *exact* integer match. The natural production choice is **cuRAND**
(see §7); §6 explains the trade-off. The energy and Metropolis test are a handful
of FLOPs per move — nothing a library would accelerate.

**The Monte-Carlo divergence lesson.** Within a warp, threads accept or reject
different moves, so they take different branches every step — the classic source of
warp divergence in GPU Monte Carlo. We accept it for clarity. Production codes
reduce it by batching many replicas of the *same* pH and processing structurally
identical work together, and by sorting/regrouping particles — the same idea as
material-sorting in Monte-Carlo dose transport (flagship `5.01`).

---

## 5. Numerical considerations

- **Precision: FP64 throughout.** The energies are small kcal/mol numbers and the
  `exp(−ΔG/kT)` factor is sensitive, so we use `double` for the RNG-to-uniform
  conversion, the energy, and the acceptance test. This costs throughput on
  consumer GPUs (FP64 is rate-limited) but the kernel is not FLOP-bound here, and
  double precision keeps CPU and GPU bit-identical.
- **Determinism is engineered, not assumed.** Two design choices make the result
  reproducible:
  1. **Shared RNG + shared energy** (`cph_core.h`, the `__host__ __device__`
     idiom) means each chain `(k,r)` draws the *same* random numbers and computes
     the *same* `ΔG` on CPU and GPU, so it makes the *same* accept/reject
     decisions — the chains are identical move-for-move.
  2. **Integer tallies, not float fractions.** We accumulate protonation *counts*
     (`uint64`), never a running float average. Integer `atomicAdd` is
     associative and commutative, so the parallel reduction is order-independent
     and equals the serial sum exactly. A float `atomicAdd` would **not** —
     floating-point addition is non-associative, so thread-ordering noise would
     break reproducibility (PATTERNS.md §3). We convert to the fraction only at
     the very end, identically on both sides.
- **Avoiding `exp` overflow.** Uphill moves with large `ΔG` give a tiny acceptance
  probability; `exp(−ΔG/kT)` underflows harmlessly to 0 (rejected), and the
  `ΔG ≤ 0` short-circuit handles the downhill case without calling `exp`.
- **Stream/stdout split.** Deterministic results go to **stdout** (diffed by the
  demo); timings and the run-varying mismatch count go to **stderr**.

---

## 6. How we verify correctness

Two independent checks:

1. **Exact CPU↔GPU agreement (tolerance `== 0`).** `src/reference_cpu.cpp` runs
   the identical ensemble serially with plain `+=`, and `src/main.cu` compares
   every slot of the `[n_pH × N]` integer tally. Because both sides run the same
   RNG-seeded chains and accumulate integers, **every** count must match exactly;
   a single mismatch is a real bug, not Monte-Carlo noise. This is the `==`
   tolerance class of PATTERNS.md §4 (the same as popcount/integer-DP/fixed-point
   flagships `1.12`, `3.01`, `5.01`, `11.09`). The demo's PASS line confirms it.
2. **Analytic / physical sanity (Henderson–Hasselbalch).** With coupling switched
   off (`coulomb_k = 0`), the coupling term vanishes and each residue must titrate
   at *exactly* its intrinsic pKa: the analytic single-site curve is
   `f_i(pH) = 1 / (1 + 10^{pH − pKa_i})` for a base (and the mirror for an acid),
   which crosses 0.5 at `pH = pKa_i`. The learner can verify this directly
   (Exercise 1): the predicted pKa returns to the intrinsic value within Monte
   Carlo noise. Turning coupling back on produces the **shift** the project
   teaches. (An exact `==` check on CPU vs GPU validates the *implementation*; the
   H–H check validates the *science*.)

Edge cases handled: ragged last grid block (grid-stride guard); off-grid pKa
(reported as `(off-grid)`, NaN crossing); degenerate zero distance (clamped);
malformed input (loader throws with a precise message).

---

## 7. Where this sits in the real world

This project deliberately keeps the **statistical-mechanics skeleton** of CpHMD
(pH-coupled Metropolis titration, electrostatic coupling, pKa readout, the
ensemble GPU pattern) and **replaces the physics engine** with an analytic
surrogate. Production constant-pH MD differs in every one of those substitutions:

- **Full molecular dynamics, not frozen atoms.** AMBER `pmemd.cuda`, OpenMM, and
  CHARMM run real MD: thousands of atoms with a force field (AMBER ff14SB/ff19SB),
  integrating Newton's equations on the GPU. Conformation and protonation are
  sampled *together* — the histidine actually flips, the loop actually moves. A
  400-residue protein at one pH runs in ~1 hour on an RTX 2080, >1000× faster than
  CPU. The dominant cost is the **non-bonded force evaluation** (PME for explicit
  solvent, or Generalized-Born for implicit) — itself a massively parallel kernel.
- **Real solvation, not a single dielectric.** Continuous CpHMD recomputes the
  solvation free energy of each protonation move with GB (e.g. GBSW in CHARMM) or
  explicit-solvent PME, capturing desolvation penalties a constant `ε` cannot.
- **Replica exchange across pH (REX-CpHMD).** Independent pH windows periodically
  *swap* configurations (Metropolis on the pH coordinate) to escape kinetic traps,
  parallelized across GPUs with **NCCL/MPI**. We average independent replicas but
  do not exchange — adequate for the weakly coupled toy, inadequate for buried,
  strongly coupled sites.
- **cuRAND, not a hand-rolled RNG.** Production codes draw randomness with
  **cuRAND** (e.g. XORWOW/Philox), which is faster and statistically validated.
  The cost: the host can no longer reproduce the device's exact stream, so
  verification becomes *statistical* (compare titration distributions, check pKa
  to within Monte Carlo error) rather than an exact integer match. We chose
  reproducibility for teaching (Exercise 5 explores the swap).
- **Setup tools.** `PropKa` gives fast empirical pKa estimates to initialize a
  system; pKa databases like **PKAD** and benchmark sets provide experimental
  values to validate predictions.

The transferable lessons here — independent stochastic chains → one thread each,
a shared `__host__ __device__` physics core for exact verification, and integer
atomics for deterministic reduction — are exactly the ideas that scale up into
those production engines.

---

## References

- **AMBER `pmemd.cuda` CpHMD** — <https://ambermd.org/GPUSupport.php>. The
  reference GPU constant-pH MD implementation; study its GB/PME continuous
  titration and REX-CpHMD design.
- **OpenMM constant-pH** — <https://github.com/openmm/openmm>. A readable,
  Python-scriptable CpH framework; good for seeing the MC-move/MD interleave.
- **CHARMM CpHMD (GBSW)** — <https://www.charmm.org>. Implicit-solvent titration;
  contrast its solvation model with our single dielectric.
- **PropKa** — <https://github.com/jensengroup/propka>. Fast empirical pKa
  prediction used to set up CpHMD systems; a good baseline to compare against.
- **PKAD** — <https://compbio.clemson.edu/pkad/>. Experimental protein-residue pKa
  database for validating predicted shifts.
- Metropolis et al., *Equation of State Calculations by Fast Computing Machines*,
  J. Chem. Phys. 21 (1953) — the Monte Carlo acceptance rule used throughout.
