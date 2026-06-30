# THEORY — 2.28 Replica Exchange Solute Tempering (REST2) on GPU

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

A folded protein is a marble rolling in a rugged landscape of potential energy.
The biologically interesting events — folding, a loop flipping, a ligand binding,
a side chain rotating — are the marble *hopping between basins*, and those hops
require climbing over **energy barriers**. At body temperature (≈ 300 K) the
thermal energy `kT` is small compared with a typical barrier, so a hop is **rare**:
the system can rattle inside one basin for microseconds to seconds before it
escapes. A plain molecular-dynamics (MD) simulation, which advances time in
femtosecond steps, simply cannot wait that long. This is the **sampling problem**:
the simulation gets stuck, and any average you compute reflects one basin, not the
true equilibrium population.

The oldest cure is **temperature**: heat the system and barriers become easy to
cross (the Boltzmann factor `exp(−ΔE/kT)` rises). **Parallel tempering** (replica
exchange MD, REMD) runs many copies ("replicas") at a ladder of temperatures and
periodically swaps configurations between neighbours, so a configuration can wander
*up* the ladder, cross a barrier where it is hot, and wander back *down* to the
cold, physical temperature. The catch for biomolecules: a protein is dissolved in
*tens of thousands* of water molecules. Heating the **whole** box means the number
of replicas you need to keep neighbouring temperatures close enough for swaps to be
accepted grows like √(degrees of freedom) — astronomically expensive, because
almost all those degrees of freedom are uninteresting water.

**REST2 (Replica Exchange with Solute Tempering, version 2)** is the elegant fix:
heat **only the solute** (the protein/ligand and its coupling to water), and leave
the bulk **water–water** interactions at 300 K. Far fewer replicas are needed
because only the solute's degrees of freedom are being "tempered". REST2 is a
workhorse for protein–ligand binding free energies, loop and side-chain
reorganization, and fast-folder studies (chignolin, Trp-cage).

This project teaches the **REST2 machinery** — the energy bookkeeping and the swap
rule — on a deliberately tiny, transparent system so every number is checkable.

## 2. The math

**State.** The solute is `N = 8` beads with 1-D coordinates `x = (x₀,…,x₇)`. Each
bead sits in a **tilted double-well** potential

```
u(x) = h·(x² − 1)² − tilt·x
```

with `h` the barrier height (in units of `kT`, which we set to 1) and `tilt` a
small linear bias that lowers the **right** well so it is the **global** minimum
and the left well is merely **metastable**. Minima are near `x = ±1`, barrier at
`x = 0`.

**The REST2 energy decomposition.** The total potential of a solvated system splits
into three physically distinct groups of interactions:

| Group | Meaning | In this toy model |
|------|---------|-------------------|
| `E_pp` | solute–solute ("protein–protein"), the solute's internal energy | Σ double-well + harmonic bonds between neighbouring beads |
| `E_pw` | solute–solvent ("protein–water") | Σ `k_pw·(xᵢ − x_solvent)²` coupling to an implicit solvent field |
| `E_ww` | solvent–solvent ("water–water") | constant (our implicit solvent has no internal motion) |

**The scaled (effective) Hamiltonian.** Replica `m` samples not the physical
energy but a **scaled effective energy**

```
E_eff(λ_m) = λ_m · E_pp + √λ_m · E_pw + E_ww,      λ_m = β_m/β₀ = T₀/T_m ∈ (0, 1].
```

- `λ = 1` (cold replica, `T_m = T₀`): nothing is scaled → true physics.
- `λ < 1` (hot replicas): the solute and solute–solvent energies **shrink**, so
  barriers in the solute look smaller and the replica explores freely.
- The **√λ on the cross term** is the defining REST2-v2 correction (Wang, Friesner
  & Berne 2011). It is exactly what you get if you rescale the solute's partial
  charges and Lennard-Jones ε by `√λ`: a charge–charge or LJ term between two
  *solute* atoms scales as `(√λ)² = λ` (→ `E_pp`), while a term between a *solute*
  atom and a *water* atom scales as `√λ·1 = √λ` (→ `E_pw`). The water–water term is
  untouched. REST1 used a less consistent scaling; the √λ is why "v2" exists.

**The exchange (swap) criterion.** Periodically, neighbouring replicas `m, n` try
to swap configurations. The acceptance follows detailed balance in the joint
ensemble; for REST2 the famous simplification is that the **unscaled water–water
term cancels**, leaving only solute terms:

```
Δ = (λ_m − λ_n)·(E_pp(n) − E_pp(m)) + (√λ_m − √λ_n)·(E_pw(n) − E_pw(m))
accept swap with probability  min(1, exp(−Δ)).
```

(Here `kT = 1` is folded into λ.) A positive acceptance lets the cold replica
inherit a barrier-crossed configuration discovered by a hotter one.

**Sampling within a replica.** We sample the Boltzmann distribution of `E_eff` with
**Metropolis Monte Carlo**: propose a small displacement `dx` to one bead, accept
with `min(1, exp(−ΔE_eff))`. MC needs only energy *differences* (no forces), which
keeps the teaching code transparent; it leaves the same Boltzmann distribution
invariant that MD with a thermostat would.

## 3. The algorithm

```
initialize: every replica's beads in the LEFT (metastable) well, x = −1
build ladder: T_m geometric from T0..Tmax;  λ_m = T0/T_m;  per-replica RNG seed
for round = 0 .. n_rounds−1:
    (a) SAMPLE  — for each replica r (PARALLEL): run sweeps_per_round MC sweeps
                  at E_eff(λ_r); accumulate accepted moves
    (b) EXCHANGE — for neighbour pairs of parity (round mod 2):
                  compute Δ from E_pp, E_pw; accept swap with prob exp(−Δ)
report: per-replica well occupancy + acceptance; did the cold replica escape?
```

**Geometric temperature spacing.** `T_m = T₀·(Tmax/T₀)^(m/(M−1))`. Exchange
acceptance depends on temperature *ratios*, so equal ratios → roughly uniform
acceptance between every neighbouring pair, the standard choice.

**Even/odd alternation.** On even rounds we consider pairs `(0,1),(2,3),…`; on odd
rounds `(1,2),(3,4),…`. Alternating parity lets a configuration migrate the full
length of the ladder over successive rounds, and guarantees no replica is in two
swaps at once.

**Complexity.** Let `M` = replicas, `S` = `sweeps_per_round`, `R` = `n_rounds`,
`N` = beads. One MC sweep touches all `N` beads and (in this clear-but-naive code)
recomputes the `O(N)` effective energy per move, so a sweep is `O(N²)` and the
whole run is `O(M·R·S·N²)`. Serially the replicas run one after another (factor
`M` in time); **in parallel each replica is one thread**, so the *depth* drops the
`M` factor — wall-clock `O(R·S·N²)` plus the cheap `O(M·N)` exchange per round. A
local-energy-delta sweep (Exercise 2) removes the extra `N`, giving `O(M·R·S·N)`
work.

## 4. The GPU mapping

**Thread-to-data map: one thread per replica.**

```
        replicas (the ladder)                     one round
   r0    r1    r2   ...   r7
  ┌──┐  ┌──┐  ┌──┐       ┌──┐     (a) SAMPLE  : kernel, thread r owns replica r
  │T0│  │..│  │..│  ...  │T7│                    runs S sweeps in registers
  └──┘  └──┘  └──┘       └──┘     (b) EXCHANGE : host swaps neighbour configs
   λ=1                   λ=.33                    (touches only M energies)
   thread r = blockIdx.x*blockDim.x + threadIdx.x
```

- **Launch config.** `THREADS_PER_BLOCK = 64` (a multiple of the 32-lane warp);
  `blocks = ceil(M / 64)`. REST2 ladders are small (tens of replicas), so one
  block usually covers the whole ladder; the guard `if (r >= M) return;` retires
  threads in the ragged last block.
- **Registers, not global memory.** A replica's `N = 8` coordinates are pulled into
  a small **register array**, advanced entirely in registers through the whole
  sampling loop, then written back once. There is **no inter-thread communication**
  inside the kernel → **no shared memory, no `__syncthreads`, no atomics**. Each
  thread also owns its own `accepted[r]` and `rng_ctr[r]` slot, so the writes never
  race.
- **Why the exchange is on the host.** It reads `M` pairs of energies and may swap
  two coordinate blocks — negligible work. Doing it on the host keeps the data flow
  explicit for the learner. The host re-uploads/-downloads the (tiny) state each
  round; a production engine keeps state **resident** on the GPU and exchanges via
  **NCCL**/peer copy (Exercise 4).
- **No CUDA library is needed.** The whole computation is custom arithmetic plus a
  hand-written **counter-based RNG** (a SplitMix64-style integer hash of
  `(seed, draw-index)`). We deliberately avoid `curand`: a *stateful* generator's
  stream depends on history, which breaks reproducibility across CPU/GPU and across
  runs. Hashing an explicit counter — the same idea as Philox/Random123 — gives the
  same bits on every device, every run. Writing such a finalizer by hand is three
  xor-shift/multiply rounds (see `rng_hash64` in `rest2.h`).

The single most important idiom is the **shared `__host__ __device__` core**
(`rest2.h`): the energies, the RNG, one MC sweep, and the exchange Δ are all
`__host__ __device__` inline functions. The CPU reference loops them; the kernel
calls them from one thread. One source of truth → the two paths cannot silently
diverge (PATTERNS.md §2).

## 5. Numerical considerations

- **Precision: FP64 throughout.** Energies, the Boltzmann factor, and the RNG
  output are `double`. The barrier-crossing physics is sensitive to small energy
  differences, and double precision keeps the Metropolis test well-conditioned.
- **Determinism within a run.** The counter RNG makes every draw a pure function of
  `(seed, counter)`, and we thread the counter through the simulation so no two
  draws collide. Re-running the same binary gives **byte-identical stdout**.
  Per-replica accumulators are integers (accepted-move counts), which commute, so
  there is no float-summation-order issue (PATTERNS.md §3).
- **CPU vs GPU drift — the honest part.** Even in double precision, the GPU and host
  do *not* automatically produce bit-identical trajectories:
  1. **FMA contraction.** `a*b + c` may be fused into one rounded operation on the
     GPU but two on the host. We disable this with **`--fmad=false`**, after which
     IEEE `*` and `+` are correctly rounded *identically* on both sides — so the
     **energy arithmetic matches exactly**.
  2. **Transcendentals.** `exp()` (in the Metropolis test) and, in principle,
     `sqrt()` come from *different* math libraries on host vs device. `sqrt` is
     IEEE-correctly-rounded (identical), but `exp` is **not guaranteed** identical —
     it can differ by ~1 ULP. When a proposed move's acceptance probability sits
     *exactly* on the random roll, that 1-ULP difference can **flip a single
     accept**. Because an MC trajectory is **chaotic** (sensitive dependence on
     state), one flipped move can send the two trajectories apart afterward.

  This is real and worth internalizing: *bit-identical trajectories are not a sound
  correctness gate for a chaotic stochastic sampler.* (The sibling project `1.06`
  teaches the same lesson for Langevin metadynamics.)

## 6. How we verify correctness

We run the **whole REST2 simulation twice** — once with `cpu_sample_round`
(serial) and once with `gpu_sample_round` (kernel) — sharing the **same exchange
step** (`run_exchange` in `main.cu`), so the only difference is loop-vs-kernel
sampling. Then we compare **robust statistical observables** that are stable under
a few flipped moves, exactly the kind of readout a real REST2 study reports:

- **Right-well occupancy** summed across all replicas (an integer count of the
  sampled population): allowed to differ by at most `2·M` beads out of `M·N`.
- **Global Monte-Carlo acceptance ratio**: allowed to differ by at most 0.01
  (1 percentage point).

With `--fmad=false` the two paths in fact agree **exactly** on these observables in
the committed demo (`right-well beads CPU=63 GPU=63`, `acceptance diff 0`,
`exchanges 152=152`), but the *tolerances* are the documented contract — they
absorb the ~1-ULP `exp()` drift if a borderline accept ever flips. We deliberately
do **not** assert a bit-identical trajectory (§5).

A second, **physical** check (more convincing than CPU == GPU): the cold replica
r0, started with **0/8** beads in the right well, ends with **8/8** in the right
(global-minimum) well. A 300 K replica left to itself would stay trapped; only the
REST2 exchanges let r0 inherit the hot replicas' barrier crossings. That the right
well — the *known* global minimum we engineered with `tilt` — is the one populated
is the science validating itself (PATTERNS.md §4, "compare against a known result").

## 7. Where this sits in the real world

Production REST2 differs from this teaching version in scale and machinery, not in
principle:

- **Real MD, not MC.** GROMACS, NAMD, OpenMM, and DESMOND integrate Newton's
  equations with a full force field (bonded + Lennard-Jones + PME electrostatics)
  and a thermostat, on systems of 10⁴–10⁶ atoms. The √λ scaling is applied by
  **rescaling the solute's charges and LJ ε by √λ** in each replica's topology —
  precisely the algebra in §2 — so no special-cased energy split is needed at run
  time; the standard force kernels just see a modified topology.
- **GPU force kernels.** The expensive part is the non-bonded force evaluation
  (neighbour lists, PME on the GPU via cuFFT). Our toy has no forces, so we use the
  cheap MC stand-in; the **replica-parallel structure** and the **exchange logic**
  are the transferable lessons.
- **Communication.** With one replica per GPU, swaps exchange energies (and
  sometimes coordinates) via **NCCL** within a node and **MPI** across nodes. Our
  host-side swap is the single-GPU analogue.
- **Refinements.** PLUMED couples REST2 with metadynamics/bias-exchange; **vRE-REST2**
  (virtual replica exchange) reduces the replica count further; **HREX** (scale the
  *whole* Hamiltonian) is the more general but more expensive cousin REST2
  specializes (Exercise 3). Validation uses fast-folders (chignolin, Trp-cage) and
  the SAMPL blind challenges.

What this version omits: explicit solvent, real force fields, PME, thermostats,
constraints (SHAKE/LINCS), and multi-GPU communication. What it keeps and makes
inspectable: the **energy decomposition**, the **λ·E_pp + √λ·E_pw + E_ww**
effective Hamiltonian, the **λ-ladder**, and the **REST2 exchange criterion**.

---

## References

- Liu, Kim, Friesner, Berne, *PNAS* **102**, 13749 (2005) — original Replica
  Exchange with Solute Tempering (REST). The idea of tempering only the solute.
- Wang, Friesner, Berne, *J. Phys. Chem. B* **115**, 9431 (2011) — **REST2**: the
  `√λ` cross-term scaling derived from rescaling charges/ε by `√λ`. The core of §2.
- Salomon-Ferrer, Case, Walker, *WIREs Comput. Mol. Sci.* (2013) — GPU MD overview;
  why force evaluation dominates and how it is parallelized.
- GROMACS + PLUMED (https://github.com/gromacs/gromacs, https://www.plumed.org) —
  the reference open-source REST2/Hamiltonian-REMD implementation; read how the
  per-replica topology scaling is set up.
- openmmtools (https://github.com/choderalab/openmmtools) — clean Python `REST2`
  classes; the most readable way to see the scaling applied to a `System`.
- Salvador, Random123 (Philox) — the counter-based RNG idea used for our
  deterministic, device-independent random stream.
