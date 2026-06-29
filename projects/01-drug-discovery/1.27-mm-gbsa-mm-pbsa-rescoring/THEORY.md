# THEORY — 1.27 MM-GBSA / MM-PBSA Rescoring

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See
> [README.md](README.md) for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

When a small-molecule **drug (ligand)** binds a **protein (receptor)**, the
strength of binding is the **binding free energy** ΔG_bind. More negative ΔG_bind
means tighter binding, which usually means a more potent drug. Predicting
ΔG_bind from structure is one of the central problems of computational drug
discovery: it lets you rank thousands of candidate molecules *in silico* before
synthesising any of them.

A whole spectrum of methods trade accuracy for cost:

| Method | Cost | Accuracy |
|---|---|---|
| Docking score functions | very cheap | crude (rank-ordering only) |
| **MM-GBSA / MM-PBSA** | moderate | **good — the usual rescoring step** |
| Free-energy perturbation (FEP) | very expensive | best |

**MM-GBSA** ("Molecular Mechanics with Generalized-Born Surface Area") sits in
the sweet spot. The recipe: run a short **molecular-dynamics (MD)** simulation of
the complex, take **snapshots** along the trajectory, and for each snapshot
compute an *end-point* free-energy estimate

```
ΔG_bind  ≈  E_MM  +  ΔG_solv  −  T·ΔS
```

then **average over the snapshots**. `E_MM` is the molecular-mechanics
interaction energy (van der Waals + electrostatics); `ΔG_solv` is the free energy
of putting the molecule in water, approximated by an **implicit-solvent** model
(Generalized Born, GB; or Poisson-Boltzmann, PB); and `−T·ΔS` is a
configurational-entropy penalty.

**Why a GPU?** Each snapshot's energy is **completely independent** of every
other snapshot — there is no coupling between frames. With thousands of
snapshots, that is thousands of identical, independent jobs: the textbook
"embarrassingly parallel" workload (docs/PATTERNS.md §1). Production pipelines
also generate the trajectory on the GPU (AMBER's `pmemd.cuda`); here we focus on
the **rescoring** step — evaluating the energy on each snapshot in parallel.

This project implements a **reduced-scope teaching version** (CLAUDE.md §13): a
single complex, a rigid receptor, a handful of atoms, and a GB-style solvation
term, on **synthetic** data. It teaches the *structure* of MM-GBSA and the
*GPU parallelism over snapshots* — not production accuracy. §7 is honest about
the gap.

## 2. The math

For one snapshot, sum over every **(ligand atom i, receptor atom j)** pair the
three energy components (units: kcal/mol; distance r in Å; charge q in e):

**Electrostatics (Coulomb).**

```
E_elec = Σ_ij  k · q_i q_j / r_ij ,      k = 332.0637  (kcal·Å / (mol·e²))
```

`k` is Coulomb's constant in the AMBER unit system (charges in elementary
charges, distance in ångström, energy in kcal/mol).

**Van der Waals (Lennard-Jones 12-6).**

```
E_vdw = Σ_ij  4 ε_ij [ (σ_ij / r_ij)¹² − (σ_ij / r_ij)⁶ ]
```

with **Lorentz-Berthelot mixing** of per-atom parameters:
`σ_ij = (σ_i + σ_j)/2`, `ε_ij = sqrt(ε_i ε_j)`. The `r⁻¹²` term is short-range
repulsion (Pauli exclusion); the `r⁻⁶` term is long-range dispersion attraction;
`σ` is where the potential crosses zero and `ε` is the well depth.

**Generalized-Born solvation (the cross term).** GB approximates the
electrostatic free energy of solvation. The pairwise contribution uses **Still's
effective interaction distance** `f_GB`:

```
ΔG_GB = −k · (1/ε_in − 1/ε_w) · Σ_ij  q_i q_j / f_GB ,
f_GB  = sqrt( r_ij²  +  R_i R_j · exp( −r_ij² / (4 R_i R_j) ) )
```

where `R_i` is atom i's **effective Born radius** (how buried it is), `ε_in = 1`
is the solute interior dielectric and `ε_w ≈ 78.5` is water's. The prefactor
`−k(1/ε_in − 1/ε_w)` is large and negative, so GB **screens** (weakens) the
in-vacuo charge–charge interaction — exactly what surrounding water does.
Limits: as `r → 0`, `f_GB → sqrt(R_i R_j)` (a finite self/near term, not a
singularity); as `r → ∞`, `f_GB → r` (it reduces to a screened Coulomb `1/r`).

(The "SA" surface-area nonpolar term and the Poisson-Boltzmann alternative to GB
are described in §7. We implement the GB polar cross term.)

**Per-snapshot estimate** and **ensemble average**:

```
ΔG(s) = E_elec(s) + E_vdw(s) + ΔG_GB(s) + (−T·ΔS)
ΔG_bind = (1/S) Σ_{s=0}^{S−1} ΔG(s)        # the MM-GBSA result
```

`−T·ΔS` is a single constant here (see §5/§7 for why); `S` is the snapshot count.

**Inputs:** R receptor atoms + S·L ligand atoms (positions, charges, σ, ε, Born
radii) and the constant `−T·ΔS`. **Output:** the S per-snapshot ΔG values and
their mean.

## 3. The algorithm

```
for each snapshot s in 0..S-1:          # outer: independent, parallelizable
    acc = 0
    for each ligand atom i in 0..L-1:   # inner double loop: the pair sum
        for each receptor atom j in 0..R-1:
            r2  = |pos(i) - pos(j)|²
            acc += elec(i,j) + vdw(i,j) + gb(i,j)
    ΔG[s] = acc + (−T·ΔS)
ΔG_bind = mean(ΔG)
```

**Complexity.** Per snapshot the pair sum is `O(R·L)`. Over all snapshots the
serial cost is `O(S·R·L)`. The final mean is a trivial `O(S)` reduction.

**Work vs. depth (parallel).** The `S` snapshots are independent, so the *depth*
(critical path) is just one snapshot's pair sum, `O(R·L)`, while the *work* stays
`O(S·R·L)`. That is the ideal shape for a GPU: massive independent work, shallow
critical path.

**Arithmetic intensity.** Each pair does ~30 flops (a sqrt, an exp, several
mul/div) against ~2 `Atom` reads. The receptor (read by every snapshot) and the
per-snapshot ligand block are tiny and cache-resident, so at these sizes the
kernel is **compute-bound on the transcendentals** (sqrt/exp), not
bandwidth-bound — a useful contrast with the memory-bound flagships.

## 4. The GPU mapping

**Thread-to-data map.** One thread owns one snapshot:

```
                grid of blocks (128 threads each)
   block 0           block 1                 block B-1
 ┌───────────┐    ┌───────────┐            ┌───────────┐
 │t0 t1 … t127│    │ …         │    ...     │ …         │
 └─┬─┬─────┬─┘    └───────────┘            └───────────┘
   │ │     │
   ▼ ▼     ▼
  s0 s1 … s127     s128 …                   (grid-stride loop wraps around
                                              if S > total threads)
each thread t: ΔG[s] = snapshot_dg(receptor, ligand + s*L)   # the O(R·L) pair sum
```

`s = blockIdx.x*blockDim.x + threadIdx.x`, then a **grid-stride loop**
(`s += blockDim.x*gridDim.x`) so a capped grid covers any `S`.

**Launch configuration.** `block = 128` threads (a multiple of the 32-lane warp).
Each thread runs a register-heavy double-precision loop, so a 128-thread block
keeps register pressure from throttling occupancy while still giving the
scheduler 4 warps to hide latency. `grid = ceil(S/128)`, capped at 1024 blocks
(the grid-stride loop handles the remainder). These are teaching defaults, not a
tuned sweep — see the Exercises.

**Memory hierarchy.**
- **Global memory:** the receptor `[R]` and the flat ligand array `[S·L]`. The
  flat, snapshot-major ligand layout means thread `s` reads a contiguous block,
  and consecutive threads write consecutive `ΔG[s]` (coalesced output).
- **Registers:** each thread's `Atom li`, running sums, and scalars live in
  registers — the inner loop never touches shared memory.
- **Constant memory (discussed, not used):** the receptor is read by *every*
  thread and never written — a textbook `__constant__` candidate, whose
  broadcast cache would serve a whole warp in one transaction. We keep it in
  global memory because the receptor size is **data-dependent** (constant memory
  is a fixed 64 KB bank) and the L1/L2 cache already serves the repeated reads
  well at teaching sizes. The constant-memory version is left as an exercise.
- **No shared memory, no atomics:** outputs are independent; each thread writes
  one `ΔG[s]` exactly once. The ensemble **mean** is done on the host (an `O(S)`
  reduction not worth a kernel), which also keeps the summation order identical
  to the CPU reference (determinism, §5).

**No CUDA library is used here.** The pair sum is hand-written so nothing is a
black box (CLAUDE.md §6.1.6). A production solver might offload the GB/PB step or
the MD itself to libraries / dedicated kernels (§7).

## 5. Numerical considerations

**Precision: FP64 (double) throughout.** Binding energies are differences of
large electrostatic terms; FP32 would lose too many significant digits in the
GB/Coulomb sums. The RTX 2080 (sm_75) runs FP64 slowly relative to FP32, but
correctness-you-can-trust beats speed-you-can't-explain for study material.

**Determinism (the key design choice).** Each snapshot's pair sum is computed by
**one thread in a fixed nested order** (i outer, j inner) — *identical* to the
CPU reference's loop. Because no two threads contribute to the same output, there
are **no atomics** and therefore **no floating-point reduction reordering**
(contrast the Monte-Carlo / k-means flagships, which must accumulate in integers
to stay deterministic — docs/PATTERNS.md §3). The ensemble mean is likewise a
fixed left-to-right host sum. Result: stdout is byte-identical every run.

**Where CPU and GPU can differ by ~1 ULP.** The only operations that are not
*guaranteed* bit-identical between the host libm and CUDA math are the
transcendentals — here `exp()` inside `f_GB` (IEEE `sqrt` *is* correctly rounded
and identical). A ~1-ULP `exp` difference, divided into an `O(R·L)` sum, stays
far below `1e-6` kcal/mol. On the committed sample the two paths actually agree
to `0.000e+00` (the inputs happen to hit identical rounding), but we set the
tolerance to `1e-6` to be honest about the general case.

**Stability.** `r2` is clamped to a `1e-12` floor before any `1/r`, so a momentary
atomic clash in a real trajectory cannot produce a NaN/Inf. Our synthetic
geometry never overlaps receptor and ligand, so the clamp never fires here.

## 6. How we verify correctness

The GPU kernel and the CPU reference call the **same** `snapshot_dg()` function
(the shared `__host__ __device__` core in `src/reference_cpu.h`,
docs/PATTERNS.md §2). So "GPU vs CPU" is a check that the *parallel plumbing*
(indexing, memory transfer, launch config) is right — the physics is shared, so
it cannot disagree between the two.

- **Reference:** `rescore_cpu()` loops snapshots serially and calls
  `snapshot_dg()` — obviously correct, no parallelism.
- **Metric & tolerance:** `max |ΔG_cpu[s] − ΔG_gpu[s]| ≤ 1e-6` kcal/mol
  (justified in §5: short FP64 computation, only `exp` can differ by ~1 ULP;
  docs/PATTERNS.md §4). The demo prints the actual error to stderr
  (`0.000e+00` on the sample).
- **A second, physical sanity check:** the synthetic ligand drifts out of the
  pocket frame by frame, so the **interaction part** `ΔG − (−T·ΔS)` must rise
  monotonically toward 0 (the unbound limit). The demo output shows exactly that
  (`7.2021 → 7.9640`, i.e. interaction `−0.80 → −0.04`), validating the *science*
  of the formula, not just CPU==GPU agreement.

## 7. Where this sits in the real world

Production MM-GB(PB)SA tools (**AMBER `MMPBSA.py`**, **`gmx_MMPBSA`**,
**NAMD**, **OpenMM**) differ from this teaching version in important ways:

- **Real force fields & topologies.** Charges, σ/ε, and Born radii come from a
  parameterised force field (AMBER ff19SB / GAFF2) read from a topology file, not
  from a hand-written synthetic file.
- **Three end-point states.** True MM-GBSA computes the energy of the *complex*,
  the *free receptor*, and the *free ligand* and takes the difference
  `ΔG = G_complex − G_receptor − G_ligand`; both partners are flexible across
  the trajectory. We hold the receptor rigid and compute only the cross
  interaction — a deliberate simplification.
- **Full GB models + the SA term.** Real GB uses better effective-radius schemes
  (OBC, GBn2) and adds a **nonpolar surface-area** term `γ·SASA + b`. The
  **Poisson-Boltzmann** ("MM-PBSA") variant replaces the GB closed form with a
  numerical PDE solve of the Poisson-Boltzmann equation on a grid — more accurate
  and far more expensive (a finite-difference solver, itself GPU-parallelisable).
- **Entropy.** `−T·ΔS` is, in real work, estimated per-system by **normal-mode**
  or **quasi-harmonic** analysis, or by the **interaction-entropy** method — all
  expensive and noisy. We fold it into a single constant because a credible
  entropy estimate is its own project; the constant makes the pipeline runnable
  and the teaching point (parallel-over-snapshots) clear.
- **Per-residue decomposition** breaks `ΔG` down by residue to find the binding
  hot-spots — a straightforward extension of our pair loop (accumulate into a
  per-residue array instead of one scalar).
- **Scale.** Production runs rescore *many ligands × many frames*; the GPU win
  comes from the outer parallelism over (ligand, snapshot), exactly the axis this
  project parallelises.

---

## References

- **Genheden & Ryde (2015)**, *The MM/PBSA and MM/GBSA methods to estimate
  ligand-binding affinities*, Expert Opin. Drug Discov. — the standard review;
  read for the method's scope, accuracy, and pitfalls.
- **Still, Tempczyk, Hawley & Hendrickson (1990)**, *Semianalytical treatment of
  solvation…* (the original GB `f_GB` formula used here).
- **Hawkins, Cramer & Truhlar (1996)** — pairwise GB with effective radii.
- **AMBER `MMPBSA.py`** — <https://ambermd.org/AmberTools.php> — the reference
  implementation; study its three-state decomposition and entropy options.
- **`gmx_MMPBSA`** — <https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA> — a
  GROMACS-compatible front end; good for seeing real input/output formats.
- **NAMD MMPBSA** — <https://www.ks.uiuc.edu/Research/namd/> — NAMD-based PB/GB.
- **OpenMM** — <https://openmm.org> — Python MD/MM toolkit with GB models, handy
  for prototyping a more complete GBSA in a few dozen lines.
