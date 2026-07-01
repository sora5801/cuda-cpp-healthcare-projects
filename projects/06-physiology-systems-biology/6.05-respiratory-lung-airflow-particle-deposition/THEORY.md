# THEORY — 6.5 Respiratory / Lung Airflow & Particle Deposition

> A didactic deep dive. We build a **reduced-scope teaching version** of aerosol
> deposition in the lung: Lagrangian particle tracking through an idealized
> bifurcating airway tree, with the three classical deposition mechanisms
> (impaction, sedimentation, diffusion). The full CFD picture the catalog
> describes — incompressible Navier–Stokes on a CT-reconstructed geometry with
> k-ω SST turbulence and two-way coupling — is summarized in
> [§ Where this sits in the real world](#where-this-sits-in-the-real-world).
> **Not for clinical use.**

---

## The science

When you inhale, air flows down the **conducting airways** — a branching tree of
tubes that starts at the trachea (generation 0) and bifurcates ~16 times before
reaching the alveoli where gas exchange happens. Suspended in that air are
particles: pollutants, allergens, or, of clinical interest, **inhaled drug
aerosols** from an inhaler or nebulizer. Whether a drug reaches its target depends
on *where in the tree its particles deposit onto the airway walls*.

Three physical mechanisms pull a particle out of the airstream and onto a wall:

1. **Inertial impaction.** At each bifurcation the air turns sharply. A heavy or
   fast particle has too much inertia to follow the streamline and slams into the
   wall. This dominates for **large particles (> ~3 µm) in the fast upper
   airways**.
2. **Gravitational sedimentation.** Given time, a particle settles under gravity
   at its terminal velocity. This matters where the air moves slowly and residence
   time is long — the **mid-depth airways** — for **medium particles (~1–5 µm)**.
3. **Brownian diffusion.** Very small particles get kicked around by air molecules
   and random-walk to the wall. This dominates for **sub-micron particles
   (< ~0.5 µm) in the slow deep airways**.

The competition between these three is why deposition is *size-selective*: the
particle diameter you choose largely decides where the drug lands. That is the
single most important lesson of this project.

---

## The math

### Particle properties (computed once per aerosol)

For a spherical particle of diameter `d_p` and density `ρ_p` in air of viscosity
`μ`, the key derived quantities are:

- **Cunningham slip correction** (sub-micron particles slip through the gas rather
  than seeing a continuum): with Knudsen number `Kn = 2λ/d_p` (λ = air mean free
  path),
  ```
  C_c = 1 + Kn·(1.257 + 0.400·exp(-1.10/Kn))
  ```
- **Relaxation time** — how fast the particle catches up to the flow (drives
  impaction):
  ```
  τ = ρ_p · d_p² · C_c / (18 μ)
  ```
- **Settling velocity** (drives sedimentation):  `v_s = τ · g`
- **Diffusion coefficient** (Stokes–Einstein; drives diffusion):
  ```
  D = k_B · T · C_c / (3π μ d_p)
  ```

These come straight from Stokes drag on a small sphere; the derivations are in any
aerosol-mechanics text (Hinds, *Aerosol Technology*).

### Per-generation deposition efficiencies

In generation `g` (tube radius `r`, length `L`, mean axial velocity `U`), the
residence time is `t_res = L/U`. The probability that a particle deposits in that
generation via each mechanism (semi-empirical closed forms, matching ICRP-66 /
MPPD):

| Mechanism | Dimensionless group | Efficiency |
|---|---|---|
| Impaction | Stokes `Stk = τU/r` | `η_imp = 1 − exp(−2·Stk)` |
| Sedimentation | `v_s·t_res / (2r)` | `η_sed = clamp(v_s·t_res/(2r), 0, 1)` |
| Diffusion | `Δ = D·t_res/r²` | `η_diff = 1 − exp(−5.784·Δ)` |

Treating the three as independent over the tube, the probability a particle
**survives** generation `g` is
```
P_survive(g) = (1 − η_imp)(1 − η_sed)(1 − η_diff).
```

### The airway geometry (Weibel-A, symmetric)

Generation `g` has `2^g` parallel tubes; radius and length shrink geometrically:
```
r[g+1] = 0.85·r[g],   L[g+1] = 0.62·L[g],   r[0] = 9 mm, L[0] = 12 cm.
```
Incompressible **continuity** fixes the per-tube velocity from the whole-lung
flow `Q`:
```
A_tot(g) = 2^g · π · r[g]²,      U[g] = Q / A_tot(g).
```
`A_tot` grows fast with depth, so `U[g]` falls sharply — which is exactly why
impaction (∝ `U`) dominates up top and diffusion/sedimentation (∝ `t_res = L/U`)
dominate deep down.

---

## The algorithm

**Monte-Carlo Lagrangian tracking.** Launch `N` particle histories. Each history:

```
for g = 0 .. n_gen-1:
    compute η_imp, η_sed, η_diff  from (particle, r[g], L[g], U[g])
    P = (1-η_imp)(1-η_sed)(1-η_diff)         # survival probability
    draw ξ ~ Uniform[0,1)
    if ξ >= P:  deposit in generation g; STOP
survived all generations -> "exhaled"
```

Tally an **integer count** per generation (plus one "exhaled" bucket). The
deposition *fraction* is `deposited / N`; the per-generation histogram is the
deposition *profile*.

- **Serial complexity:** `O(N · n_gen)` time, `O(n_gen)` space. `n_gen` is tiny
  (~16), so cost is `O(N)` in practice — and every history is independent.
- **Convergence:** a Monte-Carlo fraction converges as `1/√N` (standard error
  `≈ √(p(1−p)/N)`). Our 200 000 histories give a ~0.1 % standard error on the
  21.9 % deposition fraction — plenty for a stable, teachable demo.

---

## GPU mapping

The histories are **independent**, so this is the textbook "per-thread jobs +
atomic scoring" pattern (docs/PATTERNS.md §1, exemplified by flagship 5.01 Monte
Carlo dose).

- **Thread ↔ data:** one GPU thread tracks one particle. A fixed grid of 1024
  blocks × 256 threads runs a **grid-stride loop**, so the same launch covers
  `N = 10³` or `10⁹` particles. Thread `start = blockIdx.x·blockDim.x+threadIdx.x`
  handles particles `start, start+stride, …` where `stride` = total thread count.
- **Per-thread RNG:** each thread seeds its own **splitmix64** stream from its
  particle index (`rng_seed(seed, i)`), so histories are uncorrelated yet
  reproducible. Because the RNG + physics live in the **shared `lung_physics.h`**
  (`__host__ __device__`), the CPU reference replays the *identical* histories.
- **Atomic integer scoring:** many threads deposit into the same per-generation
  counters, so the tally uses `atomicAdd` on **64-bit integers**. Integer adds
  commute, so the GPU result is **order-independent, deterministic, and equals the
  CPU tally exactly**. (A floating-point tally would depend on the nondeterministic
  atomic ordering — see Numerical considerations.)
- **Memory:** almost nothing crosses PCIe. There is no per-particle input array —
  particles are generated on the fly from their index — so only the tiny tally
  (n_gen+1 counters) is copied back. The particle properties and airway geometry
  are small `by-value` structs passed in registers.
- **Divergence:** particles deposit after different numbers of generations, so
  warp lanes finish their loops at different times — the classic Lagrangian cost.
  Production codes sort/compact particles by state to cut it; we keep the straight
  loop for clarity.

```
        host                          device (1024×256 threads, grid-stride)
   build_airway() ──► Airway ──►┐
   make_particle() ─► Particle ─┼─►  for each particle i:
                                │        Rng r = rng_seed(seed, i)
                                │        g = track_particle(p, aw, r)
   d_tally (zeroed) ◄───────────┘        atomicAdd(&tally[g], 1)   // integer
        │
        └─ copy back ──► compare to CPU tally (must be identical)
```

---

## Numerical considerations

- **Precision:** all per-particle physics is **double precision** (`FP64`). The
  exp/log calls and the Stokes/Cunningham formulas are evaluated with the *same*
  code on host and device, so the survival probability `P` and the RNG draw `ξ`
  are bit-identical → the deposit-in-generation decision is identical → the
  integer tallies match exactly. `--use_fast_math` is **OFF** (it would let the
  device `exp` diverge from the host `exp`).
- **Determinism via integer atomics:** this is the load-bearing trick
  (docs/PATTERNS.md §3). If we accumulated a *float* dose per generation, the
  atomic-add order (nondeterministic across warps) would perturb the low bits and
  the GPU/CPU sums would differ run-to-run. Counting integer particles sidesteps
  this entirely: `a+b = b+a` for integers, always. (Empirically confirmed: Debug
  and Release builds produce byte-identical stdout.)
- **RNG quality:** splitmix64 is a well-mixed counter-based generator — good
  enough for a teaching Monte-Carlo and, crucially, trivially reproducible from
  `(seed, index)` on both host and device. Production codes use cuRAND
  (Philox/XORWOW); we deliberately do not, so verification stays exact.
- **Edge cases:** `η_sed` is clamped to `[0,1]`; the exponential-form efficiencies
  are already in `[0,1)`; `Kn` is finite for any positive `d_p`.

---

## How we verify correctness

Two independent checks:

1. **GPU == CPU, exactly.** `main.cu` runs `deposition_cpu` (serial, plain `++`)
   and `deposition_gpu` (parallel, `atomicAdd`) on the *same* histories and asserts
   **every** per-generation count is identical (`mismatches == 0`). Tolerance is
   **exact (`== 0`)** — justified because the tally is integer and the physics is
   the shared, bit-identical `track_particle` (docs/PATTERNS.md §4).
2. **Physical sanity.** The reported profile matches known aerosol physics: for a
   5-µm particle at resting flow, total deposition is ~20 % and **peaks in the
   trachea** (impaction-dominated, highest velocity), decreasing monotonically with
   depth. Sweeping `d_p` down to sub-micron flips the peak deeper (diffusion) — the
   classic U-shaped total-deposition-vs-size curve. These are the sanity checks a
   reviewer would apply to a real deposition model.

If a change breaks either check, the demo fails loudly (nonzero exit code, `FAIL`
in stdout).

---

## Where this sits in the real world

This is a **reduced-scope teaching model**. A production inhaled-dosimetry study
differs in several ways:

- **Geometry:** instead of an idealized Weibel-A tree, it uses a patient airway
  surface **segmented from a CT scan** (LIDC-IDRI / COPDGene / SPIROMICS), meshed
  into 10⁶–10⁷ cells (3D Slicer + SlicerMorph for segmentation).
- **Flow field:** it solves the **incompressible Navier–Stokes** equations (finite
  volume) with a **k-ω SST RANS** turbulence model for the transitional/turbulent
  upper-airway flow — a coupled linear solve each timestep (the catalog's cuSPARSE
  step), rather than our algebraic continuity velocities.
- **Particle forces:** full **Lagrangian discrete-phase** integration of drag +
  gravity + Saffman lift + a Brownian random force, stepping each particle through
  the resolved velocity field — not per-generation deposition-probability formulas.
- **Coupling & chemistry:** alveolar gas exchange adds a **convection–diffusion**
  layer for O₂/CO₂ coupled to a quasi-1D (Horsfield-tree) ventilation model.
- **Tooling:** OpenFOAM's `DPMFoam` (with GPU-accelerated AmgX pressure solves),
  PALABOS (LBM for alveolar-scale flow), SimVascular (vascular flow adaptable to
  airways).

The **deposition-probability model** we implement is itself a legitimate,
widely-used approach — it is the mathematical core of ICRP-66 and the MPPD
(Multiple-Path Particle Dosimetry) model used in regulatory inhalation risk
assessment. What we omit is the CFD flow field; what we keep is the size-selective
deposition physics and, above all, the **GPU pattern**: millions of independent
Lagrangian histories, one per thread, scored with deterministic integer atomics.
