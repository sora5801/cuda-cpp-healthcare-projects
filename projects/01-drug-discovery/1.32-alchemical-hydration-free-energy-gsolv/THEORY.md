# THEORY — 1.32 Alchemical Hydration Free Energy (ΔGsolv)

> For a reader who knows C++ but is new to CUDA and to free-energy calculation.
> See [README.md](README.md) for the tour and build. _Educational only._

## 1. The science

When a drug-like molecule dissolves in water, the change in free energy is its
**hydration free energy** ΔG_hyd (the organic-solvent analogue is ΔG_solv). It
governs how a compound partitions between water and lipid (LogP), how soluble it
is (LogS), and how it crosses membranes — the "ADMET" properties that decide
whether a molecule is a viable drug. Experiment tabulates these (the **FreeSolv**
database has 643 of them), and computing them from physics is a long-standing goal.

You cannot just "measure the energy of the solvated molecule minus the gas-phase
molecule" in a simulation, because free energy is not a simple average of the
energy — it includes **entropy** (how many configurations are accessible). The
trick that works is **alchemy**: rather than physically pulling the molecule out of
the water (which would smash through high-energy configurations), we *gradually
turn off* its interaction with the solvent along a non-physical ("alchemical")
path, and integrate the free-energy change along that smooth path. The endpoints
are physical (fully solvated vs. fully decoupled = gas phase); the path between
them is imaginary but thermodynamically valid, because free energy is a **state
function** (path-independent).

## 2. The math

Introduce a coupling parameter **λ ∈ [0,1]** into the potential energy `U(x; λ)`
such that:

- **λ = 1**: the solute interacts fully with the solvent (solvated state).
- **λ = 0**: the solute is a non-interacting "ghost" (gas-phase state).

The free-energy difference between the endpoints is, exactly,

```
ΔG(0→1) = ∫₀¹ ⟨ ∂U/∂λ ⟩_λ  dλ              (Thermodynamic Integration)
```

where `⟨·⟩_λ` is the canonical (Boltzmann, `∝ e^{−βU}`, `β = 1/k_BT`) ensemble
average **at fixed λ**. This identity is just the chain rule applied to
`G(λ) = −k_BT ln Z(λ)` with `Z(λ) = ∫ e^{−βU(x;λ)} dx`:

```
dG/dλ = (1/Z) ∫ (∂U/∂λ) e^{−βU} dx = ⟨∂U/∂λ⟩_λ.
```

**Sign / convention.** Switching the coupling **on** (0→1) takes the solute from
gas phase into solvent, so the **solvation** free energy is
`ΔG_solv = −ΔG(switch on) = −∫₀¹⟨∂U/∂λ⟩dλ`. (A favorable, attractive solute lowers
the energy as it couples in, giving a negative switch-on integral and hence a
positive number here in our convention; the README's "Limitations" notes this is a
model number in reduced units, not a physical ΔG.)

**Reduced units.** Energies are in LJ well-depths ε, lengths in LJ diameters σ,
temperature in ε/k_B. To physicalize: argon-like ε≈0.65 kJ/mol≈0.16 kcal/mol and
σ≈0.34 nm; multiply the reduced ΔG by ε.

### The soft-core potential (why it is essential)

The bare Lennard-Jones interaction between the solute and a solvent site at
separation `r` is `U_LJ(r) = 4ε[(σ/r)¹² − (σ/r)⁶]`. The naive coupling
`U(λ) = λ·U_LJ` is a disaster near λ=0: a solvent atom can drift on top of the
now-ghostly solute (`r → 0`), where `U_LJ → ∞`. Then `∂U/∂λ = U_LJ` diverges, and
its variance — which controls the TI error — explodes (the **end-point
catastrophe**). The fix is the **Beutler soft-core** form, which softens the
short-range singularity at small λ:

```
x(r,λ)     = σ⁶ / ( α σ⁶ (1−λ) + r⁶ )          # α ≈ 0.5 (dimensionless)
U_sc(r,λ)  = λ · 4ε ( x² − x )
```

At λ=1 the offset vanishes (`x = (σ/r)⁶`) and `U_sc` is exactly plain LJ; at λ=0,
`U_sc = 0` and stays **finite for all r** (the `α σ⁶` term keeps the denominator
away from zero). Its λ-derivative, which TI integrates, we take **analytically**:

```
∂U_sc/∂λ = 4ε(x² − x)  +  λ·4ε(2x − 1)·(∂x/∂λ),   ∂x/∂λ = α σ⁶ σ⁶ / denom²
```

Keeping `∂U/∂λ` in closed form (not a finite difference) is what lets the CPU and
GPU evaluate the **identical** expression and agree to round-off. All of this lives
in [`src/alchemy.h`](src/alchemy.h) (`softcore_energy`, `softcore_dudl`).

## 3. The algorithm

We need `⟨∂U/∂λ⟩_λ` at each window. We get it by **Metropolis Monte Carlo**, which
samples configurations with the correct Boltzmann weight:

```
for each MC step:
    propose  solute' = solute + uniform(−h, +h)³        # symmetric trial move
    ΔU = U(solute', λ) − U(solute, λ)
    accept with probability min(1, e^{−β ΔU})           # Metropolis criterion
    if step ≥ burn-in:  accumulate ∂U/∂λ  and  U(λ±) − U(λ)   # TI + BAR samples
```

After a **burn-in** (equilibration) the chain is decorrelated and every production
step contributes one sample. Averaging `∂U/∂λ` over all walkers' samples gives the
window's TI integrand; the trapezoidal rule over the λ-grid gives ΔG.

**BAR** (Bennett 1976) is a second, lower-variance estimator that combines, for
each adjacent window pair (i, i+1), the energy differences sampled **in both
directions** (state i evaluated at i+1, and vice-versa). The full estimator solves
an implicit equation; for closely spaced windows the distributions are near-mirror
Gaussians and it reduces to the deterministic closed form we use here
([`bar_pair`](src/reference_cpu.cpp)). TI and BAR have different systematic errors,
so their agreement is a real correctness check (Exercise 3 upgrades BAR to MBAR).

**Complexity.** Per walker: `O((n_equil + n_prod) · n_solvent)` — each MC step
recomputes the solute–bath energy (O(n_solvent)). The ensemble is
`n_windows · n_walkers` such chains. The chains are independent → perfect
parallelism over `M = n_windows · n_walkers` threads.

## 4. The GPU mapping

**Decomposition.** One thread per **(window, walker)** chain. The flat global id
`gid ∈ [0, M)` decodes to window `w = gid / n_walkers` and walker `k = gid %
n_walkers`. Thread `gid` looks up its window's λ (and its neighbours' λ for BAR),
then runs the **entire Metropolis loop in registers/local memory** and writes one
`WalkerResult`. There is no shared memory, no atomics, and no inter-thread
communication — the canonical *ensemble-over-threads* pattern (flagship 9.02).

```
  (window 0, walker 0):  MC chain ---------------> result[0]
  (window 0, walker 1):  MC chain ---------------> result[1]
  ...                                               (one thread each, in parallel)
  (window W-1, walker K-1): MC chain ------------> result[M-1]
```

**Memory.** The only device-memory traffic during sampling is reading the
read-only solvent bath coordinates (SoA `double` arrays `d_x/d_y/d_z`). The solute
position, current energy, and the accumulators all live in registers. So the
kernel is compute-bound, not bandwidth-bound — the regime GPUs love. Block size is
128 threads (good occupancy on sm_75–sm_89 without register spilling).

**Determinism via counter-based RNG.** Each walker seeds a SplitMix64-style hash
RNG from `(seed, gid)`. The n-th random draw is a pure function `hash(seed, gid,
n)` — no shared mutable RNG state — so thread `gid` on the GPU and loop iteration
`gid` on the CPU draw the **same stream** and sample the **same chain**. We also
*always* consume one uniform per MC step (even on a trivially-accepted downhill
move) so the stream stays in lock-step regardless of which branch a step takes.

## 5. Numerical considerations

- **Precision.** Everything is **double**. Energies accumulate over ~1000 steps and
  the soft-core ratio spans orders of magnitude; double keeps the CPU and GPU in
  agreement and the TI integrand smooth.
- **Determinism.** No atomics, no cross-thread reductions during sampling. The
  per-window averaging is an **ordered host-side sum** afterward, so it is bit-for-
  bit reproducible (PATTERNS.md §3). stdout is therefore byte-identical every run
  (verified); timings go to stderr.
- **Why the tolerance is 1e−9, not 0.** Each walker sums thousands of double adds.
  nvcc and the host compiler may **contract** different multiply-adds into FMAs, so
  the two sums can differ by a few ULP that grow to ~1e−11 over the chain. This is
  the "short double-precision" class in PATTERNS.md §4: we verify to a tolerance a
  couple orders above the observed residual and **say so**, rather than pretend the
  results are bit-identical. Observed worst diff on the sample: ≈1.5e−11.
- **Soft-core safety.** The `α σ⁶ (1−λ)` offset guarantees the denominator is
  strictly positive for λ < 1; at λ = 1 the bath sits on a shell (r > 0), so plain
  LJ is finite too. No divide-by-zero, no overflow.
- **MC ergodicity.** `max_step` is tuned so acceptance is ~65–90% across windows;
  too large rejects everything, too small never moves. The acceptance column in the
  output is the diagnostic.

## 6. How we verify correctness

Two layers:

1. **CPU/GPU parity.** [`main.cu`](src/main.cu) runs every chain twice — `run_cpu`
   (serial) and `run_gpu` (one thread each) — and compares the raw per-walker
   accumulators (`sum_dudl`, `sum_du_fwd`, `sum_du_bwd`). Identical `run_walker()`
   ⇒ they agree to ≈1e−11 (`RESULT: PASS`). If these match, every downstream
   average and ΔG matches.

2. **Physics sanity.** The output is the method behaving correctly, not just two
   codes agreeing: `⟨∂U/∂λ⟩` falls monotonically as the favorable interaction
   couples in; acceptance drops as the landscape stiffens with λ; and **TI and BAR
   — two estimators with different error structure — land on the same ΔG** (|Δ| ≈
   0.04 in reduced units). A fuller validation against a known number (run an
   `alchemtest` system, or a Gaussian model with analytic ΔG) is Exercise material.

## 7. Where this sits in the real world

A production hydration-free-energy calculation differs from this teaching model in
scale, not in concept:

- **Real engine.** Full **molecular dynamics** (not MC) with a flexible **explicit
  water box**, periodic boundaries, and **particle-mesh Ewald (PME)** for
  long-range electrostatics — a cuFFT-backed `O(N log N)` solver. Forces come from a
  parameterized **force field** (GAFF, OpenFF, CHARMM), not a single LJ site.
- **Two-stage decoupling.** Electrostatics are switched off first (linear in λ is
  safe for charges), then LJ with soft-core — avoiding the charge–core clash.
- **Better estimators.** **MBAR** uses energies evaluated across *all* windows
  simultaneously for the minimum-variance estimate, with proper statistical error
  bars from autocorrelation analysis.
- **Solvent reorganization.** A flexible solvent contributes reorganization entropy
  that a *fixed* bath (ours) cannot — a real, named simplification here.
- **Production tools.** OpenFE, GROMACS+alchemlyb, AMBER `pmemd.cuda` run thousands
  of λ-walkers across GPU arrays and validate against FreeSolv/MNSol/SAMPL to
  sub-kcal/mol. The **ensemble-over-threads** mapping you learn here is exactly how
  those codes parallelize their λ-window sampling.

## References

- Kirkwood (1935) — thermodynamic integration (the coupling-parameter method).
- Beutler et al. (1994) — the soft-core Lennard-Jones potential.
- Bennett (1976) — the acceptance-ratio (BAR) free-energy estimator.
- Shirts & Chodera (2008) — MBAR, the multistate generalization.
- Mobley & Guthrie (2014) — FreeSolv, the hydration-free-energy benchmark.
- Frenkel & Smit, *Understanding Molecular Simulation* — MC, free energies, soft-core.
