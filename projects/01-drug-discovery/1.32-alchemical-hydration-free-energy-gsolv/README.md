# 1.32 — Alchemical Hydration Free Energy (ΔGsolv)

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.32`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Compute a **hydration free energy** ΔG_solv — the reversible work to move a solute
from vacuum into solvent — the way real free-energy codes do: **alchemically**. We
define a coupling parameter λ ∈ [0,1] that smoothly switches the solute's
interaction with the solvent on and off, run Monte Carlo sampling at a ladder of
λ-windows, and turn the samples into ΔG by **Thermodynamic Integration (TI)** and
the **Bennett Acceptance Ratio (BAR)**. Every λ-window runs many independent
Monte Carlo walkers, so each walker gets its own GPU thread — the
**ensemble-over-threads** pattern. To stay teachable, the system is a single
Lennard-Jones (+ optional charge) solute in a fixed solvent bath (a labeled
reduced-scope model — see Limitations); every method idea (λ-windows, **soft-core**
potentials, TI, BAR) is the real thing.

## What this computes & why the GPU helps

The quantity is ΔG_hyd / ΔG_solv, foundational to ADMET (LogP, solubility,
permeability). Alchemically,

> ΔG(switch on) = ∫₀¹ ⟨∂U/∂λ⟩_λ dλ   (Thermodynamic Integration)

where ⟨·⟩_λ is a Boltzmann average **sampled at fixed λ**. The expensive part is
that sampling: each λ-window needs many decorrelated configurations, and you need
several windows. Those samples come from **independent Monte Carlo chains** that
never talk to each other — perfect data parallelism. We launch **one GPU thread
per (window, walker)** chain; on the committed sample that is 704 chains in
flight. The GPU's edge over the serial CPU grows with `n_windows × n_walkers`
(real campaigns run thousands of walkers across many GPUs).

**The parallelized work** is the ensemble of Metropolis MC chains; each thread runs
its full chain in registers and writes its accumulated `⟨∂U/∂λ⟩` and BAR energy
differences.

## The algorithm in brief

- **Soft-core LJ coupling.** `U_sc(r,λ) = λ·4ε(x²−x)`, `x = σ⁶/(α σ⁶(1−λ) + r⁶)`.
  The `α σ⁶(1−λ)` term softens the r→0 singularity at small λ, killing the
  TI-variance "end-point catastrophe".
- **Metropolis Monte Carlo.** Per window, each walker proposes solute displacements
  and accepts with `min(1, e^{−βΔU})`; after burn-in it accumulates `∂U/∂λ`
  (analytic) and the neighbouring-window energy differences (for BAR).
- **TI.** ΔG = −∫₀¹⟨∂U/∂λ⟩dλ by the trapezoidal rule over the window grid.
- **BAR.** Combine adjacent windows' forward/backward energy differences into a
  minimum-variance ΔG; cross-check against TI.

See [THEORY.md](THEORY.md) for the science, the math, the soft-core derivation, the
GPU mapping, and the numerics.

## Build

Requires **Visual Studio 2026** (v145) + **CUDA 13.3** ([docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/alchemical-hydration-free-energy-gsolv.sln`.
2. Select **`Release|x64`** → **Build** →
   `build/x64/Release/alchemical-hydration-free-energy-gsolv.exe`.

CLI:
`msbuild build\alchemical-hydration-free-energy-gsolv.sln /p:Configuration=Release /p:Platform=x64`

## Run the demo

```powershell
./demo/run_demo.ps1
```

Builds if needed, runs CPU + GPU on the committed sample, verifies they agree,
and prints the per-window TI integrand plus ΔG_solv from both TI and BAR.

## Data

- **Sample (committed):** `data/sample/alchemy_config.txt` — one line describing
  the λ schedule, MC budget, and model system (11 windows × 64 walkers, LJ-only,
  reduced units). The solvent bath geometry is generated deterministically in code.
- **Regenerate / rescale:** `python scripts/make_synthetic.py [--n-windows N --n-walkers M ...]`.
- **Real benchmarks** (FreeSolv, MNSol, SAMPL, NIST ThermoML): see
  `scripts/download_data.ps1` / `.sh` and [data/README.md](data/README.md). The
  demo needs no download.

## Expected output

`demo/expected_output.txt` holds the deterministic per-window `⟨∂U/∂λ⟩` table and
the two ΔG_solv estimates. The CPU (`src/reference_cpu.cpp`) and GPU
(`src/kernels.cu`) share the per-walker Monte Carlo core (`src/alchemy.h`) and use
**double precision**, so their per-walker accumulators agree to ≈1.5e−11 — inside
the documented `1e−9` tolerance (the residual is double-precision FMA reordering,
not algorithm; see THEORY §5). `RESULT: PASS` reports that check.

## Code tour

1. [`src/main.cu`](src/main.cu) — load config, build bath, run CPU + GPU, verify,
   reduce to per-window stats, estimate ΔG by TI and BAR, print.
2. [`src/alchemy.h`](src/alchemy.h) — **the heart**: soft-core energy + analytic
   `∂U/∂λ`, the counter-based RNG, and the Metropolis walker — all `__host__
   __device__` so CPU and GPU run identical math.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU ensemble interface (one thread
   per walker).
4. [`src/kernels.cu`](src/kernels.cu) — the kernel + host wrapper (bath upload,
   launch, copy-back, timing).
5. [`src/reference_cpu.h` / `.cpp`](src/reference_cpu.cpp) — config I/O, bath
   builder, serial driver, and the TI / BAR estimators.

## Prior art & further reading

- **OpenFE** (<https://github.com/OpenFreeEnergy/openfe>) — modern open alchemical
  free-energy toolkit; study its λ-protocol and analysis (`openfe`/`openmmtools`).
- **alchemtest** (<https://github.com/alchemistry/alchemtest>) — reference test
  systems and known ΔG values for validating alchemical analysis code.
- **GROMACS + alchemlyb** (<https://github.com/gromacs/gromacs>) — a production
  GPU FEP pipeline; `alchemlyb` implements TI/BAR/MBAR on real simulation output.
- **AMBER FEP** (<https://ambermd.org>) — `pmemd.cuda` soft-core alchemical
  decoupling; a good reference for the soft-core functional form.

Study these for production practice; here we reimplement the *pattern* didactically
(CLAUDE.md §2), not copy code.

## CUDA pattern used here

**Ensemble of independent Monte Carlo chains**: one thread per (window, walker),
each running a full Metropolis chain in registers, no inter-thread communication;
a shared `__host__ __device__` walker for exact CPU/GPU parity; deterministic
counter-based RNG so stdout is reproducible (PATTERNS.md §1 "ensemble", §2 HD-core,
§3 determinism). Closest flagship: [9.02](../../09-epidemiology-public-health/9.02-large-scale-compartmental-metapopulation-models).

## Exercises

1. **Turn on the charge.** Set `--q-solute 0.5` and watch how decoupling the
   Coulomb term (the `q/r` part of `∂U/∂λ`) changes the integrand and ΔG. Then
   split the schedule: discharge **then** decouple LJ (electrostatics-first), the
   standard two-stage protocol.
2. **Window convergence.** Run `--n-windows 6, 11, 21, 41` and plot ΔG_TI vs. window
   count — the trapezoid error shrinks as the grid refines. Where does it plateau?
3. **MBAR.** Replace BAR (adjacent pairs) with **MBAR**, which uses energy
   evaluations across *all* windows at once — the modern minimum-variance estimator.
4. **Full BAR self-consistency.** Store per-sample energy differences (not just
   their means) and solve Bennett's implicit equation by bisection (THEORY §6).
5. **Replace MC with MD.** Swap the Metropolis walker for a velocity-Verlet +
   thermostat integrator (the `__host__ __device__` core stays the same shape) and
   compare ΔG and cost.

## Limitations & honesty

- **Reduced-scope teaching model (CLAUDE.md §13).** The solute is a single LJ
  (+ optional point-charge) particle in a **fixed** solvent shell, sampled by Monte
  Carlo in **reduced LJ units**. A production calculation runs full GPU molecular
  dynamics with particle-mesh Ewald electrostatics in a flexible periodic water box
  and a real force field. The method (λ-windows, soft-core, TI, BAR) is faithful;
  the **system** is a toy.
- The computed ΔG is a correct result **for this model**, **not** a force-field
  prediction of any molecule's hydration free energy, and not for any chemical or
  clinical decision.
- BAR uses a closed-form narrow-window approximation here (Exercise 4 upgrades it);
  statistical/sampling error is not reported (you would need block averaging).
- The bath is static, so there is no solvent reorganization entropy — a real,
  named simplification discussed in THEORY §7.
