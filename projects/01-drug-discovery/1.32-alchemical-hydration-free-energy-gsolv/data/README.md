# Data — 1.32 Alchemical Hydration Free Energy (ΔGsolv)

## Committed sample (`sample/alchemy_config.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** calculation setup (`scripts/make_synthetic.py`) |
| License | Public domain (CC0) |
| Contents | One line describing the λ schedule, the Monte Carlo budget, and the model system |

The committed "data" is the **setup** of an alchemical free-energy calculation,
not measured input. The solvent bath geometry is **derived deterministically**
from `bath_seed` (the same jittered-shell recipe as the C++ `build_bath()`), so
the entire calculation is reproducible from this one line — nothing else needs to
be stored.

> **Reduced (Lennard-Jones) units.** Energies are in units of the LJ well depth
> `ε`, lengths in units of the LJ diameter `σ`, temperature in `ε/k_B`. This keeps
> the numbers O(1) and the focus on the *method* (TI/BAR), not unit bookkeeping.
> THEORY.md §2 explains how to map ε≈0.65 kJ/mol, σ≈0.34 nm (argon-like) onto
> kcal/mol if you want physical numbers.

### File format (one line)

```
n_solvent box T epsilon sigma q_solute alpha_sc max_step n_windows n_walkers n_equil n_prod seed bath_seed
```

| Field | Meaning |
|---|---|
| `n_solvent` | number of fixed solvent sites in the bath |
| `box` | half-width of the cubic solute sampling box `[σ]` (solute roams in `[-box,box]³`) |
| `T` | temperature `[reduced ε/k_B]`; `β = 1/T` |
| `epsilon` | Lennard-Jones well depth `[ε]` |
| `sigma` | Lennard-Jones diameter `[σ]` |
| `q_solute` | solute partial charge (screened Coulomb term; `0` = LJ-only) |
| `alpha_sc` | soft-core α (dimensionless; ~0.5 is the standard Beutler value) |
| `max_step` | Metropolis trial-move max displacement per axis `[σ]` |
| `n_windows` | number of λ values on the uniform `[0,1]` grid (≥2; includes 0 and 1) |
| `n_walkers` | independent Monte Carlo chains per window (the parallel ensemble) |
| `n_equil`, `n_prod` | burn-in and production MC steps per walker |
| `seed` | global RNG seed (CPU and GPU share it → identical streams) |
| `bath_seed` | seed for the synthetic solvent geometry |

Default (committed): `24 3 1.5 1 1 0 0.5 0.4 11 64 200 800 20260628 7`
→ 11 windows × 64 walkers = **704 GPU threads**, 1000 MC steps each, LJ-only.

Regenerate or rescale:

```bash
python scripts/make_synthetic.py                          # the committed default
python scripts/make_synthetic.py --n-windows 21 --n-walkers 256   # finer + bigger
python scripts/make_synthetic.py --q-solute 0.5           # add the Coulomb term
```

## "Full dataset" / real hydration free energies

Real ΔG_hyd benchmarks come with atom-typed molecules and force fields, not a toy
LJ bath. `scripts/download_data.ps1` / `.sh` print pointers (and never bypass any
registration):

- **FreeSolv** (<https://github.com/MobleyLab/FreeSolv>) — 643 experimental +
  calculated hydration free energies of neutral small molecules (the standard
  benchmark). Permissive license; freely downloadable.
- **MNSol** (<https://comp.chem.umn.edu/mnsol/>) — Minnesota Solvation Database
  (water + organic solvents); requires accepting a license.
- **SAMPL** hydration challenges (<https://github.com/samplchallenges/SAMPL>) —
  blind-prediction sets.
- **NIST ThermoML** (<https://trc.nist.gov>) — curated thermochemistry.

To actually *compute* ΔG for those, you need a real alchemical engine (OpenFE,
GROMACS+alchemlyb, AMBER `pmemd.cuda`) — see README "Prior art".

## Provenance & honesty

The configuration and the bath are **synthetic** and in reduced units; the model
is a single LJ(+optional charge) solute in a fixed solvent shell sampled by Monte
Carlo. The computed ΔG is a correct TI/BAR result **for this model**, **not** a
force-field-accurate prediction of any experimental hydration free energy and not
for any chemical or clinical decision (CLAUDE.md §8).
