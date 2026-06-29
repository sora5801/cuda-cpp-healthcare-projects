# Data — 14.02 Spatial Reaction-Diffusion (Gray-Scott)

## Committed sample (`sample/grayscott_params.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** simulation parameters (`scripts/make_synthetic.py`) |
| License | Public domain (CC0) |
| Setup | 128×128 periodic grid, central seed, Gray-Scott labyrinth regime |

The "data" is the **simulation setup**; the initial grid (U=1, V=0, with a central
V seed) is built deterministically from these parameters.

### File format (one line)

```
nx  ny  Du  Dv  F  k  dt  steps  seed_half
```

| Field | Meaning |
|---|---|
| `nx`, `ny` | grid size (periodic boundaries) |
| `Du`, `Dv` | diffusion coefficients of U and V |
| `F` | feed rate; with `k`, selects the pattern (spots/stripes/mazes) |
| `k` | kill rate |
| `dt` | explicit-Euler timestep (keep `dt < 1/(4·max(Du,Dv))` for stability) |
| `steps` | number of timesteps |
| `seed_half` | half-size of the central V-seed square |

Default: `128 128 0.16 0.08 0.0545 0.062 1 8000 8` → a **labyrinth** Turing pattern.
Try `--F 0.0367 --k 0.0649 --seed-half 10` for self-replicating spots.

## "Full dataset" / molecular-resolution RD

The catalog project (14.2) is **particle-based** reaction-diffusion at molecular
resolution — every molecule tracked individually — a 🔴 frontier problem. This
flagship is the **continuum (grid) teaching version**; the real thing uses:

- **ReaDDy** (<https://github.com/readdy/readdy>) — GPU particle-based RD.
- **Smoldyn** (<https://github.com/ssandrews/Smoldyn>) — off-lattice PBRD.
- **MCell** (<https://mcell.org/>) — Monte-Carlo 3-D RD for neurons.
- **STEPS** (<https://github.com/CNS-OIST/STEPS>) — tetrahedral-mesh spatial SSA.

Bigger grid: `python scripts/make_synthetic.py --nx 256 --ny 256 --steps 12000`.

## Provenance & honesty

The configuration is **synthetic**; Gray-Scott is an abstract two-chemical model,
not real cellular biochemistry. It demonstrates the reaction-diffusion / stencil
pattern; it is **not** a molecular simulation and not for any scientific claim.
