# Data — 9.02 Large-Scale Compartmental & Metapopulation Models

## Committed sample (`sample/ensemble_params.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** ensemble configuration (`scripts/make_synthetic.py`) |
| License | Public domain (CC0) |
| Contents | One line of population/IC/integration settings + a beta×gamma sweep |

The "data" is the **ensemble setup**, not measured input; each member's
parameters are derived from the sweep grid, so the whole ensemble is reproducible.

### File format (one line)

```
N  I0  dt  steps  sigma  nb  ng  beta_lo  beta_hi  gamma_lo  gamma_hi
```

| Field | Meaning |
|---|---|
| `N` | total population (constant) |
| `I0` | initial infectious count (`S0 = N − I0`, `E0 = R0 = 0`) |
| `dt`, `steps` | RK4 timestep (days) and step count (run = `steps·dt` days) |
| `sigma` | E→I rate = 1 / latent period |
| `nb`, `ng` | sweep size: `nb` beta values × `ng` gamma values = `nb·ng` members |
| `beta_lo..hi` | transmission-rate range |
| `gamma_lo..hi` | recovery-rate range (1/infectious period) |

Default: `1000000 10 0.25 720 0.192308 64 64 0.15 0.6 0.1 0.5` → 4096 members, 180
days, R0 = β/γ from 0.30 to 6.0.

## "Full dataset" / realistic models

Real epidemic modelling adds measured contact/mobility matrices, age structure,
seasonal forcing, and many geographic patches:

- **MEmilio** (<https://github.com/SciCompMod/memilio>) — high-performance C++/CUDA epidemic simulation.
- **EpiModel** (<https://github.com/EpiModel/EpiModel>) — network compartmental modelling (R).
- **Torchdiffeq** (<https://github.com/rtqichen/torchdiffeq>) — GPU ODE solvers (`dopri5`).
- Mobility data: census commuting flows, GLEAM, mobile-phone-derived matrices.

Bigger ensemble: `python scripts/make_synthetic.py --nb 200 --ng 200` (40,000 members).

## Provenance & honesty

The configuration is **synthetic**; the parameter ranges are illustrative, not
fitted to any disease. Outputs are a software demonstration of ensemble ODE
integration, not an epidemic forecast.
