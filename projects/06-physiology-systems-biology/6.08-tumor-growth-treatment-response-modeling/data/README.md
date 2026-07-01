# Data — 6.8 Tumor Growth & Treatment-Response Modeling

## Committed sample (`sample/tumor_params.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** simulation parameters (`scripts/make_synthetic.py`) |
| License | Public domain (CC0) — it is synthetic |
| Size | < 1 KB |
| Setup | 128×128 periodic grid, dx = 0.2 mm, 400 timesteps (100 days), 10 × 2 Gy RT |

The "data" is the **simulation setup**, not a measurement. The initial tumor (a
small seed disc at the grid centre) and the whole density field are built
**deterministically** by the program from these parameters, so the demo runs
**offline with zero downloads** (CLAUDE.md §8).

### File format (one whitespace-separated line)

```
nx ny dx D rho dt steps alpha beta dose n_fractions fx_interval seed_radius seed_u
```

| Field | Meaning | Units |
|---|---|---|
| `nx`, `ny` | grid size (periodic boundaries) | cells |
| `dx` | spacing between adjacent cells | mm |
| `D` | tumor-cell diffusion (infiltration) coefficient | mm²/day |
| `rho` | net proliferation rate (logistic growth) | 1/day |
| `dt` | explicit-Euler timestep (must satisfy `dt ≤ dx²/(4D)`) | day |
| `steps` | number of growth timesteps | — |
| `alpha` | LQ linear radiosensitivity | 1/Gy |
| `beta` | LQ quadratic radiosensitivity | 1/Gy² |
| `dose` | dose per radiotherapy fraction | Gy |
| `n_fractions` | number of fractions delivered (`0` = untreated control) | — |
| `fx_interval` | timesteps between fractions | steps |
| `seed_radius` | radius of the initial tumor seed disc | mm |
| `seed_u` | initial density inside the seed (0..1) | — |

Default: `128 128 0.2 0.02 0.15 0.25 400 0.15 0.015 2 10 20 1 1` — a diffuse,
proliferating tumor treated with a schematic **10 × 2 Gy** course (α/β = 10 Gy).
The program additionally runs an **untreated control** internally (fractions
forced to 0) so the report can quote the modelled treatment response.

## "Full dataset" — real-data calibration pointers

This is a **teaching model**, not a calibrated patient model. Real mathematical
oncology calibrates D, ρ, α, β against imaging and omics:

- **TCGA** — The Cancer Genome Atlas (multi-omics + imaging): <https://portal.gdc.cancer.gov>
- **TCIA** — The Cancer Imaging Archive (multi-institutional tumor imaging): <https://www.cancerimagingarchive.net>
- **PhysioNet** — oncology waveforms and clinical time series: <https://physionet.org>
- **Zenodo** — search "tumor growth simulation" for published simulation datasets.

None of these are required to run the demo, and the `scripts/download_data.*`
helpers only **print instructions and links** — they never bypass any
registration or credentials (CLAUDE.md §8).

Bigger / different runs (no download):

```
python scripts/make_synthetic.py --nx 256 --ny 256 --steps 800
python scripts/make_synthetic.py --dose 3 --n-fractions 10     # hypofractionation
python scripts/make_synthetic.py --n-fractions 0               # control only
```

## Provenance & honesty

The configuration is **synthetic** and the model is a deliberately reduced
teaching version (single density field; LQ kill as an instantaneous per-cell
multiply; no explicit oxygen/hypoxia or drug PK/PD field). It is **not**
calibrated to any patient and is **not for clinical use** — it demonstrates the
Fisher-KPP reaction-diffusion + linear-quadratic radiobiology mathematics on the
GPU stencil pattern. See `THEORY.md` for what a production model adds.
