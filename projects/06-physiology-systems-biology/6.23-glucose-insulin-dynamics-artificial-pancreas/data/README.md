# Data — 6.23 Glucose-Insulin Dynamics & Artificial Pancreas

## Committed sample (`sample/cohort_params.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** cohort configuration (`scripts/make_synthetic.py`) |
| License | Public domain (CC0) |
| Contents | One line: fixed Bergman/meal/PID settings + the SI×SG patient sweep |
| Size | < 200 bytes |

The "data" is the **in-silico trial setup**, not a measured signal. Each virtual
patient's parameters are derived from the sweep grid, so the whole cohort is
reproducible from one line of numbers. **This is synthetic and not clinically
valid** — the constants are illustrative teaching values loosely in the range of
published Bergman minimal-model fits, not fitted to any real subject.

### File format (one whitespace-separated line, 26 values)

```
p2 n Gb Ib VG VI  meal_D meal_Ag meal_k meal_t  G_target Kp Ki Kd
u_basal u_max control_dt  G0 dt steps  nSI nSG  p3_lo p3_hi  p1_lo p1_hi
```

| Field | Meaning | Units |
|---|---|---|
| `p2` | remote-insulin action decay rate | 1/min |
| `n` | plasma-insulin clearance rate | 1/min |
| `Gb`, `Ib` | basal (fasting) glucose / insulin set-points | mg/dL, µU/mL |
| `VG`, `VI` | glucose / insulin distribution volumes (scale meal & pump) | dL, — |
| `meal_D` | meal carbohydrate load (mg glucose; 50000 ≈ 50 g) | mg |
| `meal_Ag` | meal bioavailability (0–1) | — |
| `meal_k` | gut-absorption rate (Ra peaks at `meal_t + 1/k`) | 1/min |
| `meal_t` | meal start time | min |
| `G_target` | PID glucose set-point | mg/dL |
| `Kp`, `Ki`, `Kd` | PID controller gains | — |
| `u_basal`, `u_max` | basal / maximum insulin infusion | — |
| `control_dt` | controller update period (multiple of `dt`) | min |
| `G0` | initial glucose | mg/dL |
| `dt`, `steps` | RK4 timestep and step count (run = `steps·dt` min) | min, — |
| `nSI`, `nSG` | cohort grid: `nSI` insulin-sensitivity × `nSG` glucose-effectiveness | — |
| `p3_lo..hi` | insulin-action gain range (insulin sensitivity `SI = p3/p2`) | 1/min·(µU/mL)⁻¹ |
| `p1_lo..hi` | glucose-effectiveness (`SG`) range | 1/min |

Default: `... 32 32 1e-05 4e-05 0.018 0.028` → **1024 virtual patients**, an 8-hour
run, with `SI = p3/p2` sweeping 0.0004 → 0.0016 (insulin-resistant → insulin-sensitive).

## "Full dataset" — real CGM/insulin data and reference simulators

This project is a **simulator**, so there is nothing to download to run it. For
real data and the FDA-accepted reference model, see `scripts/download_data.ps1`:

- **OhioT1DM** (<https://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html>) — 12-week
  CGM + insulin for 12 T1D subjects (requires a data-use agreement; the script
  does **not** bypass it).
- **JAEB / DirecNet** (<https://public.jaeb.org>) — public CGM datasets.
- **simglucose** (<https://github.com/jxx123/simglucose>) — Python UVA/Padova
  T1D simulator + RL gym environment.
- **GluCoEnv** (<https://github.com/chirathyh/GluCoEnv>) — GPU-accelerated glucose
  control RL environment.

Bigger synthetic cohort: `python scripts/make_synthetic.py --nSI 64 --nSG 64` (4096 patients).

## Provenance & honesty

The configuration is **synthetic**; the parameters are illustrative, not fitted
to any patient. The output is a software demonstration of ensemble closed-loop
ODE simulation — **not for diagnosis, treatment, or any real medical decision**.
