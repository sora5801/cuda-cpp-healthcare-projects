# Data — 6.1 Cardiac Electrophysiology Simulation

## Committed sample (`sample/tissue_params.txt`)

| Field | Value |
|---|---|
| File | `sample/tissue_params.txt` |
| Origin | **Synthetic** simulation parameters (`scripts/make_synthetic.py`) |
| License | Public domain (CC0) — it is synthetic |
| Size | < 1 KB |
| Setup | 32×32 excitable-tissue sheet, S1 stimulus patch on the left edge |

This project's "data" is the **simulation setup**, not a measured recording — the
electrical wave is *generated* by the solver. The tiny parameter file lets
`demo/run_demo` run **offline, with zero downloads** (a hard requirement,
CLAUDE.md §8).

### File format (one line, 14 whitespace-separated fields)

```
nx  ny  steps  dt  dx  D  a  eps  b  stim_x0  stim_y0  stim_w  stim_h  stim_v
```

| Field | Meaning | Units (dimensionless FHN) |
|---|---|---|
| `nx`, `ny` | tissue grid size (columns × rows) | cells |
| `steps` | number of operator-split timesteps (react + diffuse each) | — |
| `dt` | time step; must satisfy the CFL bound `dt ≤ dx²/(4·D)` | time |
| `dx` | grid spacing (same in x and y) | length |
| `D` | diffusion coefficient (electrotonic coupling) | length²/time |
| `a` | FitzHugh-Nagumo excitation threshold (`0 < a < 1`) | voltage |
| `eps` | FHN recovery time-scale (small ⇒ slow recovery, long AP) | 1/time |
| `b` | FHN recovery coupling (controls the refractory return to rest) | — |
| `stim_x0`, `stim_y0` | top-left corner of the S1 stimulus patch | cell index |
| `stim_w`, `stim_h` | width/height of the stimulus patch | cells |
| `stim_v` | voltage the S1 patch is clamped to at `t = 0` | voltage |

Default sample: `32 32 400 0.1 1.0 0.2 0.1 0.002 0.5 0 0 3 32 1.0` → an
action-potential wave launched from the left edge that has propagated about a
third of the way across the sheet by step 400 (a depolarised plateau on the left,
a sharp wavefront near `x≈11`, resting tissue ahead — visible in the reported
voltage slice). The small `eps=0.002` gives a slow recovery, so a depolarised
plateau trails the front (the classic action-potential shape).

### Units honesty

The FitzHugh-Nagumo model is **nondimensional** — `V`, `w`, `dt`, `dx` are
caricature units, not millivolts/milliseconds. THEORY.md §"real world" explains
how a physiological model (ten Tusscher-Panfilov, O'Hara-Rudy) restores physical
units. Do not read the numbers here as clinical voltages.

## "Full dataset" / realistic models

A validated, patient-specific cardiac EP simulation replaces the synthetic sheet
with (a) a real ionic **cell model** and (b) a real **anatomy**:

- **CellML Physiome Repository** (<https://models.physiomeproject.org>) — curated
  ionic cell models (ten Tusscher, O'Hara-Rudy) in CellML/SBML, importable by openCARP.
- **UK Biobank Cardiac MRI** (<https://www.ukbiobank.ac.uk>) — cine CMR to build
  patient geometries; **access via application** (credentialed).
- **ACDC MICCAI Cardiac Challenge**
  (<https://www.creatis.insa-lyon.fr/Challenge/acdc/>) — CMR with LV/RV/myocardium
  ground truth for segmenting an anatomy.
- **PhysioNet MIT-BIH & MIMIC-III Waveform** (<https://physionet.org>) — ICU
  ECG/hemodynamic waveforms to compare a simulated pseudo-ECG against.

`scripts/download_data.*` prints these links and **never bypasses** the
registration required by UK Biobank / MIMIC. For a bigger synthetic run:
`python scripts/make_synthetic.py --nx 128 --ny 128 --steps 1200`.

## Provenance & honesty

The setup is **synthetic** and the model is a **simplified 2-D FitzHugh-Nagumo
monodomain** (not a physiological ionic model, not a real heart geometry). It
demonstrates the reaction-diffusion / operator-splitting / GPU-stencil pattern;
it is **not** a validated electrophysiology simulation and is **not for any
clinical use**.
