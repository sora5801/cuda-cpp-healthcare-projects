# Data — 5.7 Brachytherapy Dose & Source Modeling

## Committed sample (`sample/plan_sample.txt`)

| Field | Value |
|---|---|
| File | `sample/plan_sample.txt` |
| Origin | **Synthetic** (hand-crafted teaching values; **not** an AAPM consensus dataset) |
| License | Public domain (CC0) — it is synthetic |
| Size | < 2 KB |
| Contents | one Ir-192-*like* line source, its TG-43 tables, a 41×41×1 dose grid, 3 dwells |

This tiny file lets `demo/run_demo` run **offline, with zero downloads** (a hard
requirement, CLAUDE.md §8). It is deliberately engineered so the result is
*interpretable* (PATTERNS.md §6): the three dwells sit at the grid center, so the
dose peaks there and falls off ~1/r² outward — visible directly in the printed
center-row profile.

> **Synthetic, not clinical.** The numbers below are plausible in *shape* but are
> not measured and describe no real, named source. Do not use them for anything
> but learning. `src/main.cu`'s built-in fallback plan is byte-identical to this
> file, so the program prints the same result with or without it.

### File format (whitespace text; `#` and blank lines ignored)

```
L Lambda                       # source active length [cm], dose-rate const [cGy/(h·U)]
n_g                            # number of radial-dose-function samples
r  g   (× n_g lines)           # g_L(r): radius [cm], value (dimensionless, g_L(1cm)=1)
n_Fr n_Ft                      # anisotropy grid: #radii, #angles
F_r_1 … F_r_{n_Fr}             # the anisotropy radii   [cm]   (one line)
F_t_1 … F_t_{n_Ft}             # the anisotropy angles  [deg]  (one line, 0..180)
F row for r_1  (× n_Fr lines)  # F(r,θ) values, n_Ft per line, row-major
nx ny nz  ox oy oz  spacing    # dose grid: counts, origin (voxel(0,0,0) center) [cm], pitch [cm]
n_dwells                       # number of dwell positions
x y z weight  (× n_dwells)     # dwell center [cm] and weight (= relative dwell time × S_K) [U·h]
```

### Per-field meaning (the TG-43 quantities)

- **`L`** — active source length. The line-source geometry function `G_L` uses it;
  `L→0` recovers a mathematical point source (`1/r²`).
- **`Lambda` (Λ)** — dose-rate constant: dose rate at the reference point
  `(r₀=1 cm, θ₀=90°)` per unit air-kerma strength. Source-specific.
- **`g_L(r)`** — radial dose function: attenuation + scatter buildup in water along
  the transverse axis, normalized so `g_L(1 cm)=1`. Interpolated linearly.
- **`F(r,θ)`** — 2-D anisotropy function: dose falloff away from the transverse
  plane (source/capsule self-absorption). `F(r,90°)=1`. Bilinearly interpolated.
- **dwell `weight`** — folds the relative dwell time and the source air-kerma
  strength into one number; optimizing these is *inverse planning* (project 5.2).

## Full / real datasets

The real TG-43 inputs are **published consensus source tables**, not a single
downloadable blob. `scripts/download_data.ps1` / `.sh` print where to obtain them
and never bypass any registration:

- **AAPM TG-43U1 consensus source data** — the canonical `Λ`, `g_L(r)`, and
  `F(r,θ)` tables per source model (Ir-192 HDR, Pd-103, I-125, …):
  <https://www.aapm.org/pubs/reports/>. Transcribe a source's tables into the
  format above to run this project on real consensus data.
- **ESTRO ACROP** brachytherapy guideline test cases (planning geometry).
- **TCIA** prostate brachytherapy CT datasets (imaging context; free
  registration): <https://www.cancerimagingarchive.net/>.

Catalog dataset notes (verbatim):

> AAPM TG-43 consensus datasets (radial/anisotropy tables — https://www.aapm.org/pubs/reports/); TCIA prostate BT CT datasets; ESTRO ACROP BT guideline test cases; BrachyView QA data (verify URL).

## Regenerating the sample

```
python scripts/make_synthetic.py            # rewrite data/sample/plan_sample.txt
python scripts/make_synthetic.py --grid 81  # a larger 81×81 grid
python scripts/make_synthetic.py --dwells 7 # more dwell positions
```

The default output parses to the same values `src/main.cu` builds in, so the demo
stays reproducible. Everything here is labeled **synthetic**; no clinical validity
is implied.
