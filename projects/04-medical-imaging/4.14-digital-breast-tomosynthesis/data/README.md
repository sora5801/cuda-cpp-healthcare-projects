# Data — 4.14 Digital Breast Tomosynthesis

## Committed sample (`sample/dbt_sample.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** (`scripts/make_synthetic.py`) — a phantom forward-projected over a narrow angular wedge |
| License | Public domain (CC0) — synthetic, generated locally |
| Size | ~13 KB |
| Geometry | 15 projections over +/-25 deg, 96 detector bins, spacing ds ≈ 0.0232 |
| Reconstruction | 64×64 image over world [-1, 1]², SART with 20 iterations, relaxation λ = 0.30 |
| Phantom | fibroglandular ellipse (μ += 0.20) + two dense lesion discs (μ += 0.80 and 0.60) |

### File format

```
<n_angles> <n_det> <ds> <img> <world_half> <half_span> <relax> <n_iters>   # header
<row 0: n_det projection values>                                          # projection at angle 0
<row 1: ...>
... (n_angles rows)
```

- **n_angles** — number of projections (DBT uses ~9–25; here 15).
- **n_det** — detector bins per projection.
- **ds** — detector bin spacing in world units; bin `j` is at signed offset
  `s_j = (j − (n_det−1)/2)·ds`.
- **img** — reconstructed image side length in pixels.
- **world_half (W)** — the image covers world square `[−W, W]²`.
- **half_span** — HALF the angular wedge in **radians** (0.4363 ≈ 25°). Angles are
  `θ_k = −half_span + k·(2·half_span/(n_angles−1))`, symmetric about the straight-down view.
- **relax (λ)** — SART relaxation factor (0 < λ ≤ 1); larger converges faster but
  can oscillate.
- **n_iters** — number of SART sweeps (fixed, so the demo output is deterministic).
- Each projection value is a **line integral** (Radon transform) of the phantom's
  attenuation along that ray, in arbitrary units.

The tiny sample lets `demo/run_demo` run **offline with zero downloads**, a hard
requirement for every project (CLAUDE.md §8). Regenerate or scale it with, e.g.,
`python scripts/make_synthetic.py --img 128 --angles 21 --det 160`.

## Full / real datasets

Real DBT projection and mammography data (from the catalog):

- **OPTIMAM (OMI-DB)** — large UK mammography database, access via ICR UK (credentialed).
- **CBIS-DDSM** — 2,620 curated mammograms via TCIA:
  <https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM>
- **VinDr-Mammo** — annotated mammography, PhysioNet (credentialed):
  <https://physionet.org/content/vindr-mammo/1.0.0/>
- **BCS-DBT** — Duke DBT challenge dataset (true tomosynthesis projections):
  <https://bcs-dbt.grand-challenge.org/>

`scripts/download_data.ps1` / `.sh` print how to obtain these. Credentialed sets
(OPTIMAM, VinDr-Mammo) require registration; the scripts **never** bypass it —
they print instructions and links only, and defer to `make_synthetic.py` for an
offline stand-in.

## Provenance & honesty

The committed sample is **synthetic** and labeled as such. Reconstructed values
are in **arbitrary attenuation units** (the phantom's chosen densities); this is a
software demonstration of the reconstruction math, **not** a calibrated image and
**not for any clinical use**.
