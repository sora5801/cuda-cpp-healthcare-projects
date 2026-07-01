# Data — 4.2 Iterative / Model-Based CT Reconstruction

## Committed sample (`sample/sinogram_sample.txt`)

| Field | Value |
|---|---|
| File | `sample/sinogram_sample.txt` |
| Origin | **Synthetic** — a disc phantom forward-projected + noise (see `scripts/make_synthetic.py`) |
| License | Public domain (CC0) — it is entirely synthetic, no patient data |
| Size | ~52 KB |
| Contents | a noisy sinogram **and** the ground-truth phantom image |

This tiny file lets `demo/run_demo` run **offline, with zero downloads** — a hard
requirement for every project (CLAUDE.md §8). It is **synthetic**: a known disc
phantom (a large soft-tissue disc with bright and one "cold" insert) is
forward-projected with the same voxel-driven model the reconstruction uses, then
Gaussian noise (fixed RNG seed, so the file is reproducible) is added. Because the
true phantom is shipped alongside the sinogram, the demo can report reconstruction
error vs. truth — a scientific check on top of the CPU-vs-GPU check.

### File format (parsed by `load_ct` in `src/reference_cpu.cpp`)

```
line 1 (header):  n_angles  n_det  ds  img  world_half  iters  lambda  tv_weight  has_truth
next n_angles lines:  n_det floats each   -> the measured (noisy) sinogram, row k = angle k
if has_truth==1,
next img lines:       img   floats each   -> the ground-truth image, row py
```

Field meanings and units:

| Field | Meaning | Units |
|---|---|---|
| `n_angles` | number of projection angles, uniform over `[0, π)` | count |
| `n_det` | detector bins per projection | count |
| `ds` | detector bin spacing | world units |
| `img` | reconstructed image side length | pixels |
| `world_half` | image spans `[-world_half, +world_half]` in x and y | world units |
| `iters` | SIRT iterations the demo runs | count |
| `lambda` | SIRT relaxation / step size (`0 < λ ≤ 2`) | dimensionless |
| `tv_weight` | strength of the TV smoothing step (`0` = pure SIRT) | dimensionless |
| `has_truth` | `1` if a ground-truth image follows the sinogram, else `0` | flag |
| sinogram values | line integrals `−ln(I/I₀)` (with noise) | attenuation·length |
| truth values | phantom attenuation `μ` per pixel | dimensionless here |

Regenerate (deterministic): `python scripts/make_synthetic.py`.

## Full / real datasets

Real low-dose CT data exists but is **credentialed** and **not redistributable**
here, so the committed sample is synthetic and `scripts/download_data.*` only
prints instructions (it never bypasses registration). From the catalog:

- **2016 AAPM Low-Dose CT Grand Challenge** — <https://www.aapm.org/grandchallenge/lowdosect/>
  — paired low/normal-dose scans; requires registration.
- **Mayo Clinic Low-Dose CT** — via **TCIA** (The Cancer Imaging Archive).
- **LIDC-IDRI** — via **TCIA** — <https://www.cancerimagingarchive.net/> — CT lung
  scans with a data-use agreement.

**License:** respect each dataset's data-use agreement. None of them are committed
to this repo. Never present any output here as clinically valid — this is study
material only, and the sample is synthetic everywhere it appears.
