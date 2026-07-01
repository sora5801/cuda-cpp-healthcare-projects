# Data — Project 6.18 ECG Forward Problem & Body-Surface Potential Mapping

## What is committed here

A single **tiny, fully synthetic** sample: [`sample/ecg_sample.txt`](sample/ecg_sample.txt).
It is generated deterministically by [`../scripts/make_synthetic.py`](../scripts/make_synthetic.py)
— **no patient data, no real recordings**. It exists so the demo builds, runs, and
verifies offline with zero downloads.

> **SYNTHETIC / not clinical.** This is a geometric toy (a cylindrical "torso"
> with a few current dipoles inside standing in for the heart). It must not be
> used for any diagnostic or clinical purpose.

## File format

Whitespace/newline separated. Blank lines and lines starting with `#` are ignored,
so the file is self-documenting. The layout is:

```
L S T                       # header: #electrodes, #dipole sources, #time frames
<L lines>  x y z             # electrode positions on the body surface (metres)
<S lines>  x y z             # dipole (source) anchor positions inside (metres)
<S lines>  dx dy dz          # dipole unit directions (normalized on load)
<S lines>  s(0) s(1) ... s(T-1)   # each source's strength time series (T frames)
```

Meaning of each field:

| Field | Units | Meaning |
|-------|-------|---------|
| `L`   | count | body-surface electrodes → rows of the transfer matrix `A` and of `Phi` |
| `S`   | count | equivalent current dipoles modelling the heart → columns of `A` |
| `T`   | count | time frames of the activation sequence → columns of `X` and `Phi` |
| electrode `x y z` | metres | electrode location on the torso surface |
| source `x y z`    | metres | fixed dipole anchor inside the torso |
| direction `dx dy dz` | unitless | dipole orientation (a unit vector after loading) |
| strength `s(t)`   | arbitrary (∝ A·m) | the dipole's time-varying moment magnitude |

The default committed sample is `L=8, S=3, T=24`.

## How the synthetic sample is built (and its known answer)

`make_synthetic.py` places `L` electrodes evenly on a ring of a cylindrical torso
(radius 15 cm) and `S` dipoles on a small inner ring shifted toward `+x` (the
"left chest"). Each dipole fires a smooth Gaussian bump at a staggered time (a
crude depolarization sweep); **source 0 is deliberately the strongest** and sits
nearest **electrode 0**. Because the lead field falls off as `1/distance³`, the
electrode nearest the strongest source must record the largest peak-to-peak
deflection — a definite ground truth the demo recovers (see
[`../demo/expected_output.txt`](../demo/expected_output.txt)). Everything is
closed-form (no RNG), so the committed bytes never drift.

Regenerate or scale it:

```bash
python ../scripts/make_synthetic.py                       # default 8/3/24
python ../scripts/make_synthetic.py --L 64 --S 8 --T 500  # bigger synthetic case
```

## Real-world datasets (provenance & licensing)

This project's catalog entry points at these sources. They are **not committed**
(registration-gated and/or far too large — full 3-D torso meshes), and the
download helper only prints instructions; it never bypasses credentials.

| Source | URL | Notes / license |
|--------|-----|-----------------|
| PhysioNet ECG databases | https://physionet.org | Recorded surface ECGs; some sets are credentialed. |
| EDGAR body-surface potential DB | https://edgar.sci.utah.edu *(verify URL)* | Multi-lead body-surface potential maps + torso geometries. |
| Visible Human torso geometry | https://www.nlm.nih.gov/research/visible/visible_human.html | Realistic torso volume-conductor mesh; license/registration. |
| Cardioid (LLNL) ECG module | https://github.com/llnl/cardioid | Reference ECG forward solver (study, don't copy). |
| openCARP ECG lead calculation | https://git.opencarp.org/openCARP/openCARP | ECG post-processing from EP simulations. |

Fetch guidance: [`../scripts/download_data.ps1`](../scripts/download_data.ps1) /
[`../scripts/download_data.sh`](../scripts/download_data.sh). Respect every
dataset's license; do not redistribute credentialed data.
