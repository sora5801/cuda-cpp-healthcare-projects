# Data — 4.26 Vessel Segmentation & Centerline Extraction

## Committed sample (`sample/vessel_volume.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** 3-D volume (`scripts/make_synthetic.py`, fixed seed 42) |
| License | Public domain (CC0) — it is synthetic |
| Size | ~54 KB, 24 × 16 × 16 = 6144 voxels |
| Contents | one bright cylindrical "vessel" along the x-axis + seeded noise |

This project's input is a small **3-D intensity volume**. The committed sample
embeds a single **known** structure — a straight bright tube of radius 2 voxels
running along x through the (y, z) center — so the Frangi filter's response is
**interpretable and checkable**: vesselness should peak on the tube axis, be
near-zero away from it, and a threshold should recover roughly the tube's
cross-section. That is exactly what the demo output shows.

The tiny file lets `demo/run_demo` run **offline, with zero downloads** — a hard
requirement for every project (CLAUDE.md §8).

### File format

```
line 1:  nx ny nz sigma alpha beta c bright mask_threshold
then  :  nx*ny*nz float intensities, row-major (x fastest, then y, then z)
```

| Field | Meaning |
|---|---|
| `nx ny nz` | volume dimensions (x fastest in memory) |
| `sigma` | Gaussian pre-smoothing scale, in voxels (the vessel radius the filter targets) |
| `alpha` | Frangi sensitivity to `R_A = |λ2|/|λ3|` (plate-vs-line) |
| `beta` | Frangi sensitivity to `R_B = |λ1|/√(|λ2 λ3|)` (blob-vs-line) |
| `c` | Frangi "structureness" scale (suppresses response in flat noise) |
| `bright` | `1` = vessels brighter than background (CTA); `0` = darker |
| `mask_threshold` | vesselness ≥ this counts as a segmented vessel voxel |

Default sample header: `24 16 16 1.5 0.5 0.5 15.0 1 0.5`. The value of `c` is
scaled to this synthetic intensity range (peak amplitude ~200); on real Hounsfield
data you would re-tune `c` (see THEORY §4).

## Full / real datasets

The real vessel datasets need registration or a challenge sign-up, so
`scripts/download_data.*` only prints links and instructions (it never bypasses
credentials, CLAUDE.md §8). To use one, register, export a volume to NIfTI, and
convert it to the plain-text format above (a converter is a README exercise).

- **ASOCA** — coronary CTA segmentation challenge: <https://asoca.grand-challenge.org/>
- **ImageCAS** — 1000 coronary CTAs: <https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT>
- **3D-IRCADb-01** — abdominal/liver vasculature: <https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/>

Bigger synthetic volume:
`python scripts/make_synthetic.py --nx 128 --ny 96 --nz 96 --radius 4`.

## Provenance & honesty

The committed volume is **synthetic** and geometrically trivial (one straight
tube). It demonstrates the Frangi-vesselness GPU pattern; it is **not** a real
angiogram, is not validated, and is **not for any clinical use**.
