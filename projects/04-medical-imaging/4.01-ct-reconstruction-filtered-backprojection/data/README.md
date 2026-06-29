# Data — 4.01 CT Reconstruction (Filtered Backprojection)

## Committed sample (`sample/sinogram_sample.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** (`scripts/make_synthetic.py`) — analytic sinogram of a disc phantom |
| License | Public domain (CC0) — synthetic |
| Geometry | 120 projection angles over [0, π), 183 detector bins, spacing 0.012 |
| Reconstruction | 128×128 image over world [-0.75, 0.75]² |

### File format

```
<n_angles> <n_det> <ds> <img> <world_half>     # header line
<row 0: n_det projection values>               # projection at angle 0
<row 1: ...>
... (n_angles rows)
```

- Angles are uniform: `theta_k = k·π/n_angles`.
- Detector bin `j` is at offset `s_j = (j − (n_det−1)/2)·ds`.
- Each value is a **line integral** (Radon transform) of the phantom along that
  ray. Because the phantom is a sum of uniform discs, the integral is analytic:
  chord length `2·sqrt(r²−(s−c)²)` times density, summed over discs.

## Full dataset

Real CT data is a **measured sinogram** (or a standard digital phantom):

- **Shepp-Logan phantom** — the canonical CT test object (built into ASTRA/TIGRE).
- **TCIA** (The Cancer Imaging Archive) — real DICOM CT projection/volume data:
  <https://www.cancerimagingarchive.net>
- **Reconstruction toolkits** with example data: RTK, ASTRA, TIGRE (see README "Prior art").

`scripts/download_data.ps1` / `.sh` describe how to obtain a phantom/sinogram. For
a larger synthetic problem: `python scripts/make_synthetic.py --angles 360 --det 367 --img 256`.

## Provenance & honesty

The sample is **synthetic** and labeled as such. Reconstructed values are in
arbitrary density units (the disc densities of the phantom); this is a software
demonstration, not a calibrated CT image and not for any clinical use.
