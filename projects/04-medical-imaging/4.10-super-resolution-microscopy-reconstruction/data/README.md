# Data — 4.10 Super-Resolution Microscopy Reconstruction

## Committed sample (`sample/smlm_stack.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** STORM/PALM movie (`scripts/make_synthetic.py`, seed 7) |
| License | Public domain (CC0) — it is synthetic |
| Contents | 60 frames × 40×40 px; sparse Gaussian PSFs on two crossing sub-pixel lines |
| Size | ~660 KB |

### File format

```
<F> <H> <W> <background> <threshold>      # header: frames, height, width, bg, detect cutoff
<frame 0: H*W floats, row-major>          # pixel intensities, one frame ...
<frame 1: H*W floats>                     # ... after another
... (F frames)
```

- **F, H, W** — number of frames and each frame's pixel dimensions.
- **background** — flat level subtracted from every pixel during localization
  (camera offset + out-of-focus haze). In the sample: `40.0`.
- **threshold** — a candidate pixel must exceed this to be a detection. In the
  sample: `100.0` (well above background + noise, below an on-emitter's peak).
- **pixels** — intensities in arbitrary detector units, written at 3-decimal
  precision so the C++ loader reads identical values every run (keeping CPU==GPU
  verification exact).

### How the sample is engineered (so the demo is interpretable)

Two thin lines are drawn at **sub-pixel** positions (a stand-in for microtubules).
In each frame, each site on those lines blinks **on** with low probability, so a
frame contains a sparse scatter of separated Gaussian blobs. The localizer fits
each blob's centre to a fraction of a pixel; overlaying ~187 localizations across
the 60 frames renders the two lines far sharper than any single diffraction-
limited frame — the essence of super-resolution.

## Full dataset

Real SMLM data are large multi-frame TIFF/OME-TIFF stacks:

- **EPFL SMLM Challenge** (<https://srm.epfl.ch/srm/dataset/challenge-2016/>) —
  synthetic **and** real STORM/PALM frames with ground-truth positions (ideal for
  scoring a localizer). 
- **BioImage Archive** (<https://www.ebi.ac.uk/biostudies/bioimages>) — public
  SMLM collections.
- **OME-TIFF** (<https://www.openmicroscopy.org/ome-files/>) — the standard movie
  container; read with `tifffile` (Python), Fiji/ImageJ, or ThunderSTORM.

Export each frame's pixels into the header+floats format above (see
`scripts/download_data.*`). Bigger synthetic movie:
`python scripts/make_synthetic.py --frames 200 --width 64 --height 64`.

## Provenance & honesty

The sample is **synthetic** blinking-emitter data, not a real microscope
acquisition, and carries no clinical or scientific claim about any specimen. It
exists to make the reconstruction interpretable (the two lines are recovered) and
the GPU/CPU comparison verifiable.
