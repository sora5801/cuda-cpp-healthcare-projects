# Data — 4.12 Optical Coherence Tomography Processing (SD-OCT)

## Committed sample (`sample/oct_bscan.txt`)

| Field | Value |
|---|---|
| File | `sample/oct_bscan.txt` |
| Origin | **Synthetic** raw SD-OCT spectra (`scripts/make_synthetic.py`, seed 7) |
| License | Public domain (CC0) — it is synthetic |
| Size | ~75 KB |
| Contents | 32 A-scans × 256 spectral samples; injected dispersion `a2=18, a3=9` |

This tiny file lets `demo/run_demo` run **offline, with zero downloads** (a hard
requirement, CLAUDE.md §8).

### File format

```
<n_ascan> <n_spec> <a2> <a3>      # A-scans, samples/A-scan (= FFT length N), dispersion coeffs
<A-scan 0: n_spec raw spectrum values>
<A-scan 1: ...>
... (n_ascan rows)
```

- **`n_ascan`** — number of A-scans (lateral pixels) in the B-scan.
- **`n_spec`** — spectral samples per A-scan; also the FFT length `N` (must be even).
- **`a2`, `a3`** — 2nd/3rd-order dispersion coefficients injected into the raw
  spectra; the reconstruction removes exactly this phase (numerical dispersion
  compensation), which is why the reconstructed peaks are sharp.
- Each **row** is one A-scan's raw interferometric spectrum: reflectors at known
  depths appear as cosine fringes (fringe frequency ∝ depth) on a DC offset, plus
  low-level detector noise. See `scripts/make_synthetic.py` for the forward model.

The sample encodes a gently curved bright "surface" reflector plus two dimmer
deeper layers, so the reconstruction has an obvious right answer (the demo
recovers the surface arc as the per-A-scan peak depth).

## Full dataset

Public OCT datasets provide **reconstructed** B-scan images/volumes (for
segmentation and classification), not the vendor **raw spectra** this project
reconstructs from — raw interferograms are device-specific and usually behind an
OCT device SDK. `scripts/download_data.ps1` / `.sh` point at these and download
nothing:

- **OCTDL** (<https://www.nature.com/articles/s41597-024-03182-7>) — 2,064 labeled OCT B-scans.
- **Duke DME OCT** (<https://people.duke.edu/~sf59/Chiu_BOE_2012_dataset.htm>) — 110 annotated volumes.
- **OCTA-500** (<https://arxiv.org/abs/2012.07261>) — OCT angiography volumes with labels.

For a larger synthetic B-scan: `python scripts/make_synthetic.py --n-ascan 128 --n-spec 1024`.

## Provenance & honesty

The sample is **synthetic** and labeled as such — reflectors at known depths with
injected dispersion and noise, not a real recording, and of no diagnostic meaning.
It exists to make the reconstruction result interpretable and exactly verifiable.
Respect every dataset's license; raw-spectrum access via a device SDK is governed
by the vendor's terms and this repo does not redistribute it.
