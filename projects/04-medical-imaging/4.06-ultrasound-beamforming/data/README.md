# Data — 4.6 Ultrasound Beamforming

## Committed sample (`sample/rf_sample.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** (`scripts/make_synthetic.py`) — simulated RF echoes from a point scatterer |
| License | Public domain (CC0) — synthetic |
| Size | ~210 KB |
| Array | 64-element linear probe, 0.3 mm pitch, centred on x = 0 |
| RF | 384 samples/element at fs = 40 MHz, acquisition window starting at t0 = 24 µs |
| Medium | speed of sound c = 1540 m/s (soft tissue) |
| Image grid | 96 × 96 pixels over x ∈ [−10, +10] mm, z ∈ [+5, +40] mm |
| Embedded truth | **one point scatterer at (x, z) = (4.0, 20.0) mm** |

### File format

```
<n_elements> <n_samples> <nx> <nz> <fs> <c> <pitch> <x_min> <z_min> <dx> <dz> <t0>   # header
<row 0: n_samples RF values>      # element 0's fast-time trace
<row 1: ...>
... (n_elements rows)
```

- **Coordinates** (SI, metres & seconds): the probe lies on the line `z = 0`;
  element `e` is at `x_e = (e − (n_elements−1)/2)·pitch`. The image is in the
  `(x, z)` plane, `x` lateral and `z` depth (into the body). Pixel `(ix, iz)` is
  at `x = x_min + ix·dx`, `z = z_min + iz·dz`.
- **RF value** `rf[e·n_samples + t]` is element `e`'s recorded signal at time
  `t0 + t/fs` seconds after transmit. Each value is the **sum of the (delayed,
  attenuated) transmit pulse** over all scatterers — a short Gaussian-windowed
  5 MHz cosine (the standard band-limited ultrasound pulse). See
  `scripts/make_synthetic.py` for the exact forward model.
- **t0 ≠ 0 on purpose:** a real scanner records a depth-gated window, not from
  `t = 0`. Starting at `t0 = 24 µs` skips the long silent pre-echo stretch so the
  committed file stays small; the loader and beamformer subtract `t0` everywhere.

### Why this sample is verifiable

Because we KNOW the scatterer is at `(4.0, 20.0) mm`, a correct delay-and-sum
beamformer must focus all 64 element echoes back onto that point. The demo's
"brightest pixel" lands at `(3.9, 20.1) mm` — within one pixel of the truth —
which is human-readable proof the beamforming worked (on top of the GPU-vs-CPU
agreement check).

## Full dataset

Real ultrasound RF data comes from a scanner or a wave simulator:

- **PICMUS** — the canonical plane-wave RF challenge set (point targets, cysts,
  in-vivo): <https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/>
  (registration may be required; we never bypass it).
- **Field II** (<https://field-ii.dk/>) — a CPU simulator that generates
  realistic RF data for arbitrary phantoms; export to the format above and
  beamform with this project's GPU kernel.
- **k-Wave / k-Wave-Fluid-CUDA**, **MUST** — see README "Prior art".

`scripts/download_data.ps1` / `.sh` print these sources and instructions. For a
larger synthetic problem:
`python scripts/make_synthetic.py --elements 128 --samples 512 --nx 192 --nz 192 --extra`.

## Provenance & honesty

The sample is **synthetic** and labeled as such. The point-scatterer forward
model is a teaching simplification (no diffraction, no aberration, no
multiple-scattering, no electronic noise). Reconstructed values are in arbitrary
units; this is a software demonstration, **not** a calibrated clinical image and
**not for any clinical use**.
