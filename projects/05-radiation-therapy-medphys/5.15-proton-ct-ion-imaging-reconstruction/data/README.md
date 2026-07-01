# Data — 5.15 Proton CT & Ion Imaging Reconstruction

## Committed sample (`sample/protons_sample.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** (`scripts/make_synthetic.py`) — list-mode protons through a known RSP phantom |
| License | Public domain (CC0) — it is synthetic |
| Size | ~115 KB (1440 protons + a 32×32 ground-truth RSP grid) |
| Phantom | water disc (RSP 1.0) + dense "bone" insert (RSP 1.6) + light "lung" insert (RSP 0.3) |
| Geometry | 45 projection angles × 32 parallel protons; 32×32 grid over world [-5, 5]² cm |
| Solver knobs | 40 SART sweeps, relaxation 0.80, 64 MLP quadrature samples/proton |

### File format

```
<n> <half> <iters> <relax> <path_samples> <n_protons>     # header line
<n*n ground-truth RSP floats, row-major>                  # the known phantom
x0 y0 x1 y1 entry_angle exit_angle wepl                    # one row per proton
...  (n_protons rows)
```

- `n`, `half` — the reconstruction grid: `n×n` voxels over `[-half, half]²` (cm).
- `iters`, `relax`, `path_samples` — SART controls read by the solver.
- The **ground-truth block** is the phantom the data was generated from. The
  solver never reads it; the demo uses it only to report reconstruction error
  (RMSE) — i.e. "did we recover the known answer" (see `docs/PATTERNS.md` §6).
- Each **proton row**: entry point `(x0,y0)`, exit point `(x1,y1)` in cm; entry
  and exit **scattering angles** (radians, relative to the entry→exit chord);
  and the measured **WEPL** (water-equivalent path length, cm) = the line
  integral of RSP along the proton's most-likely path (MLP).

The WEPL is computed by integrating the phantom's RSP along the **same** cubic
Hermite MLP and nearest-voxel sampling the C++ reconstructor uses, so the
synthetic measurements are consistent with the forward model and SART recovers
the phantom.

## Full dataset

Real proton-CT list-mode data comes from prototype scanners and Monte-Carlo
simulation. None are needed to run this demo, but to work with real data:

- **PRaVDA** and **PRIMA** proton-CT consortia — prototype-scanner datasets
  (availability varies; *verify current URLs*, registration may be required).
- **TOPAS** (<https://github.com/OpenTOPAS/OpenTOPAS>) / **GATE** — Monte-Carlo
  toolkits that can simulate a pCT scan and export list-mode entry/exit tracks
  and residual range; convert those to the format above.
- **ACE collaboration** proton-CT phantom datasets (*verify URL*).

`scripts/download_data.ps1` / `.sh` print these pointers and never bypass any
registration. For a larger synthetic problem:
`python scripts/make_synthetic.py --n 48 --angles 90 --rays 48`.

## Provenance & honesty

The sample is **synthetic** and labeled as such. RSP values are relative to
water (dimensionless); reconstructed values are a software demonstration, **not
a calibrated clinical image and not for any clinical use**. Reconstruction is a
reduced-scope 2-D teaching version (see `THEORY.md` "Where this sits in the real
world").
