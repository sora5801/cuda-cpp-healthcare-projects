# Data — 4.23 Arterial Spin Labeling & Perfusion Imaging

## Committed sample (`sample/asl_sample.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** multi-delay ASL study (`scripts/make_synthetic.py`) |
| License | Public domain (CC0) — it is synthetic |
| Size | < 1 KB |
| Contents | 6 voxels × 7 post-labeling delays (PLDs), noise-free Buxton curves |

This tiny file lets `demo/run_demo` run **offline, with zero downloads**, a hard
requirement for every project (CLAUDE.md §8). Each voxel is a small tissue type
with a **known ground-truth** cerebral blood flow (CBF) and arterial transit time
(ATT); the committed signal is the noise-free perfusion-weighted difference
signal ΔM(PLD) that those parameters produce under the Buxton model. Because we
know the truth, the demo can check that the fit **recovers** it — the "embed a
known answer" idiom (docs/PATTERNS.md §6).

### File format

```
line 1:  n_voxels  n_plds  max_iters  f_init  att_init
line 2:  pld_0 pld_1 ... pld_{n_plds-1}                 (post-labeling delays, s)
then n_voxels lines:  true_cbf  true_att  s_0 s_1 ... s_{n_plds-1}
```

| Field | Meaning | Units |
|---|---|---|
| `n_voxels` | number of voxels to fit | — |
| `n_plds` | number of post-labeling delays per voxel | — |
| `max_iters` | Gauss-Newton iteration cap | — |
| `f_init`, `att_init` | initial CBF / ATT guess shared by all voxels | mL/100g/min, s |
| `pld_j` | the j-th post-labeling delay | s |
| `true_cbf`, `true_att` | ground-truth physiology used to synthesize the curve | mL/100g/min, s |
| `s_j` | measured perfusion-weighted difference signal ΔM at PLD j | MR units (rel. to M0) |

Default study: 6 voxels, PLDs `{0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5}` s, tissue
types from cortical grey matter (CBF ≈ 60–80) to deep white matter (CBF ≈ 18).
The acquisition constants (T1 of blood/tissue, labeling efficiency α, partition
coefficient λ, bolus duration τ) are the ASL consensus defaults and live in
`src/asl.h` (`asl_default_constants`); they are **assumed known** during the fit,
so only CBF and ATT are estimated per voxel (exactly like the base BASIL model).

## "Full dataset" / real ASL data

Real multi-delay ASL is 4-D MRI (x, y, z, PLD) after label/control subtraction and
averaging. See `scripts/download_data.ps1` / `.sh` for pointers:

- **OpenNeuro** (<https://openneuro.org/>, search "ASL") — open, BIDS-formatted
  ASL datasets, many directly downloadable.
- **HCP ASL** (<https://db.humanconnectome.org/>) — requires free registration + a
  data-use agreement.
- **ISMRM 2015 ASL challenge** data; **UK Biobank** ASL pilot (approved application
  required).

The download scripts **do not** bypass any credential/registration; for credentialed
sets they print instructions only, and `make_synthetic.py` provides the offline
stand-in.

Bigger synthetic map: `python scripts/make_synthetic.py --voxels 1000000`.

## Provenance & honesty

The sample is **synthetic** and the fit uses a single-compartment teaching model
(fixed T1/α/λ, no dispersion, no partial-volume correction). Parameters are
illustrative, not fitted to any subject. Outputs are a software demonstration of
the kinetic-model fit, **not** a perfusion measurement and **not for any clinical
use**.
