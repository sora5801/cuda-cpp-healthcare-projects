# Data — 4.19 Motion-Compensated 4D-CT Reconstruction

## Committed sample (`sample/sinogram4d_sample.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** (`scripts/make_synthetic.py`) — analytic sinogram of a *breathing* disc phantom |
| License | Public domain (CC0) — synthetic, safe to redistribute |
| Geometry | 8 breathing phases × 10 angles = 80 projections, 129 detector bins, spacing 0.02 |
| Motion | Breathing DVF amplitude 0.22 world units (phase 0 = reference, zero motion) |
| Reconstruction | 96×96 image over world [−1.0, 1.0]² |

This tiny file (~90 KB) lets `demo/run_demo` run **offline, with zero downloads**,
which is a hard requirement for every project (CLAUDE.md §8).

### File format

```
<img> <n_det> <n_phases> <n_ang_phase> <ds> <world_half> <amp>   # header line
<row 0: n_det projection values>            # phase 0, angle 0
<row 1: ...>                                 # phase 0, angle 1
...                                          # (n_phases * n_ang_phase) rows total
```

- Rows are **phase-major**: global index `k = p·n_ang_phase + a` (phase `p`, angle `a`).
- Projection angles tile a full half-turn: `theta_k = k·π/total`, so the *union*
  of all phases is well-sampled but each **single phase is sparse/under-sampled**.
- Detector bin `j` sits at offset `s_j = (j − (n_det−1)/2)·ds`.
- Each value is a **line integral** (Radon transform) of the phantom *as it is
  positioned during that phase*. The phantom's disc centers are displaced by the
  same breathing Deformation Vector Field the reconstruction uses
  (`src/mc4dct.h::dvf_at`), so the forward model and the motion model are
  consistent end-to-end. Phase 0 has zero motion and is the reference frame.

## Full / real dataset

Real 4D-CT lung data (and the way production tools consume it):

- **DIR-Lab 4D-CT lung** — <https://www.dir-lab.com/> — 10 cases with expert
  landmark pairs (the standard benchmark for deformable-registration accuracy).
- **TCIA 4D-CT lung radiotherapy** collections — <https://www.cancerimagingarchive.net>
  (real DICOM; registration may be required — the scripts do **not** bypass it).
- **POPI model** — <https://www.creatis.insa-lyon.fr/rio/popi-model> — a point-validated
  breathing 4D-CT dataset.
- **CIRS dynamic lung phantom** — a physical moving phantom used to validate 4D-CT.
- Reconstruction toolkits that ingest these: **RTK** (4D ROOSTER MCR), **ASTRA**,
  **TIGRE**, **Plastimatch** (DIR + 4D dose). See README "Prior art & further reading".

`scripts/download_data.ps1` / `.sh` describe how to obtain them. For a larger
synthetic problem:
`python scripts/make_synthetic.py --phases 10 --ang-phase 12 --det 257 --img 128`.

## Provenance & honesty

The sample is **synthetic** and labeled as such. Reconstructed values are in
arbitrary phantom-density units; the breathing motion is a smooth analytic model,
not a measured respiratory trace. This is a software demonstration of the
motion-compensated backprojection idea — **not** a calibrated CT image and **not**
for any clinical use.
