# Data — 6.5 Respiratory / Lung Airflow & Particle Deposition

## Committed sample (`sample/lung_params.txt`)

| Field | Value |
|---|---|
| File | `sample/lung_params.txt` |
| Origin | **Synthetic** (hand-written parameters; not patient-derived) |
| License | Public domain (CC0) — it is synthetic |
| Size | < 1 KB |
| Layout | one data line of six whitespace-separated fields (`#` lines are comments) |

This tiny file lets `demo/run_demo` run **offline, with zero downloads**, which
is a hard requirement for every project (CLAUDE.md §8). It is **synthetic** — it
describes an idealized experiment, not any real patient — and must never be read
as clinically meaningful.

### Field meanings (the one data line)

```
d_p_microns  rho_p_kg_m3  n_gen  flow_L_per_min  n_particles  seed
5.0          1000         16     30              200000       12345
```

| Field | Meaning | Units |
|---|---|---|
| `d_p_microns` | aerosol particle aerodynamic diameter | micrometres (µm) |
| `rho_p_kg_m3` | particle mass density (1000 = water-like) | kg/m³ |
| `n_gen` | conducting-airway generations modelled (0 = trachea) | count |
| `flow_L_per_min` | steady inspiratory volumetric flow rate | litres/min |
| `n_particles` | number of Monte-Carlo particle histories to track | count |
| `seed` | base RNG seed (particle *i* uses stream `(seed, i)`) | integer |

The loader (`src/reference_cpu.cpp`) converts µm → m and L/min → m³/s. The
**airway geometry itself is not in the file**: it is built deterministically
from `n_gen` and `flow_rate` by `build_airway()` as a scaled symmetric
(Weibel-A) tree — see `THEORY.md`. Regenerate the file with:

```bash
python scripts/make_synthetic.py                       # defaults above
python scripts/make_synthetic.py --d_p 1.0 --n 500000  # sub-micron, more MC
```

## Full / real dataset

A patient-specific study replaces the idealized tree with an airway geometry
**segmented from a lung CT scan** (per-generation radii and lengths fitted from
the segmentation). Those archives require registration and a data-use agreement;
`scripts/download_data.*` prints the links and **never** bypasses credentials.

Catalog dataset notes (verbatim):

> LIDC-IDRI lung CT — 1 010 cases with nodule annotations, TCIA (https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI); COPDGene lung CT dataset — 10 000 subjects (https://www.copdgene.org); SPIROMICS bronchial CT (https://www.spiromics.org); PhysioNet respiratory waveform databases (https://physionet.org).

| Dataset | Access | Use here |
|---|---|---|
| LIDC-IDRI (TCIA) | free account + license | source CT for airway segmentation |
| COPDGene | application / DUA | diseased-airway geometries |
| SPIROMICS | application / DUA | bronchial CT |
| PhysioNet respiratory | free account | breathing waveforms → time-varying flow |

**Segmentation tooling:** 3D Slicer + SlicerMorph
(https://github.com/SlicerMorph/SlicerMorph) extract the airway centerline tree
from a CT volume; per-generation radii/lengths then feed `build_airway()`.

## Honesty

The committed sample and all demo output are **synthetic and educational**. No
number here is patient-derived or clinically valid; this project must not be used
for diagnosis, treatment, or inhaler/drug-delivery decisions (CLAUDE.md §1, §8).
