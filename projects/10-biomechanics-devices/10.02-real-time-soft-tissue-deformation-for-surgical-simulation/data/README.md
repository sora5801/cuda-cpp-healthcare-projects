# Data — 10.2 Real-Time Soft-Tissue Deformation for Surgical Simulation

## Committed sample (`sample/`)

| Field | Value |
|---|---|
| File | `sample/saxpy_sample.txt` |
| Origin | **Synthetic** (generated; template placeholder) |
| License | Public domain (CC0) — it is synthetic |
| Size | < 1 KB |
| Layout | line 1: `n`; line 2: `a`; line 3: `n` x-values; line 4: `n` y-values |

This tiny file lets `demo/run_demo` run **offline, with zero downloads**, which
is a hard requirement for every project (CLAUDE.md §8).

TODO(impl): replace with this project's real tiny sample, and document each
field's meaning, units, and provenance below.

## Full dataset

TODO(impl): describe the real dataset(s) from the catalog and how to fetch them:

- **Source / URL:** (from the catalog "Datasets" column)
- **License:** respect it. If redistribution is forbidden, the committed sample
  MUST be synthetic and `make_synthetic.py` provides a stand-in.
- **Size & checksum:** documented in `scripts/download_data.*`.
- **Credentialed sets** (MIMIC, UK Biobank, ...): the download script must NOT
  bypass registration — it prints instructions and links only.

Catalog dataset notes (verbatim):

> SOFA Framework benchmark scenes — laparoscopic and open-surgery deformable organ models (https://www.sofa-framework.org/); Kaggle Liver CT Segmentation — 3D liver meshes for deformation benchmarking (https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation); MRI Breast Tissue Segmentation (nnU-Net preprocessed) for biomechanical modeling (https://arxiv.org/abs/2411.18784); iMSTK Test Suite — pre-built surgical scenario meshes (https://www.imstk.org/).

## Provenance & field meanings

TODO(impl): per-field meaning for the real dataset. Never imply clinical
validity; label synthetic data as synthetic everywhere it appears.
