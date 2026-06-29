# Data — 11.09 Flow Cytometry & High-Content Screening Analysis

## Committed sample (`sample/cytometry_sample.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** cytometry events (`scripts/make_synthetic.py`, seed 8) |
| License | Public domain (CC0) — synthetic |
| Contents | 20,000 events × 5 markers, drawn from 5 well-separated populations |

### File format

```
<N> <D> <K>            # events, markers, clusters to find
<event 0: D floats>    # one row per event, values normalized to [0,1]
<event 1: ...>
... (N rows)
```

Markers are conceptually FSC, SSC, CD3, CD4, CD8 (a small T-cell panel). Values
are normalized to **[0,1]** so the fixed-point centroid accumulation (kmeans.h)
is exact. Events are grouped by population so the farthest-first init seeds one
centroid per population.

## Full dataset

Real cytometry data lives in **FCS** files; GPU clustering follows segmentation:

- **FlowKit** (<https://github.com/whitews/FlowKit>) — read/transform FCS files.
- **RAPIDS cuML** (<https://github.com/rapidsai/cuml>) — GPU UMAP/HDBSCAN/k-means.
- **FlowRepository** (<http://flowrepository.org>) — public FCS datasets.

Export a few markers per event (arcsinh-transform + scale to [0,1]) into the
format above. Bigger synthetic set: `python scripts/make_synthetic.py --scale 50`.

## Provenance & honesty

The sample is **synthetic** Gaussian blobs, not real immunophenotyping data, and
carries no clinical meaning. It exists to make the clustering result interpretable
(the 5 populations are recovered) and the GPU/CPU comparison verifiable.
