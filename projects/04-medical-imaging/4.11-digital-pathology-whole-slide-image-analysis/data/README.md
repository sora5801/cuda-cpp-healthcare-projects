# Data — 4.11 Digital Pathology / Whole-Slide Image Analysis

## Committed sample (`sample/slide_sample.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** tile-feature bag (`scripts/make_synthetic.py`, seed 411) |
| License | Public domain (CC0) — it is synthetic |
| Size | < 5 KB |
| Contents | One "slide": 64 tiles × 8 features, 6 planted **tumor** tiles, slide label 1 |

This tiny file lets `demo/run_demo` run **offline, with zero downloads**, which
is a hard requirement for every project (CLAUDE.md §8).

### File format

```
<N> <D> <label>          # N tiles, D features/tile (D MUST equal FEAT_DIM=8), label 0/1/-1
<tile 0: D floats>       # one row per tile: the tile's feature vector
<tile 1: ...>
... (N rows)
```

Each **tile** is a small patch of the slide; each row is that tile's **feature
vector** — in a real pipeline the output of a frozen CNN/ViT encoder (ResNet-50,
UNI). Here we synthesize the features directly (we do **not** reimplement the
encoder). Features 0 and 1 are the "tumor markers": **background** tiles keep them
low (~0.1), the 6 planted **tumor** tiles push them high (~0.9). The frozen
attention head (`default_params()` in `src/reference_cpu.cpp`) is tuned to fire on
that pattern, so attention concentrates on the tumor tiles and the slide is called
"TUMOR". The remaining 6 features are small nuisance values (a stand-in for the
hundreds of non-diagnostic dimensions of a real encoder).

The `label` (1 here) is the slide-level ground truth; the model does **not** see
it (weakly-supervised MIL), but the demo can report whether its call matches.

## Full dataset

Real WSIs are multi-gigabyte pyramids; you tile them, run an encoder per tile, and
save the `N × D` feature bag in the format above. Everything is credentialed/large,
so `scripts/download_data.*` **prints instructions only** (never bypasses logins):

- **TCGA slides (GDC):** <https://portal.gdc.cancer.gov/> — pan-cancer WSIs.
- **CAMELYON16/17:** <https://camelyon17.grand-challenge.org/> — lymph-node metastasis.
- **TUPAC16:** <http://tupac.tue-image.nl/> — tumor proliferation.
- **OpenSlide** (<https://openslide.org/>) reads the pyramids; **CLAM**
  (<https://github.com/mahmoodlab/CLAM>) does tiling + feature bags + MIL; **UNI**
  (<https://github.com/mahmoodlab/UNI>) is a pretrained ViT feature extractor.

Bigger synthetic bag (no download): `python scripts/make_synthetic.py --n 20000`.
A benign slide: `python scripts/make_synthetic.py --tumor-frac 0`.

## Provenance & honesty

The sample is a **synthetic** feature bag — Gaussian "tumor" and "background"
tiles — **not** real histology and with **no clinical meaning**. It exists to make
the attention-MIL result interpretable (attention lands on the planted tumor tiles)
and the GPU-vs-CPU comparison verifiable. Synthetic data is labeled synthetic
everywhere it appears (CLAUDE.md §8).
