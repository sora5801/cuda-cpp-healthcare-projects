#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic WSI tile-feature bag
# ---------------------------------------------------------------------------
# Project 4.11 : Digital Pathology / Whole-Slide Image Analysis
#
# WHAT THIS BUILDS
#   One "slide" = a BAG of N tile FEATURE vectors (each FEAT_DIM long). In a real
#   pipeline these come from a frozen CNN/ViT encoder run over each 224x224 tile;
#   here we synthesize them directly (we do NOT reimplement the encoder). The bag
#   is engineered so the attention-MIL result is INTERPRETABLE and VERIFIABLE:
#
#     * BACKGROUND tiles (the vast majority) have LOW features 0 and 1 -- normal
#       tissue / stroma the model should ignore.
#     * A small fraction of TUMOR tiles have HIGH features 0 and 1 -- the pattern
#       the frozen attention head (default_params() in reference_cpu.cpp) is tuned
#       to detect. Attention should concentrate on these few tiles, and the slide
#       probability should come out high (a "tumor" call).
#
#   This mirrors weakly-supervised WSI classification (CLAM): only a handful of
#   tiles carry the diagnosis, yet the slide label is driven by them. Everything
#   here is SYNTHETIC and carries no clinical meaning.
#
# OUTPUT (data/README.md format):
#   line 1 : "N D label"      (D must equal FEAT_DIM=8; label 0/1 or -1)
#   next N lines : D feature values per tile
#
# USAGE
#   python scripts/make_synthetic.py                 # default 64-tile slide
#   python scripts/make_synthetic.py --n 20000       # a realistic-size bag
#   python scripts/make_synthetic.py --tumor-frac 0  # a benign slide
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "slide_sample.txt"

FEAT_DIM = 8          # MUST match FEAT_DIM in src/wsi.h
TUMOR_FEATURES = 2    # features 0 and 1 are the "tumor markers" the head detects


def clip01(v):
    """Keep features in [0,1] (a plausible normalized encoder-output range)."""
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def make_tile(rng, is_tumor, spread):
    """One tile feature vector. Tumor tiles elevate features 0 and 1; every tile
    gets small random values in the remaining features so the bag is not
    degenerate (a stand-in for the many nuisance dimensions of a real encoder)."""
    tile = [clip01(rng.gauss(0.15, spread)) for _ in range(FEAT_DIM)]
    if is_tumor:
        for d in range(TUMOR_FEATURES):
            tile[d] = clip01(rng.gauss(0.90, spread))   # strong tumor signal
    else:
        for d in range(TUMOR_FEATURES):
            tile[d] = clip01(rng.gauss(0.10, spread))   # background: low signal
    return tile


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic WSI tile-feature bag.")
    ap.add_argument("--n", type=int, default=64, help="number of tiles in the bag")
    ap.add_argument("--tumor-frac", type=float, default=0.09,
                    help="fraction of tiles that are tumor (0 => benign slide)")
    ap.add_argument("--spread", type=float, default=0.03, help="Gaussian std per feature")
    ap.add_argument("--seed", type=int, default=411)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n_tumor = int(round(args.n * args.tumor_frac))
    # Place tumor tiles at deterministic, spread-out indices so the demo output is
    # stable and the "top attention tile" is a fixed, easy-to-check index.
    tumor_idx = set()
    if n_tumor > 0:
        step = max(1, args.n // n_tumor)
        i = 3                                # first tumor tile near the start
        while len(tumor_idx) < n_tumor and i < args.n:
            tumor_idx.add(i)
            i += step

    rows = []
    for i in range(args.n):
        tile = make_tile(rng, i in tumor_idx, args.spread)
        rows.append(" ".join(f"{v:.5f}" for v in tile))

    # Slide-level ground-truth label: tumor (1) if any tumor tile is present.
    label = 1 if n_tumor > 0 else 0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        f"{args.n} {FEAT_DIM} {label}\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(N={args.n} tiles, D={FEAT_DIM}, {n_tumor} tumor tiles at {sorted(tumor_idx)}, "
          f"label={label}; SYNTHETIC)")


if __name__ == "__main__":
    main()
