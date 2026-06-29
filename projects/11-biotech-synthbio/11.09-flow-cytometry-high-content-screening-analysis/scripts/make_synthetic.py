#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate synthetic flow-cytometry events
# ---------------------------------------------------------------------------
# Project 11.09 : Flow Cytometry & High-Content Screening Analysis
#
# Builds N cell "events", each a D-marker vector, drawn from K well-separated
# Gaussian populations (a stand-in for distinct cell types in a marker panel).
# Values are normalized to [0,1] (so the fixed-point accumulation in kmeans.h is
# valid). Events are grouped by population, so the evenly-spaced centroid init
# seeds one cluster per population and k-means recovers them cleanly. Real data
# comes from FCS files (see download_data.*).
#
# OUTPUT (data/README.md format): "N D K" then N rows of D floats.
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --scale 5     # 5x more events
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "cytometry_sample.txt"

# K=5 populations in a D=5 marker space (e.g. FSC, SSC, CD3, CD4, CD8), with a
# representative count each (imbalanced, like real immunophenotyping).
POPULATIONS = [
    ([0.80, 0.20, 0.80, 0.70, 0.20], 6000),   # CD3+ CD4+ T-helper
    ([0.80, 0.30, 0.80, 0.20, 0.70], 5000),   # CD3+ CD8+ T-cytotoxic
    ([0.30, 0.30, 0.20, 0.20, 0.20], 4000),   # CD3- B / other
    ([0.60, 0.80, 0.30, 0.30, 0.30], 3000),   # high SSC monocytes
    ([0.20, 0.20, 0.50, 0.60, 0.50], 2000),   # mixed
]
SPREAD = 0.04


def clip01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic cytometry dataset.")
    ap.add_argument("--scale", type=int, default=1, help="multiply every population's count")
    ap.add_argument("--spread", type=float, default=SPREAD, help="Gaussian std per marker")
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    D = len(POPULATIONS[0][0])
    K = len(POPULATIONS)
    rows = []
    for (center, count) in POPULATIONS:
        for _ in range(count * args.scale):
            ev = [clip01(center[j] + rng.gauss(0.0, args.spread)) for j in range(D)]
            rows.append(" ".join(f"{v:.5f}" for v in ev))
    N = len(rows)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(f"{N} {D} {K}\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  (N={N} events, D={D} markers, K={K} populations; SYNTHETIC)")


if __name__ == "__main__":
    main()
