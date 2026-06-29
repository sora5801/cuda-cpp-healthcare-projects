#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic spectral-search problem
# ---------------------------------------------------------------------------
# Project 12.01 : Mass-Spectrometry Proteomics Search
#
# Builds N "theoretical" library spectra (each a sparse set of fragment peaks
# binned to a fixed length) plus one "observed" QUERY derived from a chosen
# target library spectrum with intensity jitter and a little noise -- so the
# cosine search has a clear best match (the target). Real data is MS/MS spectra
# from mzML files searched against a peptide database (see download_data.*).
#
# OUTPUT (data/README.md format):
#   header: "N bins target"  then 1 query row, then N library rows (bins floats).
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --N 8192
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "spectra_sample.txt"


def make_spectrum(rng, bins, npeaks):
    vec = [0.0] * bins
    for _ in range(npeaks):
        b = rng.randint(0, bins - 1)
        vec[b] += rng.uniform(10.0, 100.0)      # fragment-ion intensity
    return vec


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic spectral-search dataset.")
    ap.add_argument("--N", type=int, default=1024, help="number of library spectra")
    ap.add_argument("--bins", type=int, default=200, help="intensity bins per spectrum")
    ap.add_argument("--peaks", type=int, default=18, help="fragment peaks per spectrum")
    ap.add_argument("--target", type=int, default=7, help="library index the query is derived from")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    N, bins = args.N, args.bins
    lib = [make_spectrum(rng, bins, args.peaks) for _ in range(N)]

    # Query = the target spectrum with per-peak intensity jitter + a few stray peaks.
    q = list(lib[args.target])
    for b in range(bins):
        if q[b] > 0.0:
            q[b] *= rng.uniform(0.8, 1.2)
    for _ in range(3):
        q[rng.randint(0, bins - 1)] += rng.uniform(5.0, 20.0)

    def fmt(v):
        return " ".join(f"{x:.3f}" for x in v)

    lines = [f"{N} {bins} {args.target}", fmt(q)] + [fmt(v) for v in lib]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  (N={N} library spectra, {bins} bins, "
          f"query from target {args.target}; SYNTHETIC)")


if __name__ == "__main__":
    main()
