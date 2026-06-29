#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write the Monte Carlo parameter file
# ---------------------------------------------------------------------------
# Project 5.01 : Monte Carlo Dose Calculation (simplified slab)
#
# This project's "data" is the set of SIMULATION PARAMETERS (the slab + how many
# histories), not measured input. This script writes the one-line parameter file
# the program reads. A fixed seed makes the whole simulation reproducible.
#
# OUTPUT (data/README.md format), one line:
#   L n_bins mu p_abs E0 scatter_dep n_photons seed
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --photons 4000000 --mu 0.20
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "mc_params.txt"


def main():
    ap = argparse.ArgumentParser(description="Write the MC slab parameter file.")
    ap.add_argument("--L", type=float, default=20.0, help="slab thickness (cm)")
    ap.add_argument("--bins", type=int, default=40, help="number of depth bins")
    ap.add_argument("--mu", type=float, default=0.15, help="attenuation coeff (1/cm)")
    ap.add_argument("--p-abs", type=float, default=0.30, help="P(interaction is absorption)")
    ap.add_argument("--E0", type=int, default=1024, help="starting energy quanta per photon")
    ap.add_argument("--scatter-dep", type=int, default=128, help="quanta deposited per scatter")
    ap.add_argument("--photons", type=int, default=262144, help="number of photon histories")
    ap.add_argument("--seed", type=int, default=12345, help="base RNG seed")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    line = f"{args.L} {args.bins} {args.mu} {args.p_abs} {args.E0} {args.scatter_dep} {args.photons} {args.seed}"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(line + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}")
    print(f"  L={args.L} bins={args.bins} mu={args.mu} p_abs={args.p_abs} "
          f"E0={args.E0} scatter_dep={args.scatter_dep} photons={args.photons} seed={args.seed}")


if __name__ == "__main__":
    main()
