#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write the TPS simulation parameter file
# ---------------------------------------------------------------------------
# Project 2.32 : Protein Folding Pathway Extraction (Transition Path Sampling)
#
# This project's "data" is the set of SIMULATION PARAMETERS that define the
# folding free-energy landscape (a 1-D double well) and the shooting run -- not
# a measured trajectory. This script writes the one-line parameter file the
# program reads. A fixed seed makes the whole simulation reproducible, so the
# demo's stdout is byte-stable.
#
# The DEFAULTS here are deliberately identical to the built-in fallback in
# src/main.cu::make_synthetic(), so the program prints the same result whether
# or not this file is passed on the command line.
#
# OUTPUT (data/README.md format), one whitespace-separated line:
#   barrier x0 w D dt basin_tol max_steps n_shooters n_bins seed
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --shooters 16384 --barrier 6.0
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "tps_params.txt"


def main():
    ap = argparse.ArgumentParser(description="Write the TPS double-well parameter file (SYNTHETIC).")
    ap.add_argument("--barrier", type=float, default=5.0,
                    help="double-well barrier height in units of kT")
    ap.add_argument("--x0", type=float, default=0.5,
                    help="landscape centre (transition-state position)")
    ap.add_argument("--w", type=float, default=0.4,
                    help="basin half-separation (basins at x0 +/- w)")
    ap.add_argument("--D", type=float, default=1.0,
                    help="diffusion constant on the reaction coordinate (reduced)")
    ap.add_argument("--dt", type=float, default=0.0005,
                    help="Brownian-dynamics timestep (reduced units)")
    ap.add_argument("--basin-tol", type=float, default=0.05,
                    help="distance from a minimum that counts as 'arrived' (< w)")
    ap.add_argument("--max-steps", type=int, default=20000,
                    help="per-leg BD step budget (rare-event safety net)")
    ap.add_argument("--shooters", type=int, default=4096,
                    help="number of independent shooting moves")
    ap.add_argument("--bins", type=int, default=20,
                    help="committor-histogram resolution along the reaction coord")
    ap.add_argument("--seed", type=int, default=20240517,
                    help="base RNG seed (shooter i uses stream (seed, i))")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    if args.basin_tol >= args.w:
        raise SystemExit("basin_tol must be < w so the two basins do not overlap")

    line = (f"{args.barrier} {args.x0} {args.w} {args.D} {args.dt} "
            f"{args.basin_tol} {args.max_steps} {args.shooters} {args.bins} {args.seed}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(line + "\n", encoding="utf-8")

    print(f"[make_synthetic] wrote {args.out}  (SYNTHETIC)")
    print(f"  barrier={args.barrier} kT  x0={args.x0}  w={args.w}  D={args.D}  dt={args.dt}")
    print(f"  basin_tol={args.basin_tol}  max_steps={args.max_steps}  "
          f"shooters={args.shooters}  bins={args.bins}  seed={args.seed}")


if __name__ == "__main__":
    main()
