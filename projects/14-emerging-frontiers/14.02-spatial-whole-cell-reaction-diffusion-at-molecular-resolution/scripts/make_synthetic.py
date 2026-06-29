#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write the Gray-Scott reaction-diffusion config
# ---------------------------------------------------------------------------
# Project 14.02 : Spatial / Whole-Cell Reaction-Diffusion (teaching stencil)
#
# The "data" is the simulation setup: grid size, diffusion coefficients, the
# feed/kill rates (which SELECT the Turing pattern), timestep, step count, and
# the size of the central seed. The grid itself is built deterministically.
#
# OUTPUT (data/README.md format), one line:
#   nx ny Du Dv F k dt steps seed_half
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --F 0.0545 --k 0.0620   # labyrinth pattern
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "grayscott_params.txt"


def main():
    ap = argparse.ArgumentParser(description="Write the Gray-Scott RD configuration.")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--Du", type=float, default=0.16, help="U diffusion")
    ap.add_argument("--Dv", type=float, default=0.08, help="V diffusion")
    ap.add_argument("--F", type=float, default=0.0545, help="feed rate (selects pattern)")
    ap.add_argument("--k", type=float, default=0.0620, help="kill rate (selects pattern)")
    ap.add_argument("--dt", type=float, default=1.0, help="timestep (explicit Euler)")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seed-half", type=int, default=8, help="half-size of the central V seed")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    line = (f"{args.nx} {args.ny} {args.Du:g} {args.Dv:g} {args.F:g} {args.k:g} "
            f"{args.dt:g} {args.steps} {args.seed_half}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(line + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  ({args.nx}x{args.ny} grid, {args.steps} steps, "
          f"F={args.F} k={args.k} -> labyrinth pattern)")


if __name__ == "__main__":
    main()
