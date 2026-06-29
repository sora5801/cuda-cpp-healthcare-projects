#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write the SEIR ensemble configuration
# ---------------------------------------------------------------------------
# Project 9.02 : Large-Scale Compartmental & Metapopulation Models
#
# The "data" is the ensemble setup: population, initial condition, integration
# settings, and the beta x gamma parameter sweep. The program derives each
# member's (beta, gamma) from the grid, so the whole ensemble is reproducible.
#
# OUTPUT (data/README.md format), one line:
#   N I0 dt steps sigma nb ng beta_lo beta_hi gamma_lo gamma_hi
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --nb 200 --ng 200      # 40,000 members
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "ensemble_params.txt"


def main():
    ap = argparse.ArgumentParser(description="Write the SEIR ensemble configuration.")
    ap.add_argument("--N", type=float, default=1_000_000.0, help="total population")
    ap.add_argument("--I0", type=float, default=10.0, help="initial infectious")
    ap.add_argument("--dt", type=float, default=0.25, help="RK4 timestep (days)")
    ap.add_argument("--steps", type=int, default=720, help="number of steps (run = steps*dt days)")
    ap.add_argument("--sigma", type=float, default=1.0 / 5.2, help="E->I rate (1/latent period)")
    ap.add_argument("--nb", type=int, default=64, help="number of beta values")
    ap.add_argument("--ng", type=int, default=64, help="number of gamma values")
    ap.add_argument("--beta-lo", type=float, default=0.15)
    ap.add_argument("--beta-hi", type=float, default=0.60)
    ap.add_argument("--gamma-lo", type=float, default=0.10)
    ap.add_argument("--gamma-hi", type=float, default=0.50)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    line = (f"{args.N:g} {args.I0:g} {args.dt:g} {args.steps} {args.sigma:.6f} "
            f"{args.nb} {args.ng} {args.beta_lo:g} {args.beta_hi:g} "
            f"{args.gamma_lo:g} {args.gamma_hi:g}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(line + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  ({args.nb * args.ng} members, "
          f"{int(args.steps * args.dt)} days; R0 in "
          f"[{args.beta_lo / args.gamma_hi:.2f}, {args.beta_hi / args.gamma_lo:.2f}])")


if __name__ == "__main__":
    main()
