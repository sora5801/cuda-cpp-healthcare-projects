#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write the LBM channel parameter file
# ---------------------------------------------------------------------------
# Project 6.04 : Lattice-Boltzmann Blood/Airflow Solver
#
# This project's "data" is the simulation setup: a periodic channel with no-slip
# walls, driven by a body force, that develops a Poiseuille (parabolic) velocity
# profile. This writes the one-line parameter file the program reads.
#
# OUTPUT (data/README.md format), one line:  "nx ny steps tau gx"
#   nx,ny : lattice size (x=flow direction, y=across the channel)
#   steps : number of collide+stream iterations
#   tau   : BGK relaxation time (viscosity nu=(tau-0.5)/3); must be > 0.5
#   gx    : body force per unit mass in +x (keep u << 1 for stability)
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --nx 128 --ny 64 --steps 20000
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "channel_params.txt"


def main():
    ap = argparse.ArgumentParser(description="Write the LBM channel parameter file.")
    ap.add_argument("--nx", type=int, default=16, help="lattice width (flow direction, periodic)")
    ap.add_argument("--ny", type=int, default=24, help="lattice height (across the channel)")
    ap.add_argument("--steps", type=int, default=6000, help="collide+stream iterations")
    ap.add_argument("--tau", type=float, default=0.8, help="BGK relaxation time (>0.5)")
    ap.add_argument("--gx", type=float, default=1e-5, help="body force per unit mass (+x)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    line = f"{args.nx} {args.ny} {args.steps} {args.tau} {args.gx}"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(line + "\n", encoding="utf-8")
    nu = (args.tau - 0.5) / 3.0
    print(f"[make_synthetic] wrote {args.out}")
    print(f"  nx={args.nx} ny={args.ny} steps={args.steps} tau={args.tau} "
          f"(nu={nu:.4f}) gx={args.gx}")


if __name__ == "__main__":
    main()
