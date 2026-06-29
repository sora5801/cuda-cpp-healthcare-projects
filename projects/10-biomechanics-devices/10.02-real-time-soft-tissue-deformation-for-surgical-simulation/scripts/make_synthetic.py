#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write the PBD cloth/tissue parameters
# ---------------------------------------------------------------------------
# Project 10.02 : Real-Time Soft-Tissue Deformation for Surgical Simulation
#
# The "data" is the simulation setup: the grid mesh size, timestep, gravity,
# constraint stiffness/relaxation, and iteration/step counts. The mesh itself
# (positions, pinned top row) is built deterministically from these.
#
# OUTPUT (data/README.md format), one line:
#   R C spacing dt gravity stiffness omega iters steps
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --R 128 --C 128 --steps 600
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "cloth_params.txt"


def main():
    ap = argparse.ArgumentParser(description="Write the PBD soft-tissue parameters.")
    ap.add_argument("--R", type=int, default=24, help="grid rows")
    ap.add_argument("--C", type=int, default=24, help="grid columns")
    ap.add_argument("--spacing", type=float, default=1.0, help="rest spacing")
    ap.add_argument("--dt", type=float, default=0.02, help="timestep")
    ap.add_argument("--gravity", type=float, default=10.0, help="gravity (-y)")
    ap.add_argument("--stiffness", type=float, default=1.0, help="constraint stiffness [0,1]")
    ap.add_argument("--omega", type=float, default=1.0, help="Jacobi relaxation factor")
    ap.add_argument("--iters", type=int, default=20, help="constraint iterations per step")
    ap.add_argument("--steps", type=int, default=300, help="number of timesteps")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    line = (f"{args.R} {args.C} {args.spacing:g} {args.dt:g} {args.gravity:g} "
            f"{args.stiffness:g} {args.omega:g} {args.iters} {args.steps}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(line + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  ({args.R}x{args.C} particles, "
          f"{args.iters} iters x {args.steps} steps; top row pinned)")


if __name__ == "__main__":
    main()
