#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write the QM/MM ensemble configuration
# ---------------------------------------------------------------------------
# Project 1.23 : QM/MM Molecular Dynamics   (reduced-scope teaching version)
#
# WHAT THIS WRITES (and the honesty)
#   The "data" is the ENSEMBLE SETUP, not measured input: the integration
#   settings plus a 2-D parameter sweep over the MM electrostatic-embedding field
#   and the initial proton position. The program derives each trajectory's
#   (field, x0) from the grid, so the whole ensemble is reproducible from this
#   one line. EVERYTHING HERE IS SYNTHETIC -- a model two-state double-well
#   surface, not a fitted potential-energy surface from quantum chemistry (see
#   data/README.md and THEORY.md for the full scope statement).
#
# OUTPUT (data/README.md format), one line:
#   dt steps v0 nf nx field_lo field_hi x0_lo x0_hi
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --nf 64 --nx 64       # 4096 trajectories
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "ensemble_params.txt"


def main():
    ap = argparse.ArgumentParser(description="Write the QM/MM ensemble configuration (SYNTHETIC).")
    ap.add_argument("--dt", type=float, default=0.004, help="velocity-Verlet timestep (model time)")
    ap.add_argument("--steps", type=int, default=5000, help="Verlet steps per trajectory")
    ap.add_argument("--v0", type=float, default=0.0, help="initial proton velocity (shared)")
    ap.add_argument("--nf", type=int, default=16, help="number of MM-field values")
    ap.add_argument("--nx", type=int, default=16, help="number of initial-position values")
    ap.add_argument("--field-lo", type=float, default=0.0,
                    help="MM embedding field, low end (no bias -> proton trapped)")
    ap.add_argument("--field-hi", type=float, default=-24.0,
                    help="MM embedding field, high end (more negative -> drives transfer)")
    ap.add_argument("--x0-lo", type=float, default=-0.70, help="initial proton position, low end")
    ap.add_argument("--x0-hi", type=float, default=-0.50,
                    help="initial proton position, high end (both inside the donor well)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    line = (f"{args.dt:g} {args.steps} {args.v0:g} {args.nf} {args.nx} "
            f"{args.field_lo:g} {args.field_hi:g} {args.x0_lo:g} {args.x0_hi:g}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(line + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"({args.nf * args.nx} trajectories, {args.steps} steps; "
          f"field in [{args.field_lo:g}, {args.field_hi:g}]; SYNTHETIC)")


if __name__ == "__main__":
    main()
