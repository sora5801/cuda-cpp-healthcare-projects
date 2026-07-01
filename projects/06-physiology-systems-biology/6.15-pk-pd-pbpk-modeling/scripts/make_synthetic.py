#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write the PK/PD virtual-population config
# ---------------------------------------------------------------------------
# Project 6.15 : PK/PD & PBPK Modeling
#
# The "data" here is the POPULATION SETUP, not a table of measurements: an oral
# dose, the median PK parameters (absorption ka, clearance CL, central volume Vc),
# the median PD parameters (turnover kin/kout, inhibition Imax/IC50), the
# between-subject log-normal variability (CV), the integration settings, and the
# population size. Each virtual patient's individual parameters are then sampled
# deterministically from a seeded RNG inside the program (src/pkpd.h), so the whole
# study is reproducible and the CPU and GPU produce the identical population.
#
# OUTPUT (data/README.md format), one whitespace-separated line:
#   dose ka CL Vc kin kout Imax IC50 cv dt steps n_patients seed
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --patients 100000 --cv 0.4
# ===========================================================================
import argparse
from pathlib import Path

# scripts/ lives one level under the project root; the sample sits in data/sample.
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "pkpd_params.txt"


def main():
    ap = argparse.ArgumentParser(description="Write the PK/PD population configuration.")
    # ---- dosing ----
    ap.add_argument("--dose", type=float, default=100.0, help="oral dose (mg)")
    # ---- PK medians (one-compartment oral) ----
    ap.add_argument("--ka", type=float, default=1.0, help="median absorption rate (1/h)")
    ap.add_argument("--CL", type=float, default=5.0, help="median clearance (L/h)")
    ap.add_argument("--Vc", type=float, default=30.0, help="median central volume (L)")
    # ---- PD medians (indirect-response turnover; baseline R0 = kin/kout) ----
    ap.add_argument("--kin", type=float, default=10.0, help="biomarker production rate (units/h)")
    ap.add_argument("--kout", type=float, default=0.20, help="biomarker loss rate (1/h)")
    ap.add_argument("--Imax", type=float, default=0.90, help="max fractional inhibition of loss [0,1]")
    ap.add_argument("--IC50", type=float, default=2.0, help="conc for half-max inhibition (mg/L)")
    # ---- variability + integration ----
    ap.add_argument("--cv", type=float, default=0.25, help="log-normal between-subject variability")
    ap.add_argument("--dt", type=float, default=0.05, help="RK4 step (h)")
    ap.add_argument("--steps", type=int, default=960, help="number of steps (run = steps*dt h)")
    ap.add_argument("--patients", type=int, default=4096, help="virtual population size")
    ap.add_argument("--seed", type=int, default=99, help="base RNG seed")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    # Emit exactly the 13 fields the loader (reference_cpu.cpp) expects, in order.
    line = (f"{args.dose:g} {args.ka:g} {args.CL:g} {args.Vc:g} "
            f"{args.kin:g} {args.kout:g} {args.Imax:g} {args.IC50:g} "
            f"{args.cv:g} {args.dt:g} {args.steps} {args.patients} {args.seed}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(line + "\n", encoding="utf-8")

    # A couple of sanity numbers a learner can check against the program output:
    #   PK: mean AUC ~ dose/CL (complete-absorption identity).
    #   PD: baseline biomarker R0 = kin/kout.
    print(f"[make_synthetic] wrote {args.out}  (SYNTHETIC)")
    print(f"    {args.patients} patients, {int(args.steps * args.dt)} h horizon")
    print(f"    expected mean AUC ~ dose/CL = {args.dose / args.CL:.1f} mg.h/L")
    print(f"    biomarker baseline R0 = kin/kout = {args.kin / args.kout:.1f} units")


if __name__ == "__main__":
    main()
