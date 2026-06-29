#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write the PBPK virtual-population config
# ---------------------------------------------------------------------------
# Project 13.02 : PBPK at Scale
#
# The "data" is the population setup: the oral dose, the model's median
# parameters (absorption ka, clearance CL, volumes Vc/Vp, inter-compartment flow
# Q), the log-normal variability (CV), integration settings, and the population
# size. Each patient's parameters are sampled deterministically from a seeded RNG
# (pbpk.h), so the whole study is reproducible.
#
# OUTPUT (data/README.md format), one line:
#   dose ka CL Vc Vp Q cv dt steps n_patients seed
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --patients 100000 --cv 0.4
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "pbpk_params.txt"


def main():
    ap = argparse.ArgumentParser(description="Write the PBPK population configuration.")
    ap.add_argument("--dose", type=float, default=100.0, help="oral dose (mg)")
    ap.add_argument("--ka", type=float, default=1.0, help="median absorption rate (1/h)")
    ap.add_argument("--CL", type=float, default=5.0, help="median clearance (L/h)")
    ap.add_argument("--Vc", type=float, default=30.0, help="median central volume (L)")
    ap.add_argument("--Vp", type=float, default=40.0, help="median peripheral volume (L)")
    ap.add_argument("--Q", type=float, default=7.0, help="median inter-compartment flow (L/h)")
    ap.add_argument("--cv", type=float, default=0.30, help="log-normal variability (CV)")
    ap.add_argument("--dt", type=float, default=0.05, help="integration step (h)")
    ap.add_argument("--steps", type=int, default=960, help="number of steps (run = steps*dt h)")
    ap.add_argument("--patients", type=int, default=4096, help="virtual population size")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    line = (f"{args.dose:g} {args.ka:g} {args.CL:g} {args.Vc:g} {args.Vp:g} {args.Q:g} "
            f"{args.cv:g} {args.dt:g} {args.steps} {args.patients} {args.seed}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(line + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  ({args.patients} patients, "
          f"{int(args.steps * args.dt)} h; expected mean AUC ~ dose/CL = {args.dose/args.CL:.1f})")


if __name__ == "__main__":
    main()
