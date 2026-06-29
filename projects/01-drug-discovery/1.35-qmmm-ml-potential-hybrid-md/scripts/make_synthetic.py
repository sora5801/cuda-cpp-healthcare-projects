#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic ensemble-config sample
# ---------------------------------------------------------------------------
# Project 1.35 : QMMM/ML Potential Hybrid MD   (reduced-scope teaching version)
#
# WHY THIS EXISTS
#   The real datasets for training the ML potential (Transition1x, SPICE,
#   ANI-1ccx) are large QM/DFT reference sets that cannot be shipped in this repo
#   and require their own training pipeline (see data/README.md). This teaching
#   project does NOT train a network; its NNP weights are FIXED, committed
#   surrogate values inside src/nnpmm.h. So the only "input" the program needs is
#   a tiny ENSEMBLE CONFIG describing how many short MD trajectories to run and
#   with what integration settings. This script writes that config.
#
#   The output is therefore SYNTHETIC and deterministic -- it is a run spec, not
#   measured data, and is labeled synthetic everywhere (CLAUDE.md §8).
#
# FILE FORMAT (whitespace-separated, read by load_ensemble in reference_cpu.cpp):
#     M  dt  steps  amp
#   M     : number of ensemble members (independent trajectories)
#   dt    : velocity-Verlet timestep (time units)
#   steps : integration steps per trajectory
#   amp   : max +/- perturbation applied to the link atom across the ensemble
#
# USAGE
#   python scripts/make_synthetic.py                 # default tiny sample
#   python scripts/make_synthetic.py --M 4096        # bigger ensemble (stress GPU)
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "ensemble_params.txt"


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic ensemble config.")
    ap.add_argument("--M", type=int, default=64, help="number of trajectories")
    ap.add_argument("--dt", type=float, default=0.005, help="Verlet timestep")
    ap.add_argument("--steps", type=int, default=300, help="steps per trajectory")
    ap.add_argument("--amp", type=float, default=0.20, help="link-atom perturbation +/-")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    # One line: "M dt steps amp". A small dt keeps the symplectic integrator
    # stable; 300 steps is long enough to show energy conservation, short enough
    # to run instantly. amp=0.20 spreads the ensemble across configuration space.
    line = f"{args.M} {args.dt:g} {args.steps} {args.amp:g}\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(line, encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(M={args.M}, dt={args.dt}, steps={args.steps}, amp={args.amp}; SYNTHETIC)")


if __name__ == "__main__":
    main()
