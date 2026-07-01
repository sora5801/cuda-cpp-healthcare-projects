#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write the SSA ensemble configuration (synthetic)
# ---------------------------------------------------------------------------
# Project 6.11 : Stochastic (Gillespie) Biochemical Simulation
#
# WHY THIS EXISTS
#   The "data" for this project is not a downloaded dataset but a tiny model
#   SPECIFICATION: the rate constants and run settings of a well-mixed reaction
#   network. Real BioModels/SBML files (see data/README.md) encode far larger
#   networks; here we generate a clearly-SYNTHETIC, analytically-checkable
#   birth-death gene-expression model so the demo runs offline and its answer is
#   known in closed form. Synthetic data is always LABELED synthetic.
#
# THE MODEL (constitutive gene expression, single species M = mRNA):
#     R1:  0 -> M   at rate k_prod          (transcription, zeroth order)
#     R2:  M -> 0   at rate k_deg * x_M     (degradation, first order)
#   Stationary distribution is Poisson with mean = k_prod / k_deg. The C++ demo
#   recovers that mean from the ensemble, validating the SSA.
#
# OUTPUT (data/README.md format), one line:
#   k_prod  k_deg  m0  t_end  n_traj  base_seed
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --k-prod 20 --k-deg 1 --n-traj 1024
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent            # the project folder
OUT = ROOT / "data" / "sample" / "gene_network.txt"


def main():
    ap = argparse.ArgumentParser(description="Write the synthetic SSA ensemble config.")
    ap.add_argument("--k-prod", type=float, default=10.0, help="transcription rate (molecules/time)")
    ap.add_argument("--k-deg", type=float, default=0.5, help="degradation rate (1/time)")
    ap.add_argument("--m0", type=int, default=0, help="initial mRNA count")
    ap.add_argument("--t-end", type=float, default=50.0, help="simulation horizon (time units)")
    ap.add_argument("--n-traj", type=int, default=256, help="number of independent trajectories")
    ap.add_argument("--base-seed", type=int, default=20240611, help="RNG base seed")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    # One whitespace-separated line the C++ loader (reference_cpu.cpp) parses.
    line = (f"{args.k_prod:g} {args.k_deg:g} {args.m0} "
            f"{args.t_end:g} {args.n_traj} {args.base_seed}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(line + "\n", encoding="utf-8")

    mean = args.k_prod / args.k_deg if args.k_deg else float("inf")
    print(f"[make_synthetic] wrote {args.out}  (SYNTHETIC; "
          f"{args.n_traj} trajectories, analytic Poisson mean = {mean:g})")


if __name__ == "__main__":
    main()
