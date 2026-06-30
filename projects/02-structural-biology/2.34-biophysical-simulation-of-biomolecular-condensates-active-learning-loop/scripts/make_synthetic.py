#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write the condensate-ensemble configuration
# ---------------------------------------------------------------------------
# Project 2.34 : Biophysical Simulation of Biomolecular Condensates
#                (Active Learning Loop)  --  reduced-scope teaching version
#
# WHY THIS EXISTS
#   The "data" for this project is the EXPERIMENT SETUP: the coarse-grained MD
#   model constants plus the stickiness (lambda) sweep that defines the candidate
#   sequences for one active-learning iteration, plus the experimental target
#   diffusion coefficient we are trying to match by design. No patient or
#   proprietary data is involved, so the whole input is a single synthetic line
#   this script writes. The program derives every replica's lambda from the grid,
#   so the entire ensemble is reproducible (deterministic counter-based RNG).
#
#   Synthetic data is always LABELED synthetic (see data/README.md).
#
# OUTPUT (one whitespace-separated line; field order matches the loader):
#   n_beads steps dt kT gamma k_bond r0 eq_steps lag seed
#   n_members lambda_lo lambda_hi k_cohese target_D
#
# USAGE
#   python scripts/make_synthetic.py                       # default 24-member sweep
#   python scripts/make_synthetic.py --n-members 200       # a larger ensemble
#   python scripts/make_synthetic.py --target-D 0.18       # aim at a different D
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "condensate_ensemble.txt"


def main():
    ap = argparse.ArgumentParser(
        description="Write the synthetic condensate-ensemble configuration.")
    # --- coarse-grained MD model constants (reduced MD units) ---
    ap.add_argument("--n-beads", type=int, default=12,
                    help="beads per chain (<= 16, the kernel's local-array cap)")
    ap.add_argument("--steps", type=int, default=500, help="Brownian-dynamics steps")
    ap.add_argument("--dt", type=float, default=0.005, help="integration timestep")
    ap.add_argument("--kT", type=float, default=1.0, help="thermal energy (noise scale)")
    ap.add_argument("--gamma", type=float, default=1.0, help="friction coefficient")
    ap.add_argument("--k-bond", type=float, default=80.0, help="harmonic bond stiffness")
    ap.add_argument("--r0", type=float, default=1.0, help="bond rest length")
    ap.add_argument("--eq-steps", type=int, default=150,
                    help="equilibration steps discarded before measuring")
    ap.add_argument("--lag", type=int, default=20,
                    help="MSD time-lag in steps (<= 24); sets the mobility probe")
    ap.add_argument("--seed", type=int, default=20260628, help="global RNG seed")
    # --- active-learning sweep ---
    ap.add_argument("--n-members", type=int, default=24,
                    help="number of candidate sequences (ensemble size)")
    ap.add_argument("--lambda-lo", type=float, default=0.5, help="low stickiness")
    ap.add_argument("--lambda-hi", type=float, default=8.0, help="high stickiness")
    ap.add_argument("--k-cohese", type=float, default=2.0,
                    help="base cohesive stiffness scale (all replicas)")
    ap.add_argument("--target-D", type=float, default=0.165,
                    help="experimental target diffusion coefficient to match")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    fields = [args.n_beads, args.steps, f"{args.dt:g}", f"{args.kT:g}",
              f"{args.gamma:g}", f"{args.k_bond:g}", f"{args.r0:g}",
              args.eq_steps, args.lag, args.seed,
              args.n_members, f"{args.lambda_lo:g}", f"{args.lambda_hi:g}",
              f"{args.k_cohese:g}", f"{args.target_D:g}"]
    line = " ".join(str(f) for f in fields)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(line + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"({args.n_members} candidate sequences, lambda in "
          f"[{args.lambda_lo:g}, {args.lambda_hi:g}]; SYNTHETIC)")


if __name__ == "__main__":
    main()
