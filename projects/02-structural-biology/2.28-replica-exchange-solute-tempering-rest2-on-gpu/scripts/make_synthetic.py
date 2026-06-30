#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the tiny committed REST2 config
# ---------------------------------------------------------------------------
# Project 2.28 : Replica Exchange Solute Tempering (REST2) on GPU
#
# WHAT THIS WRITES
#   data/sample/rest2_config.txt -- the 10 numbers that define a REST2 run for
#   the demo. The file is SYNTHETIC (no real protein/trajectory): a toy solute
#   of beads, each in a symmetric DOUBLE-WELL potential, sampled by Metropolis
#   Monte Carlo. It is engineered so the demo is INTERPRETABLE (PATTERNS.md
#   section 6): the barrier is high enough that a 300 K replica on its own would
#   stay trapped in the left well, but the REST2 temperature ladder plus
#   configuration swaps let the cold replica escape to the right well -- the
#   whole point of enhanced sampling, made visible.
#
#   The C++ program uses this same file (committed) so the demo runs offline; a
#   learner can regenerate or sweep parameters here (try a taller barrier!).
#
# CONFIG FORMAT (whitespace-separated, this exact order):
#   line 1: n_replicas  sweeps_per_round  n_rounds
#   line 2: barrier_h  tilt  k_bond  k_pw  x_solvent  step_size
#   line 3: T0  Tmax
#   (tilt lowers the RIGHT well so it is the GLOBAL minimum -- the "answer" the
#    cold replica should reach via REST2 exchanges, having started trapped left.)
#
# USAGE
#   python scripts/make_synthetic.py                  # -> data/sample/rest2_config.txt
#   python scripts/make_synthetic.py --barrier-h 9    # a harder barrier
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "rest2_config.txt"


def main() -> None:
    ap = argparse.ArgumentParser(description="Write the synthetic REST2 demo config.")
    # Defaults chosen to make the teaching point land (see header).
    ap.add_argument("--n-replicas", type=int, default=8,
                    help="replicas in the temperature ladder (>= 2)")
    ap.add_argument("--sweeps-per-round", type=int, default=200,
                    help="MC sweeps between exchange attempts")
    ap.add_argument("--n-rounds", type=int, default=60,
                    help="number of (sample, exchange) rounds")
    ap.add_argument("--barrier-h", type=float, default=5.0,
                    help="double-well barrier height in kT units (high = hard to cross)")
    ap.add_argument("--tilt", type=float, default=2.0,
                    help="linear bias lowering the right well (makes it the global min)")
    ap.add_argument("--k-bond", type=float, default=0.5,
                    help="bead-bead bond stiffness (solute internal energy)")
    ap.add_argument("--k-pw", type=float, default=0.2,
                    help="solute-solvent coupling stiffness (the sqrt(lambda) term)")
    ap.add_argument("--x-solvent", type=float, default=0.0,
                    help="position of the implicit solvent field")
    ap.add_argument("--step-size", type=float, default=0.35,
                    help="Metropolis trial-move half-width (tunes acceptance)")
    ap.add_argument("--t0", type=float, default=300.0,
                    help="cold (physical) temperature in K -> lambda = 1")
    ap.add_argument("--tmax", type=float, default=900.0,
                    help="hottest effective solute temperature in K")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    lines = [
        f"{args.n_replicas} {args.sweeps_per_round} {args.n_rounds}",
        f"{args.barrier_h:g} {args.tilt:g} {args.k_bond:g} {args.k_pw:g} "
        f"{args.x_solvent:g} {args.step_size:g}",
        f"{args.t0:g} {args.tmax:g}",
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[make_synthetic] wrote {args.out}")
    print(f"[make_synthetic] {args.n_replicas} replicas, "
          f"T = {args.t0:g}..{args.tmax:g} K, barrier h = {args.barrier_h:g} kT (SYNTHETIC)")


if __name__ == "__main__":
    main()
