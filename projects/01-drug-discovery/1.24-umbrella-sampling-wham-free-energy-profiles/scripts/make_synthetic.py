#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write the umbrella-sampling experiment config
# ---------------------------------------------------------------------------
# Project 1.24 : Umbrella Sampling / WHAM Free Energy Profiles
#
# The "data" here is the EXPERIMENT SETUP, not measured input: a synthetic
# double-well landscape plus the umbrella-window layout and Langevin-dynamics
# settings. The C++ program runs the biased dynamics, histograms the reaction
# coordinate per window, and reconstructs the potential of mean force with WHAM.
# Everything is reproducible from this one config + the RNG seed.
#
#   *** SYNTHETIC. The double-well is a teaching model, not a real molecule. ***
#
# OUTPUT (data/README.md format), whitespace-separated, in this exact order:
#   A b
#   x_min x_max nbins
#   n_windows win_min win_max k_spring
#   D dt n_equil n_sample seed
#
# The defaults are tuned so neighbouring windows' histograms OVERLAP (the
# precondition for WHAM) and the demo runs in well under a second.
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --n-windows 51 --n-sample 50000   # finer/longer
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "umbrella.txt"


def main():
    ap = argparse.ArgumentParser(description="Write the umbrella-sampling experiment config.")
    # The true landscape: U(x) = A (x^2 - b^2)^2 / b^4. Wells at +/- b, barrier A.
    ap.add_argument("--A", type=float, default=4.0, help="barrier height (kT)")
    ap.add_argument("--b", type=float, default=1.0, help="well half-separation")
    # Histogram grid (where the PMF is evaluated).
    ap.add_argument("--x-min", type=float, default=-1.6)
    ap.add_argument("--x-max", type=float, default=1.6)
    ap.add_argument("--nbins", type=int, default=32)
    # Umbrella windows: centers evenly spaced in [win-min, win-max], one spring.
    ap.add_argument("--n-windows", type=int, default=27)
    ap.add_argument("--win-min", type=float, default=-1.3)
    ap.add_argument("--win-max", type=float, default=1.3)
    ap.add_argument("--k-spring", type=float, default=12.0, help="harmonic spring (kT/x^2)")
    # Overdamped Langevin dynamics.
    ap.add_argument("--D", type=float, default=0.05, help="diffusion constant")
    ap.add_argument("--dt", type=float, default=0.005, help="Langevin timestep")
    ap.add_argument("--n-equil", type=int, default=4000, help="discarded warm-up steps/window")
    ap.add_argument("--n-sample", type=int, default=60000, help="recorded steps/window")
    ap.add_argument("--seed", type=int, default=20240117, help="base RNG seed")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    lines = [
        f"{args.A:g} {args.b:g}",
        f"{args.x_min:g} {args.x_max:g} {args.nbins}",
        f"{args.n_windows} {args.win_min:g} {args.win_max:g} {args.k_spring:g}",
        f"{args.D:g} {args.dt:g} {args.n_equil} {args.n_sample} {args.seed}",
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}")
    print(f"  double-well A={args.A} kT, b={args.b} (wells at +/-{args.b}, barrier {args.A} kT)")
    print(f"  {args.n_windows} windows in [{args.win_min}, {args.win_max}], "
          f"k_spring={args.k_spring}; {args.n_sample} steps/window")


if __name__ == "__main__":
    main()
