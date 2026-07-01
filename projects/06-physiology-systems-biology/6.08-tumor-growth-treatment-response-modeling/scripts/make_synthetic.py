#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write the tumor-growth + treatment config
# ---------------------------------------------------------------------------
# Project 6.8 : Tumor Growth & Treatment-Response Modeling
#
# The "data" here is the SIMULATION SETUP: grid geometry, the Fisher-KPP growth
# parameters (diffusion D, proliferation rho), the explicit-Euler timestep, and
# the fractionated-radiotherapy schedule (LQ radiosensitivities alpha/beta, dose
# per fraction, number of fractions, spacing). The initial tumor (a small seed
# disc at the grid centre) is built deterministically by the program from these
# numbers. Everything is SYNTHETIC and labeled synthetic (see data/README.md).
#
# OUTPUT (data/README.md format), one whitespace-separated line:
#   nx ny dx D rho dt steps alpha beta dose n_fractions fx_interval seed_radius seed_u
#
# The defaults are chosen to be didactically clear AND numerically stable:
#   * dt = 0.25 day satisfies the explicit-Euler limit dt <= dx^2/(4 D) = 0.5 day.
#   * 400 steps * 0.25 day = 100 simulated days.
#   * 10 x 2 Gy fractions (alpha/beta = 10 Gy) is a schematic tumor RT course;
#     each fraction's LQ surviving fraction is exp(-(0.15*2 + 0.015*4)) ~ 0.70.
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --n-fractions 0   # untreated control setup
#   python scripts/make_synthetic.py --dose 3 --n-fractions 10
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "tumor_params.txt"


def main():
    ap = argparse.ArgumentParser(description="Write the tumor-growth + RT configuration.")
    # --- grid + growth (Fisher-KPP) ---
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--dx", type=float, default=0.2, help="cell spacing [mm]")
    ap.add_argument("--D", type=float, default=0.02, help="cell diffusion [mm^2/day]")
    ap.add_argument("--rho", type=float, default=0.15, help="proliferation rate [1/day]")
    ap.add_argument("--dt", type=float, default=0.25, help="timestep [day]")
    ap.add_argument("--steps", type=int, default=400)
    # --- treatment (linear-quadratic radiobiology) ---
    ap.add_argument("--alpha", type=float, default=0.15, help="LQ alpha [1/Gy]")
    ap.add_argument("--beta", type=float, default=0.015, help="LQ beta [1/Gy^2]")
    ap.add_argument("--dose", type=float, default=2.0, help="dose per fraction [Gy]")
    ap.add_argument("--n-fractions", type=int, default=10, help="number of fractions (0=control)")
    ap.add_argument("--fx-interval", type=int, default=20, help="steps between fractions")
    # --- initial seed ---
    ap.add_argument("--seed-radius", type=float, default=1.0, help="seed radius [mm]")
    ap.add_argument("--seed-u", type=float, default=1.0, help="seed density (0..1)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    # Guard the explicit-Euler stability limit so we never emit a config that blows up.
    dt_max = (args.dx * args.dx) / (4.0 * args.D) if args.D > 0 else float("inf")
    if args.dt > dt_max:
        raise SystemExit(f"[make_synthetic] unstable: dt={args.dt} > dx^2/(4D)={dt_max:.4f} day")

    line = (f"{args.nx} {args.ny} {args.dx:g} {args.D:g} {args.rho:g} {args.dt:g} "
            f"{args.steps} {args.alpha:g} {args.beta:g} {args.dose:g} "
            f"{args.n_fractions} {args.fx_interval} {args.seed_radius:g} {args.seed_u:g}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(line + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  ({args.nx}x{args.ny} grid, {args.steps} steps, "
          f"{args.n_fractions}x{args.dose} Gy; SYNTHETIC)")


if __name__ == "__main__":
    main()
