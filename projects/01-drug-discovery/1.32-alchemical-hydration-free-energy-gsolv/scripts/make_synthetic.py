#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write the alchemical delta-G calculation config
# ---------------------------------------------------------------------------
# Project 1.32 : Alchemical Hydration Free Energy (delta-G_solv)
#
# The committed "data" is the SETUP of a free-energy calculation, not measured
# input: the lambda schedule, the Monte Carlo sampling budget, and the model
# system parameters (Lennard-Jones + soft-core). The program builds the solvent
# bath deterministically from `bath_seed` (the same jittered-shell recipe the C++
# build_bath() uses), so the whole calculation is reproducible from this one line.
#
# OUTPUT (data/README.md format), a single whitespace-separated line:
#   n_solvent box T epsilon sigma q_solute alpha_sc max_step
#   n_windows n_walkers n_equil n_prod seed bath_seed
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --n-windows 21 --n-walkers 256   # finer/bigger
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "alchemy_config.txt"


def main():
    ap = argparse.ArgumentParser(description="Write the alchemical delta-G config line.")
    # --- physical model (reduced LJ units; see data/README.md + THEORY) -------
    ap.add_argument("--n-solvent", type=int, default=24, help="fixed solvent sites in the bath")
    ap.add_argument("--box", type=float, default=3.0, help="half-width of the solute sampling box [sigma]")
    ap.add_argument("--T", type=float, default=1.5, help="temperature [reduced eps/k_B]")
    ap.add_argument("--epsilon", type=float, default=1.0, help="LJ well depth [eps]")
    ap.add_argument("--sigma", type=float, default=1.0, help="LJ diameter [sigma]")
    ap.add_argument("--q-solute", type=float, default=0.0, help="solute charge (0 = LJ-only)")
    ap.add_argument("--alpha-sc", type=float, default=0.5, help="soft-core alpha (dimensionless)")
    ap.add_argument("--max-step", type=float, default=0.4, help="Metropolis trial displacement [sigma]")
    # --- sampling schedule ----------------------------------------------------
    ap.add_argument("--n-windows", type=int, default=11, help="lambda windows (>=2, includes 0 and 1)")
    ap.add_argument("--n-walkers", type=int, default=64, help="independent MC chains per window")
    ap.add_argument("--n-equil", type=int, default=200, help="burn-in MC steps per walker")
    ap.add_argument("--n-prod", type=int, default=800, help="production MC steps per walker")
    ap.add_argument("--seed", type=int, default=20260628, help="global RNG seed")
    ap.add_argument("--bath-seed", type=int, default=7, help="solvent-geometry seed")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    if args.n_windows < 2:
        ap.error("--n-windows must be at least 2 (the endpoints lambda=0 and lambda=1)")

    line = (f"{args.n_solvent} {args.box:g} {args.T:g} {args.epsilon:g} "
            f"{args.sigma:g} {args.q_solute:g} {args.alpha_sc:g} {args.max_step:g} "
            f"{args.n_windows} {args.n_walkers} {args.n_equil} {args.n_prod} "
            f"{args.seed} {args.bath_seed}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(line + "\n", encoding="utf-8")
    total = args.n_windows * args.n_walkers
    print(f"[make_synthetic] wrote {args.out}")
    print(f"  {args.n_windows} windows x {args.n_walkers} walkers = {total} GPU threads; "
          f"{args.n_equil}+{args.n_prod} MC steps each")
    print(f"  model: {args.n_solvent} solvent sites, T={args.T}, alpha_sc={args.alpha_sc} "
          f"(SYNTHETIC, reduced LJ units -- not a force-field prediction)")


if __name__ == "__main__":
    main()
