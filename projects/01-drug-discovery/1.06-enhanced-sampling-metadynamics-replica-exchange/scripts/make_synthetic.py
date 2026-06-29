#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write the metadynamics run configuration
# ---------------------------------------------------------------------------
# Project 1.6 : Enhanced Sampling -- Metadynamics & Replica Exchange
#
# The "data" for this project is the RUN CONFIGURATION: the double-well model,
# the Langevin thermostat, the metadynamics controls (hill height/width/pace,
# bias factor), the bias grid, and the ensemble size. The program derives
# everything else (each walker's RNG stream + start well) deterministically, so
# the whole run is reproducible from this one line.
#
# This is a SYNTHETIC, analytically-known landscape on purpose: because we KNOW
# the true free-energy surface F0(s) = A*(s^2-1)^2, we can verify that
# metadynamics RECOVERS it. (No real MD trajectory is involved -- see
# data/README.md and THEORY.md "Where this sits in the real world".)
#
# OUTPUT (single line, whitespace-separated; field order documented in
# data/README.md and parsed by src/reference_cpu.cpp::load_config):
#   A kT mass friction dt steps hill_w hill_sigma deposit_every bias_factor
#   s_lo s_hi nbins n_walkers seed s_start
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --n-walkers 256 --steps 40000   # bigger run
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "metad_config.txt"


def main():
    ap = argparse.ArgumentParser(description="Write the metadynamics run configuration.")
    # --- Double-well landscape + thermostat (reduced units, kT-scaled energies) ---
    ap.add_argument("--A", type=float, default=5.0, help="barrier height of F0 (kT); minima at s=+/-1")
    ap.add_argument("--kT", type=float, default=1.0, help="thermal energy k_B*T")
    ap.add_argument("--mass", type=float, default=1.0, help="CV particle mass")
    ap.add_argument("--friction", type=float, default=2.0, help="Langevin friction (1/time)")
    ap.add_argument("--dt", type=float, default=0.005, help="integration timestep")
    ap.add_argument("--steps", type=int, default=20000, help="Langevin steps per walker")
    # --- Metadynamics controls ---
    ap.add_argument("--hill-w", type=float, default=0.3, help="Gaussian hill height (energy), pre-tempering")
    ap.add_argument("--hill-sigma", type=float, default=0.10, help="Gaussian hill width along s")
    ap.add_argument("--deposit-every", type=int, default=50, help="deposit a hill every N steps (pace)")
    ap.add_argument("--bias-factor", type=float, default=10.0, help="well-tempered bias factor gamma (>1)")
    # --- Bias grid ---
    ap.add_argument("--s-lo", type=float, default=-2.0, help="grid lower edge (CV)")
    ap.add_argument("--s-hi", type=float, default=2.0, help="grid upper edge (CV)")
    ap.add_argument("--nbins", type=int, default=121, help="grid points (FES resolution)")
    # --- Ensemble ---
    ap.add_argument("--n-walkers", type=int, default=64, help="independent walkers (GPU threads)")
    ap.add_argument("--seed", type=int, default=20240617, help="base RNG seed")
    ap.add_argument("--s-start", type=float, default=1.0, help="|start| CV; walkers alternate +/-")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    fields = [
        f"{args.A:g}", f"{args.kT:g}", f"{args.mass:g}", f"{args.friction:g}",
        f"{args.dt:g}", f"{args.steps}", f"{args.hill_w:g}", f"{args.hill_sigma:g}",
        f"{args.deposit_every}", f"{args.bias_factor:g}",
        f"{args.s_lo:g}", f"{args.s_hi:g}", f"{args.nbins}",
        f"{args.n_walkers}", f"{args.seed}", f"{args.s_start:g}",
    ]
    line = " ".join(fields)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(line + "\n", encoding="utf-8")
    hills = args.steps // args.deposit_every
    print(f"[make_synthetic] wrote {args.out}")
    print(f"  {args.n_walkers} walkers x {args.steps} steps; barrier A={args.A} kT, "
          f"gamma={args.bias_factor}; ~{hills} hills/walker  (SYNTHETIC)")


if __name__ == "__main__":
    main()
