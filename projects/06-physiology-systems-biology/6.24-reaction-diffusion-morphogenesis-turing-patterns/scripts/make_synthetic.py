#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write the Turing (Gierer-Meinhardt) config
# ---------------------------------------------------------------------------
# Project 6.24 : Reaction-Diffusion Morphogenesis (Turing Patterns)
#
# WHY THIS EXISTS
#   There is no downloadable "Turing pattern dataset" -- the data IS the model
#   configuration, and the pattern is produced BY the simulation from a
#   deterministic seed (data/README.md explains this). This script writes that
#   one-line configuration so the demo runs offline. The output is clearly
#   SYNTHETIC (a chosen parameter point), never real biological measurement.
#
# THE PARAMETER LINE (matches load_params() in src/reference_cpu.cpp):
#   nx ny Da Dh rho mu_a mu_h rho_a dt steps noise_seed
#
#   The committed default (Da=0.02, Dh=0.5 => Dh/Da=25) sits squarely in the
#   Turing regime: the inhibitor diffuses 25x faster than the activator, so
#   short-range activation + long-range inhibition break the uniform state into
#   a spot/labyrinth pattern (contrast ~7 in the activator field).
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --Dh 0.20 --steps 3000   # weaker inhibition
#   python scripts/make_synthetic.py --nx 128 --ny 128        # finer grid
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "turing_params.txt"


def main():
    ap = argparse.ArgumentParser(description="Write the Gierer-Meinhardt Turing configuration.")
    ap.add_argument("--nx", type=int, default=64, help="grid width (cells)")
    ap.add_argument("--ny", type=int, default=64, help="grid height (cells)")
    ap.add_argument("--Da", type=float, default=0.02, help="activator diffusion (small)")
    ap.add_argument("--Dh", type=float, default=0.5,  help="inhibitor diffusion (large; Dh>>Da for Turing)")
    ap.add_argument("--rho", type=float, default=0.05, help="reaction strength")
    ap.add_argument("--mu-a", type=float, default=0.1, help="activator decay rate")
    ap.add_argument("--mu-h", type=float, default=0.14, help="inhibitor decay rate")
    ap.add_argument("--rho-a", type=float, default=0.0, help="basal activator source")
    ap.add_argument("--dt", type=float, default=0.4, help="explicit-Euler timestep")
    ap.add_argument("--steps", type=int, default=3000, help="number of timesteps")
    ap.add_argument("--noise-seed", type=int, default=12345, help="seed for the deterministic initial noise")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    line = (f"{args.nx} {args.ny} {args.Da:g} {args.Dh:g} {args.rho:g} "
            f"{args.mu_a:g} {args.mu_h:g} {args.rho_a:g} {args.dt:g} "
            f"{args.steps} {args.noise_seed}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(line + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  ({args.nx}x{args.ny} grid, "
          f"{args.steps} steps, Dh/Da={args.Dh/args.Da:.0f} -> Turing pattern; SYNTHETIC)")


if __name__ == "__main__":
    main()
