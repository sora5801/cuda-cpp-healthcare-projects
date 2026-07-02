#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write the 1-D cable configuration (SYNTHETIC)
# ---------------------------------------------------------------------------
# Project 6.14 : Multi-Scale Physiological Modeling
#
# The "data" here is the simulation SETUP: the cable geometry, the time-stepping
# settings, the left-end stimulus, and the FitzHugh-Nagumo (FHN) cell + tissue
# parameters. There is no measured input -- the whole point is a SELF-CONTAINED,
# reproducible multi-scale toy: an action potential launched at the left end that
# PROPAGATES rightward as a traveling wave. The committed sample is therefore
# explicitly SYNTHETIC (see data/README.md).
#
# OUTPUT (data/README.md format), one line:
#   n dx dt steps stim_nodes a eps b D
#
# USAGE
#   python scripts/make_synthetic.py                 # default demo cable
#   python scripts/make_synthetic.py --n 512 --steps 4000
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "cable.txt"


def main():
    ap = argparse.ArgumentParser(description="Write the 1-D monodomain cable configuration.")
    ap.add_argument("--n", type=int, default=128, help="number of tissue nodes")
    ap.add_argument("--dx", type=float, default=0.5, help="node spacing (space units)")
    ap.add_argument("--dt", type=float, default=0.02, help="global split time step")
    ap.add_argument("--steps", type=int, default=5000, help="number of global steps")
    ap.add_argument("--stim-nodes", type=int, default=5,
                    help="number of left-end nodes held excited at t=0")
    # FitzHugh-Nagumo cell parameters (dimensionless). Tuned (with D below) so the
    # stimulus launches a self-sustaining traveling wave that crosses the cable.
    ap.add_argument("--a", type=float, default=0.13, help="FHN excitation threshold")
    ap.add_argument("--eps", type=float, default=0.005, help="FHN recovery time-scale")
    ap.add_argument("--b", type=float, default=0.50, help="FHN recovery coupling")
    # Tissue diffusion coefficient (space^2 / time). Larger => faster conduction.
    ap.add_argument("--D", type=float, default=2.0, help="tissue diffusion coefficient")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    # Explicit-diffusion stability guard (forward Euler on the Laplacian):
    #   D * dt / dx^2 <= 0.5 for a stable 1-D diffusion step. We warn (not fail)
    #   so the learner can SEE what instability looks like if they push it.
    r = args.D * args.dt / (args.dx * args.dx)
    stability = "OK" if r <= 0.5 else "UNSTABLE (D*dt/dx^2 > 0.5!)"

    line = (f"{args.n} {args.dx:g} {args.dt:g} {args.steps} {args.stim_nodes} "
            f"{args.a:g} {args.eps:g} {args.b:g} {args.D:g}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(line + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}")
    print(f"[make_synthetic] cable: {args.n} nodes, dx={args.dx}, dt={args.dt}, "
          f"{args.steps} steps (T={args.steps * args.dt:g})")
    print(f"[make_synthetic] diffusion number D*dt/dx^2 = {r:.3f}  -> {stability}")


if __name__ == "__main__":
    main()
