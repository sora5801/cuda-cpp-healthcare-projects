#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write the monodomain tissue parameter file
# ---------------------------------------------------------------------------
# Project 6.1 : Cardiac Electrophysiology Simulation
#
# This project's "data" is the SIMULATION SETUP: a square sheet of excitable
# cardiac tissue (FitzHugh-Nagumo cells) coupled by diffusion, sparked by a
# small S1 stimulus patch on the left edge. The solver then propagates an
# action-potential wave across the sheet. This script writes the whitespace-
# separated parameter file the program reads.
#
# All values are SYNTHETIC and dimensionless (the FHN model is a nondimensional
# caricature of real ionic models). They are NOT patient data and NOT for any
# clinical use -- see data/README.md.
#
# OUTPUT (data/README.md format), 14 fields:
#   nx ny steps dt dx D a eps b  stim_x0 stim_y0 stim_w stim_h stim_v
#
# STABILITY: the explicit diffusion step is stable only for dt <= dx^2/(4*D).
# With the defaults below dx=1, D=0.1 -> dt_max=2.5, and dt=0.1 is well inside it.
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --nx 64 --ny 64 --steps 800
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "tissue_params.txt"


def main():
    ap = argparse.ArgumentParser(
        description="Write the monodomain (cardiac EP) tissue parameter file.")
    # --- Grid ---------------------------------------------------------------
    ap.add_argument("--nx", type=int, default=32, help="grid columns (x)")
    ap.add_argument("--ny", type=int, default=32, help="grid rows (y)")
    ap.add_argument("--steps", type=int, default=400,
                    help="operator-split timesteps (react+diffuse per step)")
    # --- Numerics -----------------------------------------------------------
    ap.add_argument("--dt", type=float, default=0.1, help="time step (< dx^2/(4D))")
    ap.add_argument("--dx", type=float, default=1.0, help="grid spacing")
    ap.add_argument("--D", type=float, default=0.2, help="diffusion coefficient")
    # --- FitzHugh-Nagumo cell model -----------------------------------------
    #   a small  : easy to excite (low threshold)         -> the wave launches
    #   eps small: SLOW recovery -> long action potential -> a depolarised
    #              plateau trails the wavefront (the classic AP shape). If eps
    #              is too large the tissue recovers before the wave can spread
    #              and no propagation is seen -- try --eps 0.02 to observe that.
    ap.add_argument("--a", type=float, default=0.10, help="excitation threshold (0<a<1)")
    ap.add_argument("--eps", type=float, default=0.002, help="recovery time-scale")
    ap.add_argument("--b", type=float, default=0.5, help="recovery coupling")
    # --- S1 stimulus patch (left edge) --------------------------------------
    ap.add_argument("--stim_x0", type=int, default=0, help="patch top-left x")
    ap.add_argument("--stim_y0", type=int, default=0, help="patch top-left y")
    ap.add_argument("--stim_w", type=int, default=3, help="patch width (cells)")
    ap.add_argument("--stim_h", type=int, default=32, help="patch height (cells)")
    ap.add_argument("--stim_v", type=float, default=1.0, help="patch clamp voltage")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    fields = [args.nx, args.ny, args.steps,
              args.dt, args.dx, args.D,
              args.a, args.eps, args.b,
              args.stim_x0, args.stim_y0, args.stim_w, args.stim_h, args.stim_v]
    line = " ".join(str(v) for v in fields)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(line + "\n", encoding="utf-8")

    dt_max = (args.dx * args.dx) / (4.0 * args.D)
    print(f"[make_synthetic] wrote {args.out}")
    print(f"  grid {args.nx}x{args.ny}, {args.steps} steps, dt={args.dt} "
          f"(CFL dt_max={dt_max:.3f})")
    print(f"  FHN a={args.a} eps={args.eps} b={args.b}; "
          f"S1 patch {args.stim_w}x{args.stim_h} at "
          f"({args.stim_x0},{args.stim_y0}) V={args.stim_v}")
    if args.dt > dt_max:
        print("  WARNING: dt exceeds the CFL limit; the explicit solver will be "
              "unstable. Reduce dt or D.")


if __name__ == "__main__":
    main()
