#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write a synthetic coarse-grained MD system
# ---------------------------------------------------------------------------
# Project 2.5 : Coarse-Grained / MARTINI Simulation
#
# WHAT THIS WRITES (a tiny, fully SYNTHETIC CG system -- no patient data)
#   A cubic box of MARTINI-like beads of two types:
#     - type 0 = "C" (apolar / lipid-tail-like) and
#     - type 1 = "P" (polar  / water-like).
#   We place the C beads in a compact slab on one side of the box and the P
#   beads on the other, all at rest. The interaction matrix favours like-like
#   contacts (epsCC, epsPP > epsCP), so during the simulation the two species
#   stay demixed and tighten into clusters -- a miniature of the oil/water
#   behaviour that makes MARTINI membrane simulations interesting.
#
# OUTPUT FORMAT (see ../data/README.md):
#   line 1 : n box dt steps rcut mass sigma epsCC epsCP epsPP
#   n lines: x y z vx vy vz type
#
# DETERMINISTIC: positions are on a fixed lattice (no RNG), so the committed
# sample -- and therefore demo/expected_output.txt -- is reproducible.
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --per-side 3 --steps 400
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "cg_system.txt"


def build(per_side, box, sigma):
    """Build two lattices of C and P beads on opposite sides of the box.

    `per_side` controls the lattice: each species gets a per_side^3 block. The C
    block is centred near x = box*0.30 and the P block near x = box*0.70, so the
    two start clearly separated but close enough to feel each other within rcut.
    Returns a list of (x, y, z, vx, vy, vz, type) tuples; velocities are zero.
    """
    beads = []
    spacing = 1.05 * sigma           # nearest-neighbour spacing ~ one bead diameter
    span = (per_side - 1) * spacing  # extent of one block along an axis
    # Centre each block in y and z; offset the two species in x.
    cy = 0.5 * box - 0.5 * span
    cz = 0.5 * box - 0.5 * span
    for btype, xc in ((0, 0.30 * box), (1, 0.70 * box)):
        x0 = xc - 0.5 * span
        for ix in range(per_side):
            for iy in range(per_side):
                for iz in range(per_side):
                    x = x0 + ix * spacing
                    y = cy + iy * spacing
                    z = cz + iz * spacing
                    beads.append((x, y, z, 0.0, 0.0, 0.0, btype))
    return beads


def main():
    ap = argparse.ArgumentParser(description="Write a synthetic CG-MD system.")
    ap.add_argument("--per-side", type=int, default=2,
                    help="beads per axis per species (n = 2*per_side^3)")
    ap.add_argument("--box", type=float, default=6.0, help="cubic box edge (nm)")
    ap.add_argument("--dt", type=float, default=0.005, help="timestep")
    ap.add_argument("--steps", type=int, default=200, help="number of MD steps")
    ap.add_argument("--rcut", type=float, default=2.5, help="non-bonded cutoff (nm)")
    ap.add_argument("--mass", type=float, default=1.0, help="bead mass")
    ap.add_argument("--sigma", type=float, default=0.47, help="LJ sigma (nm)")
    ap.add_argument("--epsCC", type=float, default=4.0, help="C-C well depth")
    ap.add_argument("--epsCP", type=float, default=1.0, help="C-P well depth")
    ap.add_argument("--epsPP", type=float, default=4.0, help="P-P well depth")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    beads = build(args.per_side, args.box, args.sigma)
    n = len(beads)

    lines = [f"{n} {args.box:g} {args.dt:g} {args.steps} {args.rcut:g} "
             f"{args.mass:g} {args.sigma:g} "
             f"{args.epsCC:g} {args.epsCP:g} {args.epsPP:g}"]
    for (x, y, z, vx, vy, vz, t) in beads:
        lines.append(f"{x:.6f} {y:.6f} {z:.6f} {vx:.1f} {vy:.1f} {vz:.1f} {t}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"({n} beads: {n//2} C + {n//2} P, box {args.box} nm, {args.steps} steps; SYNTHETIC)")


if __name__ == "__main__":
    main()
