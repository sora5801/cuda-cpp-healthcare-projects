#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic CT sinogram
# ---------------------------------------------------------------------------
# Project 4.01 : CT Reconstruction (Filtered Backprojection)
#
# A "phantom" is a synthetic object of known shape; its sinogram is what a CT
# scanner would measure. We use a sum of uniform DISCS because the line integral
# (Radon transform) of a disc is ANALYTIC and exact:
#   a ray at detector offset s sees chord length 2*sqrt(r^2 - (s-c)^2) inside a
#   disc of radius r whose center projects to c = cx*cos(theta) + cy*sin(theta).
# Summing those chords over the discs gives the projection -- no rasterization,
# fully deterministic. (Real data is a measured/standard phantom; see
# download_data.*.)
#
# OUTPUT (data/README.md format):
#   header: "<n_angles> <n_det> <ds> <img> <world_half>"
#   then n_angles rows of n_det floats (the sinogram).
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --angles 360 --det 367 --img 256
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "sinogram_sample.txt"

# Phantom: list of discs (cx, cy, radius, density). Coordinates in the same
# world units as world_half. A big disc plus a few inserts (one "cold" -0.4).
PHANTOM = [
    (0.00,  0.00, 0.60,  1.0),
    (0.00, -0.30, 0.15,  0.6),
    (0.28,  0.18, 0.10, -0.4),
    (-0.30, 0.12, 0.12,  0.5),
    (0.00,  0.35, 0.08,  0.8),
]


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic CT sinogram from a disc phantom.")
    ap.add_argument("--angles", type=int, default=120, help="number of projection angles over [0,pi)")
    ap.add_argument("--det", type=int, default=183, help="detector bins per projection")
    ap.add_argument("--ds", type=float, default=0.012, help="detector bin spacing (world units)")
    ap.add_argument("--img", type=int, default=128, help="reconstruction image side")
    ap.add_argument("--world-half", type=float, default=0.75, help="image spans [-W,W]^2")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    n_angles, n_det, ds = args.angles, args.det, args.ds
    center = 0.5 * (n_det - 1)

    rows = []
    for k in range(n_angles):
        theta = math.pi * k / n_angles
        ct, st = math.cos(theta), math.sin(theta)
        row = []
        for j in range(n_det):
            s = (j - center) * ds
            val = 0.0
            for (cx, cy, r, d) in PHANTOM:
                c = cx * ct + cy * st               # disc center projected onto detector
                dx = s - c
                if abs(dx) < r:
                    val += d * 2.0 * math.sqrt(r * r - dx * dx)   # chord length * density
            row.append(val)
        rows.append(" ".join(f"{v:.6f}" for v in row))

    header = f"{n_angles} {n_det} {ds} {args.img} {args.world_half}"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  ({n_angles} angles x {n_det} det, "
          f"img={args.img}; SYNTHETIC disc phantom)")


if __name__ == "__main__":
    main()
