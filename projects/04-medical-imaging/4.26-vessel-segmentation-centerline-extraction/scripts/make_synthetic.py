#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write a tiny 3-D volume with an embedded vessel
# ---------------------------------------------------------------------------
# Project 4.26 : Vessel Segmentation & Centerline Extraction
#
# WHAT THIS MAKES (labelled SYNTHETIC everywhere)
#   A small 3-D intensity volume containing one BRIGHT, straight, cylindrical
#   "vessel" running along the x-axis, on a dark background, plus a little
#   deterministic (seeded) noise. This is the ideal Frangi test case: the filter
#   should light up along the tube's centerline and a threshold should recover
#   roughly the tube's cross-section -- an interpretable, checkable result.
#
#   We keep it TINY (default 24x16x16 = 6144 voxels) so the demo runs offline in
#   milliseconds and expected_output.txt stays small. Real CTA volumes are
#   ~512x512x300 = ~10^8 voxels (see data/README.md / download_data).
#
# INTENSITY MODEL
#   background  = bg
#   vessel      = amp * exp(-(r/radius)^2)   for a smooth Gaussian tube of radius
#                 `radius` centered on the axis through (any x, cy, cz); r = dist.
#   noise       = uniform in [-noise, +noise], from a FIXED seed -> reproducible.
#
# OUTPUT FORMAT (matches load_volume in src/reference_cpu.cpp)
#   line 1:  nx ny nz sigma alpha beta c bright mask_threshold
#   then  :  nx*ny*nz floats, row-major (x fastest, then y, then z)
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --nx 48 --ny 32 --nz 32 --radius 3
# ===========================================================================
import argparse
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "vessel_volume.txt"


def main():
    ap = argparse.ArgumentParser(description="Write a synthetic 3-D vessel volume.")
    ap.add_argument("--nx", type=int, default=24, help="voxels along x (vessel axis)")
    ap.add_argument("--ny", type=int, default=16, help="voxels along y")
    ap.add_argument("--nz", type=int, default=16, help="voxels along z")
    ap.add_argument("--radius", type=float, default=2.0, help="vessel radius (voxels)")
    ap.add_argument("--bg", type=float, default=20.0, help="background intensity")
    ap.add_argument("--amp", type=float, default=200.0, help="vessel peak amplitude")
    ap.add_argument("--noise", type=float, default=2.0, help="uniform noise half-range")
    ap.add_argument("--sigma", type=float, default=1.5, help="Frangi smoothing scale")
    ap.add_argument("--alpha", type=float, default=0.5, help="Frangi R_A sensitivity")
    ap.add_argument("--beta", type=float, default=0.5, help="Frangi R_B sensitivity")
    ap.add_argument("--c", type=float, default=15.0, help="Frangi structureness scale")
    ap.add_argument("--bright", type=int, default=1, help="1=bright vessels (CTA)")
    ap.add_argument("--mask", type=float, default=0.5, help="vesselness mask threshold")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (reproducible)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)  # fixed seed -> byte-identical volume each run
    cy = (args.ny - 1) / 2.0        # vessel axis passes through the y,z center
    cz = (args.nz - 1) / 2.0

    vals = []
    # Row-major: x fastest, then y, then z (must match vox_idx in frangi.h).
    for z in range(args.nz):
        for y in range(args.ny):
            for x in range(args.nx):
                # Distance from the vessel axis (the axis runs along x at (cy,cz)).
                r = math.hypot(y - cy, z - cz)
                tube = args.amp * math.exp(-(r * r) / (args.radius * args.radius))
                noise = rng.uniform(-args.noise, args.noise)
                vals.append(args.bg + tube + noise)

    header = (f"{args.nx} {args.ny} {args.nz} {args.sigma} "
              f"{args.alpha} {args.beta} {args.c} {args.bright} {args.mask}")
    body = "\n".join(f"{v:.4f}" for v in vals)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(header + "\n" + body + "\n", encoding="utf-8")

    print(f"[make_synthetic] wrote {args.out}")
    print(f"  {args.nx}x{args.ny}x{args.nz} voxels, bright tube radius={args.radius} "
          f"along x at (y={cy:.1f}, z={cz:.1f})")
    print(f"  Frangi: sigma={args.sigma} alpha={args.alpha} beta={args.beta} "
          f"c={args.c} bright={args.bright} mask>={args.mask}")


if __name__ == "__main__":
    main()
