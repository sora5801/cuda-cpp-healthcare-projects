#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate synthetic landmark heatmaps
# ---------------------------------------------------------------------------
# Project 4.32 : GPU-Accelerated Landmark Detection
#
# WHAT THIS MAKES  (labeled SYNTHETIC everywhere -- no real patient data)
#   A stand-in for a heatmap-regression network's OUTPUT. A real network takes a
#   3D CT/MR volume and emits, per anatomical landmark l, a whole 3D heatmap
#   H_l[z,y,x] that peaks at the landmark. We skip the network and directly write
#   L such heatmaps, each an isotropic Gaussian blob
#         H_l(x,y,z) = exp( -((x-cx)^2 + (y-cy)^2 + (z-cz)^2) / (2 sigma^2) )
#   centred on a KNOWN ground-truth point (cx,cy,cz). Because we know the planted
#   point, the demo can report how well the argmax + soft-argmax decoder recovers
#   it -- a check on the science, not just CPU-vs-GPU agreement.
#
#   The centres are placed at FRACTIONAL voxel positions (e.g. 5.6). The integer
#   argmax can only ever return the nearest voxel (6); the soft-argmax centroid
#   recovers the fractional part -- which is the whole reason sub-voxel decoding
#   exists. Blob values are in [0,1], matching landmark.h's fixed-point weighting.
#
# OUTPUT FORMAT  (see data/README.md; whitespace-separated)
#   line 1:  nx ny nz L
#   per landmark:  cx cy cz   then  nx*ny*nz floats (row-major: x fastest).
#
# USAGE
#   python scripts/make_synthetic.py                 # default tiny sample
#   python scripts/make_synthetic.py --nx 24 --ny 24 --nz 24 --landmarks 8
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "heatmaps_sample.txt"

# A handful of landmarks at deliberately FRACTIONAL centres so soft-argmax has
# something sub-voxel to recover. (x, y, z) in voxel units. Kept inside a margin
# so the decode window never clips the grid edge. Think "vertebral endplate,
# femoral head centre, dental cusp..." -- the anatomical points a real detector
# localises, here purely synthetic.
DEFAULT_CENTERS = [
    (5.60, 6.30, 4.10),
    (12.40, 9.80, 7.70),
    (8.20, 13.50, 10.30),
    (14.90, 4.40, 12.60),
    (10.10, 10.10, 5.50),
]


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def make_heatmap(nx, ny, nz, center, sigma):
    """Return a flat row-major list of nx*ny*nz Gaussian intensities in [0,1]."""
    cx, cy, cz = center
    two_s2 = 2.0 * sigma * sigma
    vals = []
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                r2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
                vals.append(math.exp(-r2 / two_s2))
    return vals


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic landmark heatmaps.")
    ap.add_argument("--nx", type=int, default=20, help="grid extent x (voxels)")
    ap.add_argument("--ny", type=int, default=20, help="grid extent y (voxels)")
    ap.add_argument("--nz", type=int, default=16, help="grid extent z (voxels)")
    ap.add_argument("--landmarks", type=int, default=len(DEFAULT_CENTERS),
                    help="number of landmarks (uses/repeats the default centres)")
    ap.add_argument("--sigma", type=float, default=1.5, help="Gaussian blob width (voxels)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    nx, ny, nz, L = args.nx, args.ny, args.nz, args.landmarks

    # Build the centre list (repeat the defaults, nudged, if L exceeds the table).
    centers = []
    for l in range(L):
        base = DEFAULT_CENTERS[l % len(DEFAULT_CENTERS)]
        # Nudge repeats so they are not identical, staying inside a safe margin.
        shift = 0.37 * (l // len(DEFAULT_CENTERS))
        c = (clamp(base[0] + shift, 3.0, nx - 4.0),
             clamp(base[1] + shift, 3.0, ny - 4.0),
             clamp(base[2] + shift, 3.0, nz - 4.0))
        centers.append(c)

    lines = [f"{nx} {ny} {nz} {L}"]
    for c in centers:
        lines.append(f"{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}")
        vals = make_heatmap(nx, ny, nz, c, args.sigma)
        # 12 values per line for readability; the loader ignores line breaks.
        for i in range(0, len(vals), 12):
            lines.append(" ".join(f"{v:.6f}" for v in vals[i:i + 12]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(SYNTHETIC: {L} landmarks, {nx}x{ny}x{nz} grid, sigma={args.sigma})")


if __name__ == "__main__":
    main()
