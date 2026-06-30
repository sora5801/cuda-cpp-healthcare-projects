#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic 3D "CT" volume + GT mask
# ---------------------------------------------------------------------------
# Project 4.7 : Medical Image Segmentation (Deep Learning)   [REDUCED SCOPE]
#
# Builds a clearly-SYNTHETIC, low-contrast 3D volume that stands in for a small
# CT/MRI crop, with a single bright spherical "lesion" embedded at a known
# location, plus mild additive noise and a soft "tissue" background gradient.
# Because we KNOW where the sphere is, we also write the ground-truth 0/1 mask,
# which the demo uses to report a Dice accuracy for the network's prediction.
# Real labelled volumes come from the Medical Segmentation Decathlon,
# TotalSegmentator, KiTS23, BraTS, etc. (see download_data.* / data/README.md).
#
# OUTPUT (data/README.md format):
#     D H W
#     <D*H*W intensity floats, row-major (z,y,x), x fastest>
#     <D*H*W ground-truth labels 0/1, same order>
#
# The default size is intentionally tiny (12 x 16 x 16 = 3072 voxels) so the
# committed sample is small and the demo's central-slice ASCII map is readable.
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --D 16 --H 24 --W 24 --radius 4.0 --noise 0.03
# ===========================================================================
import argparse
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "volume_sample.txt"


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic 3D CT-like volume with a lesion sphere.")
    ap.add_argument("--D", type=int, default=12, help="depth  (z slices)")
    ap.add_argument("--H", type=int, default=16, help="height (y rows)")
    ap.add_argument("--W", type=int, default=16, help="width  (x cols)")
    ap.add_argument("--radius", type=float, default=3.0, help="lesion sphere radius (voxels)")
    ap.add_argument("--contrast", type=float, default=0.8, help="lesion brightness above tissue")
    ap.add_argument("--noise", type=float, default=0.02, help="additive Gaussian noise std")
    ap.add_argument("--seed", type=int, default=7, help="RNG seed (determinism)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    D, H, W = args.D, args.H, args.W
    # Lesion center: roughly the middle of the volume (slightly off-center so the
    # result is not symmetric/trivial). Coordinates in (z,y,x).
    cz, cy, cx = D / 2.0, H / 2.0 - 1.0, W / 2.0 + 1.0
    r = args.radius

    inten = [0.0] * (D * H * W)
    truth = [0] * (D * H * W)

    def at(z, y, x):
        return (z * H + y) * W + x

    for z in range(D):
        for y in range(H):
            for x in range(W):
                i = at(z, y, x)
                # Soft tissue background: a gentle intensity gradient (~0.25..0.35)
                # so the volume is not flat -- a mild "anatomy" baseline.
                base = 0.25 + 0.05 * (y / max(1, H - 1)) + 0.05 * (x / max(1, W - 1))
                val = base
                # Distance from the lesion center; voxels inside r are GT lesion.
                dz, dy, dx = z - cz, y - cy, x - cx
                dist = math.sqrt(dz * dz + dy * dy + dx * dx)
                if dist <= r:
                    truth[i] = 1
                # Bright lesion sphere with a soft cosine-tapered edge (like a real
                # partial-volume boundary): peaks at the center, fades out by r+1.5.
                if dist <= r + 1.5:
                    taper = 0.5 * (1.0 + math.cos(math.pi * min(1.0, dist / (r + 1.5))))
                    val += args.contrast * taper
                # Mild Gaussian noise (the thing the layer-1 smoother removes).
                val += rng.gauss(0.0, args.noise)
                inten[i] = val

    # Write: header, then intensities, then ground-truth labels.
    lines = [f"{D} {H} {W}"]
    lines += [f"{v:.6f}" for v in inten]
    lines += [str(t) for t in truth]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")

    nfg = sum(truth)
    print(f"[make_synthetic] wrote {args.out}")
    print(f"[make_synthetic]   SYNTHETIC volume {D}x{H}x{W} = {D*H*W} voxels, "
          f"lesion sphere r={r} -> {nfg} ground-truth lesion voxels (seed={args.seed})")


if __name__ == "__main__":
    main()
