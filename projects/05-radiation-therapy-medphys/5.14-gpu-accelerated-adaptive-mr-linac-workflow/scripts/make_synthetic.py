#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic oART sample dataset
# ---------------------------------------------------------------------------
# Project 5.14 : GPU-Accelerated Adaptive MR-Linac Workflow (reduced-scope)
#
# WHY THIS EXISTS
#   Real MR-Linac images cannot be redistributed here (patient data + license;
#   see data/README.md for the legitimate sources). So we deterministically
#   synthesize a clearly-labeled 2-D stand-in that exercises the exact code path
#   a clinical slice would: a planning image, a daily image where the anatomy has
#   MOVED, a planned dose, and a GTV (tumour) mask. Everything is SYNTHETIC.
#
#   THE SCENARIO WE ENCODE (so the demo result is interpretable):
#     * FIXED  (planning MR F): a soft elliptical "organ" with a bright round
#       "tumour" disc centred at (cx, cy).
#     * MOVING (daily MR M): the SAME anatomy but rigidly shifted by (dx, dy)
#       voxels -- e.g. a filling bladder pushed the tumour. This is what Demons
#       must recover.
#     * DOSE  : a Gaussian "dose cloud" planned to cover the tumour on F. After we
#       warp it by the recovered field it should again cover the moved tumour.
#     * GTV   : 1.0 inside the tumour disc on F, 0.0 outside.
#
#   Because we know the ground-truth shift, the demo's "peak displacement" and the
#   MSE drop are directly interpretable: the registration recovered the motion.
#
#   Uses only the Python standard library (no numpy) so it runs anywhere.
#
# USAGE
#   python scripts/make_synthetic.py                 # writes data/sample/oart_case.txt
#   python scripts/make_synthetic.py --nx 48 --ny 48 # larger synthetic slice
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "oart_case.txt"


def build(nx, ny, dx, dy):
    """Return (fixed, moving, dose, gtv) as flat row-major lists of floats."""
    cx, cy = nx * 0.5, ny * 0.5          # organ/tumour centre on the planning MR
    organ_rx, organ_ry = nx * 0.35, ny * 0.30   # soft elliptical organ radii
    tumour_r = min(nx, ny) * 0.16        # tumour disc radius (voxels)

    def anatomy(x, y):
        """Intensity of the planning anatomy at (x,y): organ + bright tumour."""
        # Elliptical organ: smooth bump, ~0.4 at centre fading to 0 at the edge.
        ex = (x - cx) / organ_rx
        ey = (y - cy) / organ_ry
        organ = 0.4 * math.exp(-(ex * ex + ey * ey))
        # Tumour: a brighter disc with a soft rim (so gradients exist for Demons).
        r = math.hypot(x - cx, y - cy)
        tumour = 0.6 * (0.5 * (1.0 - math.tanh((r - tumour_r) * 0.9)))
        return organ + tumour

    fixed, moving, dose, gtv = [], [], [], []
    for y in range(ny):
        for x in range(nx):
            # FIXED = planning anatomy.
            fixed.append(anatomy(x, y))
            # MOVING = the SAME anatomy sampled at a shifted coordinate, i.e. the
            # daily image is the planning anatomy translated by (dx,dy).
            moving.append(anatomy(x - dx, y - dy))
            # DOSE = Gaussian cloud centred on the tumour on F, peak 60 Gy.
            r = math.hypot(x - cx, y - cy)
            dose.append(60.0 * math.exp(-(r * r) / (2.0 * (tumour_r * 1.1) ** 2)))
            # GTV = tumour disc on F.
            gtv.append(1.0 if r <= tumour_r else 0.0)
    return fixed, moving, dose, gtv


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic oART sample.")
    ap.add_argument("--nx", type=int, default=32, help="image width (voxels)")
    ap.add_argument("--ny", type=int, default=32, help="image height (voxels)")
    ap.add_argument("--iters", type=int, default=60, help="Demons iterations")
    ap.add_argument("--sigma", type=float, default=1.5, help="Gaussian smoothing sigma")
    ap.add_argument("--k", type=float, default=1.0, help="Thirion normaliser K")
    ap.add_argument("--dx", type=float, default=3.0, help="ground-truth x shift (voxels)")
    ap.add_argument("--dy", type=float, default=2.0, help="ground-truth y shift (voxels)")
    ap.add_argument("--dose-thresh", type=float, default=30.0,
                    help="coverage dose threshold (Gy)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    fixed, moving, dose, gtv = build(args.nx, args.ny, args.dx, args.dy)

    # Format numbers compactly but with enough precision for a stable demo.
    def fmt_block(vals):
        return " ".join(f"{v:.6f}" for v in vals)

    header = f"{args.nx} {args.ny} {args.iters} {args.sigma:g} {args.k:g} {args.dose_thresh:g}"
    lines = [header, fmt_block(fixed), fmt_block(moving), fmt_block(dose), fmt_block(gtv)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"({args.nx}x{args.ny}, shift=({args.dx},{args.dy}) voxels; SYNTHETIC)")


if __name__ == "__main__":
    main()
