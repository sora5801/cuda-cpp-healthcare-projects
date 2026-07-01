#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write a synthetic fixed/moving image pair
# ---------------------------------------------------------------------------
# Project 4.8 : Deformable Image Registration (reduced-scope teaching version)
#
# WHAT THIS MAKES  (clearly SYNTHETIC -- no patient data anywhere in this repo)
#   A pair of tiny grayscale "images" designed so the registration result is
#   obvious and verifiable:
#     * FIXED  image: a smooth bright BLOB (a 2-D Gaussian bump) centered in the
#                     frame -- stands in for an anatomical structure.
#     * MOVING image: the SAME blob, but SHIFTED by a known displacement and
#                     given a mild non-uniform stretch, so a good deformable
#                     registration must recover a spatially-varying vector field
#                     (not just one global translation).
#   Because both images are smooth Gaussians, the intensity gradient is defined
#   everywhere, which is exactly what Thirion's Demons force needs (a flat image
#   carries no direction to move along). Registering MOVING onto FIXED should
#   drive the sum-of-squared-differences (SSD) far down -- the demo's headline
#   metric.
#
# OUTPUT (data/README.md documents this exact format), whitespace-separated:
#   line 1 : "nx ny"
#   then   : nx*ny fixed  intensities in [0,1], row-major
#   then   : nx*ny moving intensities in [0,1], row-major
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --nx 96 --ny 96 --shift 4.0
#
# Deterministic: no RNG, so the committed sample is reproducible byte-for-byte.
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "dir_pair.txt"


def gaussian_blob(nx, ny, cx, cy, sigma):
    """A smooth 2-D Gaussian bump of unit peak, centered at (cx,cy). Returns a
    row-major list of nx*ny intensities in (0,1]. Smoothness -> a well-defined
    gradient everywhere, which the Demons force relies on."""
    img = [0.0] * (nx * ny)
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    for y in range(ny):
        for x in range(nx):
            dx = x - cx
            dy = y - cy
            img[y * nx + x] = math.exp(-(dx * dx + dy * dy) * inv2s2)
    return img


def main():
    ap = argparse.ArgumentParser(description="Write a synthetic DIR image pair.")
    ap.add_argument("--nx", type=int, default=64, help="image width in pixels")
    ap.add_argument("--ny", type=int, default=64, help="image height in pixels")
    ap.add_argument("--sigma", type=float, default=9.0, help="blob width (pixels)")
    ap.add_argument("--shift", type=float, default=5.0,
                    help="blob translation applied to the MOVING image (pixels)")
    ap.add_argument("--stretch", type=float, default=0.12,
                    help="fractional non-uniform stretch of the moving blob")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    nx, ny = args.nx, args.ny
    cx, cy = nx / 2.0, ny / 2.0

    # FIXED: a centered blob.
    fixed = gaussian_blob(nx, ny, cx, cy, args.sigma)

    # MOVING: the blob moved by +shift in x and +0.6*shift in y, and made
    # slightly wider in x (the stretch) so the true deformation is NOT a pure
    # translation -- forcing the DVF to vary across the image.
    mcx = cx + args.shift
    mcy = cy + 0.6 * args.shift
    msigma_x = args.sigma * (1.0 + args.stretch)
    # Build the moving blob with an anisotropic Gaussian (wider in x).
    moving = [0.0] * (nx * ny)
    inv2sx2 = 1.0 / (2.0 * msigma_x * msigma_x)
    inv2sy2 = 1.0 / (2.0 * args.sigma * args.sigma)
    for y in range(ny):
        for x in range(nx):
            dx = x - mcx
            dy = y - mcy
            moving[y * nx + x] = math.exp(-(dx * dx) * inv2sx2 - (dy * dy) * inv2sy2)

    # Serialize. 6 decimals is plenty for [0,1] intensities and keeps the file
    # small (the whole 64x64 pair is ~55 KB of text, well under any limit).
    lines = [f"{nx} {ny}"]
    lines.append(" ".join(f"{v:.6f}" for v in fixed))
    lines.append(" ".join(f"{v:.6f}" for v in moving))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[make_synthetic] wrote {args.out}")
    print(f"  {nx}x{ny} pair: fixed blob at ({cx:.1f},{cy:.1f}), "
          f"moving blob at ({mcx:.1f},{mcy:.1f}), stretch={args.stretch}")
    print("  synthetic data -- NOT patient-derived, NOT for clinical use.")


if __name__ == "__main__":
    main()
