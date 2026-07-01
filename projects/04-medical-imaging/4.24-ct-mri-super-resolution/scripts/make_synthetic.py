#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic HR phantom sample
# ---------------------------------------------------------------------------
# Project 4.24 : CT/MRI Super-Resolution   (reduced-scope teaching version)
#
# WHY THIS EXISTS
#   Real CT/MRI SR datasets (HCP, fastMRI, IXI) require registration or forbid
#   redistribution (see data/README.md). So the committed sample is a small,
#   clearly-SYNTHETIC grayscale "phantom" that plays the role of a ground-truth
#   HIGH-RES image. main.cu degrades it (RxR box average) into a low-res image,
#   then super-resolves it back and scores PSNR against this ground truth.
#
#   The phantom is engineered to be INTERPRETABLE for SR: it has smooth blobs
#   (Gaussian disks, like soft-tissue windows) AND sharp features (bars, a ring)
#   so the learner can see the network sharpen edges relative to nearest-
#   neighbour upscaling. It is fully deterministic (no RNG seed dependence).
#
# OUTPUT FORMAT (matches load_image in src/reference_cpu.cpp)
#   line 1 : "<w> <h>"                       (both multiples of SR_SCALE=2)
#   then   : w*h floats in [0,1], row-major, one row of the image per text line.
#
# USAGE
#   python scripts/make_synthetic.py             # writes data/sample/phantom_hr.txt (32x32)
#   python scripts/make_synthetic.py --size 64   # bigger synthetic phantom
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "phantom_hr.txt"


def clamp01(v: float) -> float:
    """Keep intensities in the normalized [0,1] range the loader expects."""
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def make_phantom(size: int):
    """Return a size x size list-of-lists float image in [0,1].

    Composition (all deterministic):
      * a mid-gray background (0.15),
      * two overlapping Gaussian disks (smooth soft-tissue-like blobs),
      * three vertical bars of increasing brightness (a resolution target),
      * a thin bright ring (a sharp high-frequency edge for the SR to recover).
    """
    img = [[0.15 for _ in range(size)] for _ in range(size)]
    cx1, cy1, r1 = size * 0.35, size * 0.40, size * 0.22
    cx2, cy2, r2 = size * 0.62, size * 0.60, size * 0.16
    ring_c, ring_r = (size * 0.70, size * 0.30), size * 0.18

    for y in range(size):
        for x in range(size):
            v = img[y][x]
            # Gaussian disk 1 (bright, broad).
            d1 = ((x - cx1) ** 2 + (y - cy1) ** 2) / (2.0 * r1 * r1)
            v += 0.70 * math.exp(-d1)
            # Gaussian disk 2 (medium).
            d2 = ((x - cx2) ** 2 + (y - cy2) ** 2) / (2.0 * r2 * r2)
            v += 0.45 * math.exp(-d2)
            # Vertical resolution bars in a band near the top.
            if 2 <= y <= size // 4:
                # bars every 3 px; brightness rises left->right
                if (x // 3) % 2 == 0 and 2 <= x <= size - 3:
                    v += 0.10 + 0.30 * (x / size)
            # Thin bright ring (sharp edge): |dist - ring_r| < 1 px.
            dr = math.hypot(x - ring_c[0], y - ring_c[1])
            if abs(dr - ring_r) < 1.0:
                v += 0.55
            img[y][x] = clamp01(v)
    return img


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic HR phantom sample.")
    ap.add_argument("--size", type=int, default=32,
                    help="image side length in pixels (must be even; default 32)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    size = args.size
    if size % 2 != 0:
        raise SystemExit("--size must be even (SR_SCALE=2 requires even dims)")

    img = make_phantom(size)
    lines = [f"{size} {size}"]
    for row in img:
        lines.append(" ".join(f"{v:.4f}" for v in row))

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {outp}  ({size}x{size} HR phantom; SYNTHETIC)")


if __name__ == "__main__":
    main()
