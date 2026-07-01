#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic denoising sample
# ---------------------------------------------------------------------------
# Project 4.9 : Image Denoising & Restoration  (Non-Local Means)
#
# WHY THIS EXISTS
#   Real medical images (the AAPM Low-Dose CT pairs, TCIA/NLST, the N2V
#   fluorescence set) cannot be redistributed inside this repo (license /
#   registration). So we ship a small, CLEARLY-SYNTHETIC phantom the demo can
#   run offline, and this script regenerates it deterministically. The output
#   is labeled SYNTHETIC everywhere (data/README.md) and makes NO clinical claim.
#
#   The phantom is engineered so the result is INTERPRETABLE (PATTERNS.md §6):
#   it has large FLAT regions (where denoising should clearly raise PSNR) and
#   sharp EDGES (where a naive blur would fail but NLM should preserve them),
#   so the demo's "PSNR improvement" number is meaningful, not incidental.
#
# THE FILE FORMAT (matches load_problem() in src/reference_cpu.cpp)
#   line 1:      width height patch_radius search_radius sigma h
#   next H rows: the NOISY image  (W floats per row, values in [0,1])
#   next H rows: the CLEAN image  (W floats per row, the ground truth)
#
# DETERMINISM
#   We seed Python's RNG with a FIXED seed so the committed sample -- and any
#   regeneration of it -- is byte-reproducible. No numpy dependency (pure stdlib)
#   so it runs on any Python 3.
#
# USAGE
#   python scripts/make_synthetic.py                 # writes data/sample/phantom_sample.txt
#   python scripts/make_synthetic.py --size 48       # larger phantom
#   python scripts/make_synthetic.py --sigma 0.12    # more noise
# ===========================================================================
import argparse
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent            # the project folder
OUT  = ROOT / "data" / "sample" / "phantom_sample.txt"

# Fixed seed => the committed sample is identical every regeneration.
SEED = 20240609


def clean_pixel(r, c, size):
    """Ground-truth intensity of pixel (r,c) in a `size`x`size` phantom.

    A Shepp-Logan-flavoured toy: a dark background (0.15), a big bright disk
    (0.75) centred in the frame, and a small darker square inset (0.45) so there
    are BOTH a curved edge and a straight edge for the denoiser to preserve.
    """
    cx = cy = (size - 1) / 2.0
    # Big central disk.
    disk_r = size * 0.34
    inside_disk = (r - cy) ** 2 + (c - cx) ** 2 <= disk_r ** 2
    # Small square inset in the upper-left quadrant of the disk.
    q0, q1 = int(size * 0.30), int(size * 0.46)
    inside_square = (q0 <= r < q1) and (q0 <= c < q1)

    if inside_square:
        return 0.45
    if inside_disk:
        return 0.75
    return 0.15


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic NLM phantom sample.")
    ap.add_argument("--size", type=int, default=32, help="image side length (pixels)")
    ap.add_argument("--patch-radius", type=int, default=2, help="NLM patch radius (patch is (2r+1)^2)")
    ap.add_argument("--search-radius", type=int, default=5, help="NLM search radius (window is (2r+1)^2)")
    ap.add_argument("--sigma", type=float, default=0.08, help="Gaussian noise std-dev (in [0,1] units)")
    ap.add_argument("--h", type=float, default=None, help="NLM filter strength (default: 1.2*sigma)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    size  = args.size
    sigma = args.sigma
    # A common heuristic: filter strength scales with the noise level.
    h = args.h if args.h is not None else 1.2 * sigma

    rng = random.Random(SEED)

    clean_rows, noisy_rows = [], []
    for r in range(size):
        crow, nrow = [], []
        for c in range(size):
            truth = clean_pixel(r, c, size)
            # Additive Gaussian noise, clamped to the valid [0,1] display range.
            noisy = truth + rng.gauss(0.0, sigma)
            noisy = min(1.0, max(0.0, noisy))
            crow.append(truth)
            nrow.append(noisy)
        clean_rows.append(crow)
        noisy_rows.append(nrow)

    def fmt_rows(rows):
        return "\n".join(" ".join(f"{v:.6f}" for v in row) for row in rows)

    header = f"{size} {size} {args.patch_radius} {args.search_radius} {sigma:.6f} {h:.6f}"
    body = header + "\n" + fmt_rows(noisy_rows) + "\n" + fmt_rows(clean_rows) + "\n"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    print(f"[make_synthetic] wrote {out}  "
          f"({size}x{size}, sigma={sigma}, h={h:.4f}, patch_r={args.patch_radius}, "
          f"search_r={args.search_radius}; SYNTHETIC, seed={SEED})")


if __name__ == "__main__":
    main()
