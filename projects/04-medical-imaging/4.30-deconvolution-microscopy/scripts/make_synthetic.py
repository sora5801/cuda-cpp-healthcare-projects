#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic blurred-image sample
# ---------------------------------------------------------------------------
# Project 4.30 : Deconvolution Microscopy
#
# WHY THIS EXISTS
#   Real fluorescence-microscopy datasets (BioImage Archive, EPFL benchmarks)
#   are large and/or licensed; we cannot redistribute them. So the committed
#   demo runs on a TINY, clearly-SYNTHETIC image generated here. Everything is
#   deterministic (no RNG) so the demo's expected_output.txt is stable.
#
# WHAT IT BUILDS
#   1. A sharp ground-truth "specimen": a few bright point sources (mimicking
#      sub-resolution fluorescent beads / puncta) plus thin lines, on a small
#      grid. Points and edges are exactly the high-frequency content that blur
#      destroys and deconvolution restores -- so the demo is interpretable.
#   2. The microscope blur: CIRCULAR convolution with a normalized Gaussian PSF
#      (radius r, std sigma). This is the SAME operator (and the same PSF
#      parameters) the C++ code deconvolves with -- see the constants below and
#      in src/main.cu (PSF_RADIUS, PSF_SIGMA). They MUST stay in sync.
#   3. We write ONLY the blurred image (what a microscope would record). The
#      C++ program never sees the ground truth; it must recover the sharpness.
#
#   Output format (matches load_image() in src/reference_cpu.cpp):
#       header: "<w> <h>"
#       then h rows of w space-separated floats (the blurred intensities).
#
# USAGE
#   python scripts/make_synthetic.py                 # default 48x48 sample
#   python scripts/make_synthetic.py --w 96 --h 96   # a bigger synthetic image
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "blurred_image.txt"

# --- Blur model: keep in lockstep with src/main.cu (PSF_RADIUS, PSF_SIGMA) ---
PSF_RADIUS = 4
PSF_SIGMA = 1.5


def gaussian_psf(r, sigma):
    """Normalized (sum=1) 2-D Gaussian PSF as a (2r+1)x(2r+1) list-of-lists.
    Identical formula to make_gaussian_psf() in reference_cpu.cpp."""
    d = 2 * r + 1
    k = [[0.0] * d for _ in range(d)]
    two_sigma2 = 2.0 * sigma * sigma
    s = 0.0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            g = math.exp(-(dx * dx + dy * dy) / two_sigma2)
            k[dy + r][dx + r] = g
            s += g
    for dy in range(d):
        for dx in range(d):
            k[dy][dx] /= s
    return k


def make_ground_truth(w, h):
    """A sharp synthetic specimen: bright point sources + thin bright lines on a
    dim background. All high-frequency, deterministic, clearly synthetic."""
    img = [[2.0] * w for _ in range(h)]          # dim uniform background
    # Bright point sources (like fluorescent beads), placed at fixed spots.
    points = [
        (w // 4, h // 4, 100.0),
        (3 * w // 4, h // 4, 80.0),
        (w // 2, h // 2, 120.0),
        (w // 4, 3 * h // 4, 90.0),
        (3 * w // 4, 3 * h // 4, 110.0),
    ]
    for (px, py, val) in points:
        if 0 <= px < w and 0 <= py < h:
            img[py][px] = val
    # A thin horizontal and a thin vertical bright line (sharp edges).
    yline = h // 3
    for x in range(w):
        img[yline][x] = max(img[yline][x], 40.0)
    xline = 2 * w // 3
    for y in range(h):
        img[y][xline] = max(img[y][xline], 40.0)
    return img


def convolve_circular(img, k, w, h, r):
    """Circular 2-D convolution (matches convolve_circular() in the C++ ref)."""
    d = 2 * r + 1
    out = [[0.0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            acc = 0.0
            for dy in range(-r, r + 1):
                sy = (y + dy) % h
                for dx in range(-r, r + 1):
                    sx = (x + dx) % w
                    acc += img[sy][sx] * k[dy + r][dx + r]
            out[y][x] = acc
    return out


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic blurred microscopy image.")
    ap.add_argument("--w", type=int, default=48, help="image width (pixels)")
    ap.add_argument("--h", type=int, default=48, help="image height (pixels)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    w, h = args.w, args.h
    psf = gaussian_psf(PSF_RADIUS, PSF_SIGMA)
    truth = make_ground_truth(w, h)
    blurred = convolve_circular(truth, psf, w, h, PSF_RADIUS)

    lines = [f"{w} {h}"]
    for y in range(h):
        lines.append(" ".join(f"{blurred[y][x]:.6f}" for x in range(w)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  ({w}x{h}, Gaussian PSF r={PSF_RADIUS} "
          f"sigma={PSF_SIGMA}; SYNTHETIC -- blurred only, ground truth withheld)")


if __name__ == "__main__":
    main()
