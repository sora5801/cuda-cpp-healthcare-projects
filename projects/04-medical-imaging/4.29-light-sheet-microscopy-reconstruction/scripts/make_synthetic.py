#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic light-sheet sample
# ---------------------------------------------------------------------------
# Project 4.29 : Light-Sheet Microscopy Reconstruction
#
# WHY THIS EXISTS
#   Real LSFM datasets are terabyte-scale and often require registration or a
#   data-use agreement (see data/README.md for the real sources). So that the
#   demo RUNS OFFLINE with a meaningful, verifiable result, this script builds a
#   tiny, clearly-SYNTHETIC "microscopy plane":
#
#     1. A known GROUND-TRUTH image: a few bright, sub-diffraction "beads"
#        (point-like fluorophores) on a dim background -- the kind of structure a
#        deconvolution is meant to recover.
#     2. Blur it with the SAME Gaussian point-spread function (PSF) the program
#        uses (origin-centered, wrap-around) -> the "optically blurred" image.
#     3. Add mild, DETERMINISTIC shot-like noise (a fixed pseudo-random sequence,
#        no external RNG library) so the sample is reproducible bit-for-bit.
#
#   The program then runs Richardson-Lucy deconvolution on this blurry image and
#   should SHARPEN the beads back toward the ground truth (peak/L2 rise). Because
#   we know the ground truth, "did it sharpen?" is a real, checkable question.
#
#   Everything here is SYNTHETIC and labeled as such (CLAUDE.md 8). No real
#   patient or specimen data is used or implied.
#
# OUTPUT FORMAT (matches src/reference_cpu.cpp load_lsfm and data/README.md):
#   line 1 : H W SIGMA ITERS
#   then   : H*W blurred+noisy pixel values (row-major), whitespace-separated.
#
# USAGE
#   python scripts/make_synthetic.py                 # default 32x32, sigma=1.6, 12 iters
#   python scripts/make_synthetic.py --h 48 --w 48   # a larger synthetic plane
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "lsfm_sample.txt"


def gaussian_psf(H, W, sigma):
    """Origin-centered, wrap-around, unit-sum Gaussian PSF as a flat row-major
    list. Identical convention to src/reference_cpu.cpp gaussian_psf() so the
    generated blur matches exactly what the program deconvolves against."""
    two_s2 = 2.0 * sigma * sigma
    h = [0.0] * (H * W)
    total = 0.0
    for r in range(H):
        dr = min(r, H - r)                 # wrap-around distance along rows
        for c in range(W):
            dc = min(c, W - c)             # wrap-around distance along cols
            v = math.exp(-(dr * dr + dc * dc) / two_s2)
            h[r * W + c] = v
            total += v
    return [v / total for v in h]          # normalize to unit sum (flux-preserving)


def circular_convolve(img, psf, H, W):
    """Direct circular convolution img (conv) psf, matching the FFT convention.
    O((H*W)^2) -- fine for the tiny synthetic image. Kept simple and explicit so
    the generation is as transparent as the reconstruction."""
    out = [0.0] * (H * W)
    for p in range(H):
        for q in range(W):
            acc = 0.0
            for r in range(H):
                br = (p - r) % H
                for c in range(W):
                    bc = (q - c) % W
                    acc += psf[r * W + c] * img[br * W + bc]
            out[p * W + q] = acc
    return out


def lcg(seed):
    """A tiny deterministic linear-congruential generator (Numerical Recipes
    constants). We avoid numpy/random so the sample is byte-identical on every
    machine and Python version. Yields floats in [0,1)."""
    state = seed & 0xFFFFFFFF
    while True:
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        yield state / 4294967296.0


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic LSFM sample.")
    ap.add_argument("--h", type=int, default=32, help="image height (rows)")
    ap.add_argument("--w", type=int, default=32, help="image width (cols)")
    ap.add_argument("--sigma", type=float, default=1.6, help="Gaussian PSF sigma (px)")
    ap.add_argument("--iters", type=int, default=12, help="RL iterations the program runs")
    ap.add_argument("--noise", type=float, default=0.02, help="relative noise amplitude")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()
    H, W = args.h, args.w

    # --- 1. Ground truth: a handful of bright beads on a dim background --------
    truth = [0.05] * (H * W)               # faint uniform background
    # Bead positions (row, col, brightness), placed proportionally so they scale
    # with the image size. These are the "true" fluorophores to be recovered.
    beads = [
        (0.30, 0.30, 4.0),
        (0.30, 0.70, 3.0),
        (0.65, 0.50, 5.0),
        (0.50, 0.25, 2.5),
        (0.75, 0.78, 3.5),
    ]
    for fr, fc, amp in beads:
        r = min(H - 1, int(round(fr * H)))
        c = min(W - 1, int(round(fc * W)))
        truth[r * W + c] += amp            # a single-pixel point source

    # --- 2. Blur with the microscope PSF --------------------------------------
    psf = gaussian_psf(H, W, args.sigma)
    blurred = circular_convolve(truth, psf, H, W)

    # --- 3. Add deterministic, mild multiplicative noise ----------------------
    #   Shot noise scales with intensity; we approximate it with a small relative
    #   perturbation from the fixed LCG so brighter pixels get slightly more noise.
    rng = lcg(seed=20240629)               # fixed seed -> reproducible sample
    noisy = []
    for v in blurred:
        u = next(rng) - 0.5                # in [-0.5, 0.5)
        noisy.append(max(0.0, v * (1.0 + args.noise * u)))   # clamp non-negative

    # --- 4. Write the sample --------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"{H} {W} {args.sigma:g} {args.iters}\n")
        # One row of pixels per line for readability (loader ignores line breaks).
        for r in range(H):
            row = noisy[r * W:(r + 1) * W]
            f.write(" ".join(f"{v:.6f}" for v in row) + "\n")

    print(f"[make_synthetic] wrote {out_path}  "
          f"({H}x{W}, sigma={args.sigma}, iters={args.iters}; SYNTHETIC)")


if __name__ == "__main__":
    main()
