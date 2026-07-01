#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic STORM/PALM frame stack
# ---------------------------------------------------------------------------
# Project 4.10 : Super-Resolution Microscopy Reconstruction  (SMLM)
#
# Builds a tiny SMLM movie the way a real STORM/PALM acquisition works: a fixed
# underlying STRUCTURE of fluorophore sites (here two thin crossing lines -- a
# stand-in for, say, microtubules), imaged over many frames where only a SPARSE,
# RANDOM subset of sites "blinks" on in each frame. Each on-site is rendered as a
# 2D Gaussian PSF (sigma ~ FIT_SIGMA in smlm.h) plus a flat background and a
# little read noise. Because each frame is sparse, the blobs are separated and
# the localizer can recover each site's sub-pixel centre; overlaying the
# localizations from all frames reconstructs the fine structure -- the whole
# point of SMLM.
#
# The structure is SUB-PIXEL (lines drawn at fractional pixel positions), so a
# correct localizer + super-resolution render recovers a sharp line that the raw
# diffraction-limited frames cannot show. This makes the demo interpretable.
#
# DETERMINISTIC: fixed RNG seed -> the committed sample is reproducible, and the
# pixel intensities are written at fixed precision so the C++ loader reads the
# same doubles every time (keeping CPU==GPU verification exact).
#
# OUTPUT (data/README.md format):
#   header:  "F H W background threshold"
#   body:    F*H*W floats, row-major per frame, one frame after another.
#
# USAGE
#   python scripts/make_synthetic.py                 # the committed sample
#   python scripts/make_synthetic.py --frames 200    # bigger movie
#   python scripts/make_synthetic.py --width 64      # bigger frames
# ===========================================================================
import argparse
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "smlm_stack.txt"

# --- Fixed imaging parameters (must match smlm.h where noted) --------------
PSF_SIGMA = 1.3     # Gaussian PSF width in pixels  (== FIT_SIGMA in smlm.h)
AMPLITUDE = 220.0   # peak photons of an on-emitter (well above threshold)
BACKGROUND = 40.0   # flat background level (camera offset + haze)
THRESHOLD = 100.0   # detection threshold: a candidate pixel must exceed this
NOISE = 3.0         # small Gaussian read-noise std (keeps the demo stable)


def add_psf(frame, H, W, x, y, amp):
    """Add a 2D Gaussian PSF centred at sub-pixel (x, y) into `frame`.
    Only touches pixels within ~3*sigma so the cost stays O(1) per emitter."""
    two_s2 = 2.0 * PSF_SIGMA * PSF_SIGMA
    reach = int(math.ceil(3.0 * PSF_SIGMA))
    c0, r0 = int(round(x)), int(round(y))
    for r in range(max(0, r0 - reach), min(H, r0 + reach + 1)):
        for c in range(max(0, c0 - reach), min(W, c0 + reach + 1)):
            dx = c - x
            dy = r - y
            frame[r * W + c] += amp * math.exp(-(dx * dx + dy * dy) / two_s2)


def build_sites(H, W):
    """Ground-truth fluorophore sites: two crossing lines at sub-pixel offsets.
    Returns a list of (x, y) emitter positions. Kept clear of the frame border
    so every site can hold a full 7x7 fitting patch."""
    sites = []
    margin = 5
    # Line 1: near-horizontal, y = 0.3*x + b1 (fractional slope/intercept).
    b1 = H * 0.35 + 0.4
    for x in range(margin, W - margin):
        y = 0.30 * x + b1 * 0.5
        if margin <= y <= H - margin:
            sites.append((x + 0.25, y + 0.15))
    # Line 2: near-vertical, x = 0.2*y + b2.
    b2 = W * 0.30 + 0.6
    for y in range(margin, H - margin):
        x = 0.20 * y + b2 * 0.5
        if margin <= x <= W - margin:
            sites.append((x + 0.35, y + 0.10))
    return sites


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic SMLM frame stack.")
    ap.add_argument("--frames", type=int, default=60, help="number of camera frames")
    ap.add_argument("--width", type=int, default=40, help="frame width (pixels)")
    ap.add_argument("--height", type=int, default=40, help="frame height (pixels)")
    ap.add_argument("--on-prob", type=float, default=0.06,
                    help="probability each site blinks ON in a given frame (sparsity)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    H, W, F = args.height, args.width, args.frames
    rng = random.Random(args.seed)
    sites = build_sites(H, W)

    lines = [f"{F} {H} {W} {BACKGROUND:.1f} {THRESHOLD:.1f}"]
    for _ in range(F):
        frame = [BACKGROUND] * (H * W)
        # Sparse activation: each site independently blinks on with p=on_prob.
        for (x, y) in sites:
            if rng.random() < args.on_prob:
                add_psf(frame, H, W, x, y, AMPLITUDE)
        # A little read noise so the demo is realistic but still deterministic
        # (fixed seed) and robust (noise << signal, so detections don't flicker).
        for i in range(H * W):
            frame[i] += rng.gauss(0.0, NOISE)
            if frame[i] < 0.0:
                frame[i] = 0.0
        # Write at fixed precision so the C++ loader reads identical doubles.
        lines.append(" ".join(f"{v:.3f}" for v in frame))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(F={F} frames, {H}x{W} px, {len(sites)} ground-truth sites; SYNTHETIC)")


if __name__ == "__main__":
    main()
