#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic fundus-image sample
# ---------------------------------------------------------------------------
# Project 7.18 : Retinal Fundus AI Screening
#
# WHY THIS EXISTS
#   The real fundus datasets (EyePACS, APTOS, UK Biobank) require Kaggle accounts
#   or credentialed access and forbid blanket redistribution, so we cannot commit
#   a real retinal photo. Instead this script DETERMINISTICALLY generates a
#   clearly-SYNTHETIC colour "fundus-like" image so the demo runs offline. The
#   image is not a real retina and must never be presented as one.
#
# WHAT IT DRAWS (purely illustrative, no clinical meaning)
#   * a warm orange-red background (fundus colour),
#   * a circular vignette (the round camera aperture),
#   * a bright yellowish "optic-disc"-like blob,
#   * a curved brighter "vessel"-like arc,
#   * a few small dark-red "microaneurysm/haemorrhage"-like spots.
#   These give the fixed edge/blob/colour filters in the CNN something to fire on
#   so the CAM heatmap and class scores are interpretable (PATTERNS.md section 6).
#
# OUTPUT FORMAT (matches load_fundus() in src/reference_cpu.cpp)
#   line 1 : "C H W label"   (C=3 RGB, label = ground-truth grade or -1)
#   then   : C*H*W floats in [0,1], CHANNEL-MAJOR then row-major:
#            all of R (row by row), then all of G, then all of B.
#
# USAGE
#   python scripts/make_synthetic.py                 # 32x32, label -1
#   python scripts/make_synthetic.py --size 64       # bigger synthetic image
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "fundus_sample.txt"


def clamp(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def make_image(size, label):
    """Return (C,H,W,data) with data a flat channel-major list of floats."""
    H = W = size
    R = [[0.0] * W for _ in range(H)]
    G = [[0.0] * W for _ in range(H)]
    B = [[0.0] * W for _ in range(H)]
    cx = cy = (size - 1) / 2.0
    radius = size * 0.48                               # aperture radius
    for y in range(H):
        for x in range(W):
            dx, dy = x - cx, y - cy
            r = math.sqrt(dx * dx + dy * dy)
            if r > radius:                             # outside the round aperture -> black
                continue
            # Warm reddish background, slightly darker toward the edge (vignette).
            fade = 1.0 - 0.4 * (r / radius)
            R[y][x] = 0.55 * fade
            G[y][x] = 0.25 * fade
            B[y][x] = 0.15 * fade
    # Optic-disc-like bright blob, upper-left-of-centre.
    ox, oy = size * 0.38, size * 0.42
    for y in range(H):
        for x in range(W):
            d = math.sqrt((x - ox) ** 2 + (y - oy) ** 2)
            if d < size * 0.12:
                R[y][x] = clamp(R[y][x] + 0.35)
                G[y][x] = clamp(G[y][x] + 0.55)
                B[y][x] = clamp(B[y][x] + 0.30)
    # A curved "vessel"-like brighter arc.
    for t in range(size * 4):
        ang = math.pi * 0.15 + t / (size * 4) * math.pi
        vx = int(cx + math.cos(ang) * size * 0.30)
        vy = int(cy + math.sin(ang) * size * 0.30)
        if 0 <= vx < W and 0 <= vy < H:
            R[vy][vx] = clamp(R[vy][vx] + 0.25)
            G[vy][vx] = clamp(G[vy][vx] + 0.10)
    # A few dark-red "microaneurysm/haemorrhage"-like spots (dark in G,B; red high).
    spots = [(0.70, 0.30), (0.55, 0.72), (0.30, 0.62)]
    for fx, fy in spots:
        sx, sy = int(fx * size), int(fy * size)
        for yy in range(max(0, sy - 1), min(H, sy + 2)):
            for xx in range(max(0, sx - 1), min(W, sx + 2)):
                R[yy][xx] = 0.60
                G[yy][xx] = 0.05
                B[yy][xx] = 0.05
    data = []
    for plane in (R, G, B):                            # channel-major: R, then G, then B
        for row in plane:
            data.extend(clamp(v) for v in row)
    return 3, H, W, data


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic fundus sample.")
    ap.add_argument("--size", type=int, default=32, help="square image side in pixels")
    ap.add_argument("--label", type=int, default=-1, help="ground-truth DR grade (-1 unknown)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    C, H, W, data = make_image(args.size, args.label)
    lines = [f"{C} {H} {W} {args.label}"]
    # 16 floats per line for readability; loader ignores line breaks.
    for i in range(0, len(data), 16):
        lines.append(" ".join(f"{v:.4f}" for v in data[i:i + 16]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  ({C}x{H}x{W}, label={args.label}; SYNTHETIC)")


if __name__ == "__main__":
    main()
