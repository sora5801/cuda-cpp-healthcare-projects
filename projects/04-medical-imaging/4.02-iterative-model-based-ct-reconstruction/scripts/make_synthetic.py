#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic, NOISY CT sinogram
# ---------------------------------------------------------------------------
# Project 4.2 : Iterative / Model-Based CT Reconstruction
#
# WHY THIS EXISTS
#   Real low-dose CT datasets (AAPM Grand Challenge, Mayo, LIDC-IDRI via TCIA)
#   require registration and cannot be redistributed here (see data/README.md and
#   scripts/download_data.*). So we ship a tiny, clearly-SYNTHETIC phantom whose
#   sinogram the demo reconstructs offline. Everything below is synthetic.
#
# HOW IT WORKS
#   1. Rasterize a disc "phantom" (a big soft-tissue disc with a few inserts) onto
#      the SAME N x N pixel grid the reconstruction uses -> this is the GROUND
#      TRUTH image, which we also write out so the demo can report reconstruction
#      error vs. truth.
#   2. FORWARD-project it with the EXACT same voxel-scatter model the C++ code
#      uses (pixel -> two detector bins via linear interpolation, py-outer /
#      px-inner order, float32 math), so SIRT converges back toward the phantom.
#   3. Add a little Poisson-flavoured NOISE (fixed RNG seed -> reproducible file).
#      Noise is the whole point: iterative + TV reconstruction suppresses noise
#      that plain FBP would leave as streaks -- that is the lesson of Project 4.2.
#
# OUTPUT FORMAT (matches load_ct in src/reference_cpu.cpp; see data/README.md):
#   line 1: "n_angles n_det ds img world_half iters lambda tv_weight has_truth"
#   then n_angles rows of n_det floats   (the NOISY sinogram)
#   then img rows of img floats          (the ground-truth image, has_truth=1)
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --angles 60 --det 91 --img 64 --iters 60
# ===========================================================================
import argparse
import math
from pathlib import Path

try:
    import numpy as np
except ImportError as e:  # numpy keeps this readable and fast; require it
    raise SystemExit("make_synthetic.py needs numpy:  pip install numpy") from e

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "sinogram_sample.txt"

# Phantom: a list of discs (cx, cy, radius, density) in world units. A big
# low-density disc (soft tissue) with a couple of brighter and one "cold" insert
# -- enough structure that the reconstruction is visually and numerically checkable.
PHANTOM = [
    (0.00,  0.00, 0.62,  1.00),   # body
    (0.00, -0.28, 0.16,  0.60),   # bright insert (lower)
    (0.26,  0.20, 0.12, -0.40),   # cold insert (upper right)
    (-0.28, 0.14, 0.13,  0.50),   # bright insert (upper left)
    (0.00,  0.36, 0.08,  0.80),   # small bright insert (top)
]


def rasterize_truth(N, W):
    """Return the ground-truth image (N*N float32), pixel (px,py) at world
    (-W + px*pix, -W + py*pix). Analytic disc membership -> crisp phantom."""
    pix = (2.0 * W / (N - 1)) if N > 1 else 0.0
    img = np.zeros((N, N), dtype=np.float32)
    for py in range(N):
        wy = -W + py * pix
        for px in range(N):
            wx = -W + px * pix
            v = 0.0
            for (cx, cy, r, d) in PHANTOM:
                if (wx - cx) ** 2 + (wy - cy) ** 2 <= r * r:
                    v += d
            img[py, px] = v
    return img, pix


def forward_project(img, N, W, pix, n_angles, n_det, ds):
    """Voxel-scatter forward model IDENTICAL to forward_project_cpu (float32,
    py-outer/px-inner, linear split into two detector bins)."""
    center = 0.5 * (n_det - 1)
    sino = np.zeros((n_angles, n_det), dtype=np.float32)
    # Precompute trig in double then store float32, exactly like compute_trig.
    cos = [np.float32(math.cos(math.pi * k / n_angles)) for k in range(n_angles)]
    sin = [np.float32(math.sin(math.pi * k / n_angles)) for k in range(n_angles)]
    for py in range(N):
        wy = np.float32(-W + py * pix)
        for px in range(N):
            val = img[py, px]
            if val == 0.0:
                continue
            wx = np.float32(-W + px * pix)
            for k in range(n_angles):
                s = np.float32(wx * cos[k] + wy * sin[k])
                fidx = np.float32(s / ds + center)
                lo = math.floor(fidx)
                if 0 <= lo < n_det - 1:
                    w = np.float32(fidx - lo)
                    sino[k, lo]     += val * (np.float32(1.0) - w)
                    sino[k, lo + 1] += val * w
    return sino


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic noisy CT sinogram (disc phantom).")
    ap.add_argument("--angles", type=int, default=48, help="projection angles over [0,pi)")
    ap.add_argument("--det", type=int, default=67, help="detector bins per projection")
    ap.add_argument("--ds", type=float, default=0.032, help="detector bin spacing (world units)")
    ap.add_argument("--img", type=int, default=48, help="reconstruction image side (pixels)")
    ap.add_argument("--world-half", type=float, default=0.85, help="image spans [-W,W]^2")
    ap.add_argument("--iters", type=int, default=60, help="SIRT iterations the demo runs")
    ap.add_argument("--lambda", dest="lam", type=float, default=1.5, help="SIRT step size")
    ap.add_argument("--tv", type=float, default=0.010, help="TV smoothing weight (0 = pure SIRT)")
    ap.add_argument("--noise", type=float, default=0.02, help="relative Gaussian noise stddev")
    ap.add_argument("--seed", type=int, default=1234, help="RNG seed (fixed -> reproducible file)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    N, W, n_angles, n_det, ds = args.img, args.world_half, args.angles, args.det, args.ds
    truth, pix = rasterize_truth(N, W)
    sino = forward_project(truth, N, W, pix, n_angles, n_det, ds)

    # Add reproducible noise proportional to the local intensity (a stand-in for
    # photon-counting statistics). Fixed seed keeps the committed file stable.
    rng = np.random.default_rng(args.seed)
    scale = float(np.max(sino)) if np.max(sino) > 0 else 1.0
    noise = rng.normal(0.0, args.noise * scale, size=sino.shape).astype(np.float32)
    sino_noisy = (sino + noise).astype(np.float32)

    header = (f"{n_angles} {n_det} {ds} {N} {W} "
              f"{args.iters} {args.lam} {args.tv} 1")
    lines = [header]
    for k in range(n_angles):
        lines.append(" ".join(f"{v:.6f}" for v in sino_noisy[k]))
    for py in range(N):
        lines.append(" ".join(f"{v:.6f}" for v in truth[py]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}")
    print(f"[make_synthetic]   {n_angles} angles x {n_det} det -> {N}x{N} image, "
          f"{args.iters} SIRT iters, tv={args.tv}; SYNTHETIC disc phantom + noise")


if __name__ == "__main__":
    main()
