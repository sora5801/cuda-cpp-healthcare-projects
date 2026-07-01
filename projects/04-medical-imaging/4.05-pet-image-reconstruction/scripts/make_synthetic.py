#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic PET sinogram sample
# ---------------------------------------------------------------------------
# Project 4.5 : PET Image Reconstruction (MLEM / OS-EM)
#
# WHY THIS EXISTS
#   Real PET list-mode/sinogram data (OpenNEURO, TCIA, PETRIC, Siemens mMR
#   phantoms) is large and/or needs registration, so it is NOT committed. To keep
#   the demo runnable OFFLINE we generate a tiny, clearly-SYNTHETIC sinogram from
#   a known digital phantom. Because we know the true image, the reconstruction is
#   INTERPRETABLE: MLEM should recover a bright central disc plus a smaller hot
#   spot, which main.cu reports (docs/PATTERNS.md §6, "embed a known answer").
#
# HOW IT MATCHES THE C++ CODE
#   The forward projection here uses the EXACT same parallel-beam, pixel-driven,
#   linear-split geometry as src/pet_geometry.h + src/reference_cpu.cpp:
#     * angles theta_k = k*pi/K over 180 degrees,
#     * detector bin j at offset s_j = (j-(D-1)/2)*ds,
#     * a pixel at fractional bin fidx splits (1-w)/w into bins floor(fidx),+1.
#   So the committed counts are consistent with the model the solver inverts.
#   Poisson noise is added with a FIXED seed and BAKED INTO the committed file, so
#   the demo's stdout is deterministic no matter what (the program never RNGs).
#
# OUTPUT FORMAT (see data/README.md):
#   header: "K D ds N W iters"
#   then K rows of D floats -> the noisy measured sinogram (counts).
#
# USAGE
#   python scripts/make_synthetic.py                 # default tiny sample
#   python scripts/make_synthetic.py --N 64 --K 60 --D 91 --counts 400 --iters 40
# ===========================================================================
import argparse
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "sinogram_sample.txt"


def make_phantom(N, W):
    """A known emission phantom on an N x N grid over [-W,W]^2.
    Two uniform hot discs (a big central one + a smaller off-center one) on a
    faint warm background -- a caricature of an FDG uptake pattern. Returns a
    row-major list of length N*N (activity per pixel)."""
    pix = (2.0 * W / (N - 1)) if N > 1 else 0.0
    img = [0.0] * (N * N)
    for py in range(N):
        wy = -W + py * pix
        for px in range(N):
            wx = -W + px * pix
            v = 0.02                                    # faint background activity
            if wx * wx + wy * wy <= (0.55 * W) ** 2:    # big central disc
                v = 1.0
            if (wx - 0.45 * W) ** 2 + (wy - 0.35 * W) ** 2 <= (0.18 * W) ** 2:
                v = 1.6                                 # small bright hot spot
            img[py * N + px] = v
    return img


def forward_project(img, N, K, D, ds, W):
    """Pixel-driven forward projection A x -- identical geometry to the C++ code.
    Returns the noise-free sinogram as a row-major list of length K*D."""
    pix = (2.0 * W / (N - 1)) if N > 1 else 0.0
    center = 0.5 * (D - 1)
    cosv = [math.cos(math.pi * k / K) for k in range(K)]
    sinv = [math.sin(math.pi * k / K) for k in range(K)]
    sino = [0.0] * (K * D)
    for py in range(N):
        wy = -W + py * pix
        for px in range(N):
            xv = img[py * N + px]
            if xv == 0.0:
                continue
            wx = -W + px * pix
            for k in range(K):
                s = wx * cosv[k] + wy * sinv[k]
                fidx = s / ds + center
                j0 = math.floor(fidx)
                w = fidx - j0
                if 0 <= j0 and j0 + 1 < D:
                    sino[k * D + j0] += xv * (1.0 - w)
                    sino[k * D + j0 + 1] += xv * w
    return sino


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic PET sinogram.")
    ap.add_argument("--N", type=int, default=32, help="image side length (pixels)")
    ap.add_argument("--K", type=int, default=30, help="number of projection angles")
    ap.add_argument("--D", type=int, default=45, help="detector bins per angle")
    ap.add_argument("--W", type=float, default=1.0, help="image half-width (world units)")
    ap.add_argument("--counts", type=float, default=300.0,
                    help="peak expected counts (scales Poisson noise level)")
    ap.add_argument("--iters", type=int, default=30, help="advisory MLEM iterations")
    ap.add_argument("--seed", type=int, default=12345, help="RNG seed (noise is baked in)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    N, K, D, W = args.N, args.K, args.D, args.W
    # Detector spacing: cover the phantom diagonal (+margin) with D bins.
    ds = (2.0 * math.sqrt(2.0) * W) / (D - 1)

    img = make_phantom(N, W)
    sino = forward_project(img, N, K, D, ds, W)

    # Scale to a realistic count level, then add Poisson noise (fixed seed).
    peak = max(sino) if sino else 1.0
    scale = (args.counts / peak) if peak > 0 else 1.0
    rng = random.Random(args.seed)

    def poisson(lam):
        # Knuth's algorithm -- fine for the small lambdas in this tiny sample.
        L = math.exp(-lam)
        k = 0
        prod = 1.0
        while True:
            k += 1
            prod *= rng.random()
            if prod <= L:
                return k - 1

    counts = [float(poisson(v * scale)) for v in sino]

    lines = [f"{K} {D} {ds:.8f} {N} {W:.8f} {args.iters}"]
    for k in range(K):
        row = counts[k * D:(k + 1) * D]
        lines.append(" ".join(f"{c:g}" for c in row))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    total = sum(counts)
    print(f"[make_synthetic] wrote {args.out}")
    print(f"[make_synthetic]   SYNTHETIC PET sinogram: K={K} D={D} N={N} ds={ds:.4f} "
          f"W={W} iters={args.iters}")
    print(f"[make_synthetic]   peak_scale={scale:.3f}  total_counts={total:.0f}  seed={args.seed}")


if __name__ == "__main__":
    main()
