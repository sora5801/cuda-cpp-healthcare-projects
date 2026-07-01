#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the committed SYNTHETIC point clouds
# ---------------------------------------------------------------------------
# Project 4.17 : Real-Time Intraoperative / Image-Guided Surgery
#
# WHAT THIS MAKES
#   A tiny, fully SYNTHETIC pair of 3-D point clouds for the ICP demo:
#     * Q  -- the FIXED  "intra-operative" surface: points sampled on a curved
#             organ-like patch (a bump on a tilted plane), in millimetres.
#     * P  -- the MOVING "pre-operative" surface: a rigid-transformed copy of Q
#             by a KNOWN small rotation + translation (the misalignment ICP
#             undoes), plus a whisper of Gaussian noise so it is not a perfect
#             duplicate (real trackers are ~0.1-0.5 mm accurate).
#   The file also records the ground-truth transform (GT) used to build P so a
#   reader can see exactly how the sample was made. ICP recovers GT^{-1} (the
#   transform mapping P back onto Q).
#
#   THIS IS SYNTHETIC DATA -- it models no real patient and must never be used
#   for anything clinical (CLAUDE.md section 8). It is engineered so the demo's
#   result is meaningful: ICP should drive the RMS error toward the noise floor.
#
# OUTPUT FORMAT (see data/README.md and src/reference_cpu.cpp::load_clouds):
#   np nq
#   GT  r00 r01 r02 r10 r11 r12 r20 r21 r22  tx ty tz
#   np lines: "x y z"   (moving cloud P)
#   nq lines: "x y z"   (fixed  cloud Q)
#
# USAGE
#   python make_synthetic.py [--out PATH] [--seed N] [--noise MM] [--grid G]
# Determinism: a fixed seed (default 7) makes the committed sample reproducible.
# ===========================================================================
import argparse
import math
import os
import random


def build(seed: int, noise_mm: float, grid: int):
    rng = random.Random(seed)

    # --- Fixed cloud Q: a curved patch (Gaussian bump on a tilted plane). ---
    # A NON-planar surface makes the rotation observable in all three axes; a
    # flat plane would leave rotation about its normal poorly constrained -- a
    # real ICP pitfall worth knowing (see THEORY "numerical considerations").
    Q = []
    for iy in range(grid):
        for ix in range(grid):
            x = ix * 8.0 - 4.0 * (grid - 1)      # mm, roughly centred on origin
            y = iy * 8.0 - 4.0 * (grid - 1)      # mm
            z = 0.04 * x - 0.02 * y + 6.0 * math.exp(-((x * x + y * y) / 400.0))
            Q.append((x, y, z))

    # --- Ground-truth misalignment: a genuine 3-D rotation + a translation. ---
    def rot_x(a):
        c, s = math.cos(a), math.sin(a)
        return [[1, 0, 0], [0, c, -s], [0, s, c]]

    def rot_y(a):
        c, s = math.cos(a), math.sin(a)
        return [[c, 0, s], [0, 1, 0], [-s, 0, c]]

    def rot_z(a):
        c, s = math.cos(a), math.sin(a)
        return [[c, -s, 0], [s, c, 0], [0, 0, 1]]

    def matmul(A, B):
        return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    R = matmul(matmul(rot_z(math.radians(12.0)), rot_y(math.radians(-7.0))),
               rot_x(math.radians(5.0)))
    t = [6.0, -4.0, 3.0]                          # mm translation

    # --- Moving cloud P = R*Q + t + small Gaussian noise. ---
    P = []
    for (x, y, z) in Q:
        px = R[0][0] * x + R[0][1] * y + R[0][2] * z + t[0]
        py = R[1][0] * x + R[1][1] * y + R[1][2] * z + t[1]
        pz = R[2][0] * x + R[2][1] * y + R[2][2] * z + t[2]
        px += rng.gauss(0.0, noise_mm)
        py += rng.gauss(0.0, noise_mm)
        pz += rng.gauss(0.0, noise_mm)
        P.append((px, py, pz))

    return P, Q, R, t


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic ICP point clouds.")
    default_out = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "surface_pair.txt")
    ap.add_argument("--out", default=default_out, help="output path")
    ap.add_argument("--seed", type=int, default=7, help="RNG seed (determinism)")
    ap.add_argument("--noise", type=float, default=0.15, help="Gaussian noise std (mm)")
    ap.add_argument("--grid", type=int, default=6, help="grid side (grid*grid points)")
    args = ap.parse_args()

    P, Q, R, t = build(args.seed, args.noise, args.grid)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="ascii", newline="\n") as f:
        f.write(f"{len(P)} {len(Q)}\n")
        f.write("GT " + " ".join(f"{R[i][j]:.9f}" for i in range(3) for j in range(3))
                + " " + " ".join(f"{v:.9f}" for v in t) + "\n")
        for (x, y, z) in P:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        for (x, y, z) in Q:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    print(f"[make_synthetic] wrote {out}: {len(P)} moving + {len(Q)} fixed points (SYNTHETIC)")


if __name__ == "__main__":
    main()
