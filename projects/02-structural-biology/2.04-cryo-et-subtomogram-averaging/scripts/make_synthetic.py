#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic subtomogram sample
# ---------------------------------------------------------------------------
# Project 2.4 : Cryo-ET Subtomogram Averaging  (reduced-scope teaching version)
#
# WHY THIS EXISTS
#   Real cryo-ET subtomograms (EMPIAR/EMDB) are large and licensed; we cannot
#   redistribute them, and they need heavy preprocessing. So we generate a
#   clearly-SYNTHETIC stand-in that exercises the exact same alignment+averaging
#   pipeline, with a KNOWN answer baked in so the demo result is interpretable
#   (PATTERNS.md §6). Synthetic data is labeled synthetic everywhere it appears.
#
# WHAT WE GENERATE (file layout the loader expects -- see data/README.md):
#     header:  n_sub d n_angles
#     ref  cube:  d*d*d floats   (the clean "average" / template)
#     cand cubes: n_sub * d*d*d  (each = ref ROTATED by a planted angle + noise)
#
#   The motif is a handful of 3-D Gaussian blobs arranged ASYMMETRICALLY in the
#   x-y plane, so an in-plane rotation about z genuinely changes the volume and
#   the correct angle is recoverable. Each candidate is the motif rotated by one
#   of the trial angles (planted, so we KNOW the right answer) plus mild additive
#   noise -- mimicking how every real subtomogram is the same particle at a
#   different pose, buried in noise. Everything is deterministic (fixed LCG seed),
#   so the committed sample -- and the demo's stdout -- never change.
#
# USAGE
#   python scripts/make_synthetic.py                 # writes the committed sample
#   python scripts/make_synthetic.py --d 24 --n 12   # a bigger problem
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "subtomograms_sample.txt"


# --- A tiny deterministic LCG so we need no numpy and get identical output ----
# (glibc's constants.) Returns floats in [0,1); two draws make a Gaussian via
# the Box-Muller transform for the additive noise.
class LCG:
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF

    def next_u01(self) -> float:
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return self.state / 0x7FFFFFFF

    def gauss(self, sigma: float) -> float:
        u1 = max(self.next_u01(), 1e-12)   # avoid log(0)
        u2 = self.next_u01()
        return sigma * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def motif(d: int):
    """Build the clean motif: 3 Gaussian blobs placed asymmetrically in x-y.
    Returns a flat list of d*d*d floats (z-major: idx = (z*d + y)*d + x)."""
    c = 0.5 * (d - 1)                       # cube center
    # (cx, cy, cz, amplitude, sigma) for each blob -- deliberately off-center and
    # not rotationally symmetric, so the right rotation is unambiguous.
    blobs = [
        (c + 0.30 * d, c,            c, 1.0, 0.16 * d),
        (c - 0.18 * d, c + 0.22 * d, c, 0.7, 0.13 * d),
        (c,            c - 0.26 * d, c, 0.5, 0.11 * d),
    ]
    vol = [0.0] * (d * d * d)
    for z in range(d):
        for y in range(d):
            for x in range(d):
                v = 0.0
                for (bx, by, bz, amp, sig) in blobs:
                    r2 = (x - bx) ** 2 + (y - by) ** 2 + (z - bz) ** 2
                    v += amp * math.exp(-r2 / (2.0 * sig * sig))
                vol[(z * d + y) * d + x] = v
    return vol


def rotate_z(vol, d: int, theta: float):
    """Rotate a cube in-plane about z by theta (radians), bilinear, OOB->0.
    Mirrors rotate_cube_cpu()/rotate_kernel() so planted poses are exact."""
    cs, sn = math.cos(theta), math.sin(theta)
    center = 0.5 * (d - 1)
    out = [0.0] * (d * d * d)
    for z in range(d):
        for y in range(d):
            for x in range(d):
                ox, oy = x - center, y - center
                sx = center + (cs * ox + sn * oy)      # inverse map (backward)
                sy = center + (-sn * ox + cs * oy)
                x0, y0 = math.floor(sx), math.floor(sy)
                fx, fy = sx - x0, sy - y0
                val = 0.0
                for dy in (0, 1):
                    for dx in (0, 1):
                        xx, yy = x0 + dx, y0 + dy
                        if 0 <= xx < d and 0 <= yy < d:
                            wx = fx if dx else (1.0 - fx)
                            wy = fy if dy else (1.0 - fy)
                            val += wx * wy * vol[(z * d + yy) * d + xx]
                out[(z * d + y) * d + x] = val
    return out


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic subtomogram sample.")
    ap.add_argument("--n", type=int, default=6, help="number of candidate subtomograms")
    ap.add_argument("--d", type=int, default=16, help="cube edge length (voxels)")
    ap.add_argument("--angles", type=int, default=12, help="number of trial rotation angles")
    ap.add_argument("--noise", type=float, default=0.04, help="Gaussian noise sigma")
    ap.add_argument("--seed", type=int, default=12345, help="LCG seed (determinism)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    d, n, na = args.d, args.n, args.angles
    rng = LCG(args.seed)
    base = motif(d)

    # Each candidate should be RECOVERED at a distinct, known trial index, spread
    # around the wheel. To make the alignment search recover index `recover[s]`,
    # we rotate the motif by the NEGATIVE of that angle: re-aligning then needs
    # the +angle that cancels it (rotate_z uses a backward map, so the candidate
    # rotated by -theta is best matched by rotating it back by +theta == recover).
    # This way the demo cleanly reports "recovered angle == planted answer".
    recover = [(s * 2 + 1) % na for s in range(n)]   # spread across the wheel

    cubes = [base]   # the reference is the clean motif (the ideal "average")
    for s in range(n):
        theta = -2.0 * math.pi * recover[s] / na     # rotate by the inverse angle
        rot = rotate_z(base, d, theta)
        noisy = [v + rng.gauss(args.noise) for v in rot]
        cubes.append(noisy)

    # Serialize: header then each cube as one whitespace line of d*d*d floats.
    lines = [f"{n} {d} {na}"]
    for cube in cubes:
        lines.append(" ".join(f"{v:.6f}" for v in cube))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[make_synthetic] wrote {args.out}")
    print(f"[make_synthetic] SYNTHETIC: {n} candidates, {d}^3 voxels, {na} trial angles, "
          f"noise sigma={args.noise}")
    print(f"[make_synthetic] recoverable angle indices (ground truth) = {recover}")


if __name__ == "__main__":
    main()
