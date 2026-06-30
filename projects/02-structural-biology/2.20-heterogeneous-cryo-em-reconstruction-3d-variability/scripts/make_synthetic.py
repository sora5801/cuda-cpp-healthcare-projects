#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic cryo-EM 3DVA dataset
# ---------------------------------------------------------------------------
# Project 2.20 : Heterogeneous Cryo-EM Reconstruction (3D Variability)
#
# WHY THIS EXISTS
#   The real EMPIAR datasets (spliceosome, 80S ribosome, TRPV1) are large and
#   require download + preprocessing. So the COMMITTED sample is SYNTHETIC: a
#   tiny, deterministic, clearly-labeled stand-in that makes the demo run offline
#   AND embeds a known answer we can check the algorithm recovers (PATTERNS.md §6).
#
# WHAT IT SIMULATES (the heterogeneity)
#   One "flexible molecule" on a G x G x G grid. Its density is a single Gaussian
#   blob. The molecule's ONE conformational degree of freedom is how far the blob
#   has slid along the z-axis: particle p has a continuous coordinate t[p] in
#   [-1, +1] that shifts the blob center in z. We render N such volumes (one per
#   particle), flatten each G^3 cube to a row, add a little reproducible noise,
#   and also write the ground-truth t[p].
#
#   3DVA / PCA should then discover that the dominant axis of variation IS the
#   z-slide: PC1 captures most of the variance, and the recovered latent
#   coordinate z[p] correlates ~1.0 with t[p]. main.cu prints exactly that.
#
# OUTPUT FORMAT (whitespace-separated text; '#' starts a comment):
#   line 1 : N G D            (N particles, grid edge G, D = G^3 voxels)
#   next N : D floats each     -> one flattened volume per particle
#   last 1 : N floats          -> ground-truth conformational coords t[p]
#
# DETERMINISM: pure-Python, fixed seed via a tiny LCG (no numpy dependency), so
#   the committed sample and demo/expected_output.txt never drift.
#
# USAGE
#   python scripts/make_synthetic.py                 # writes data/sample/volumes.txt
#   python scripts/make_synthetic.py --n 64 --g 8    # a bigger synthetic problem
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "volumes.txt"


class LCG:
    """A tiny deterministic linear-congruential generator (Numerical Recipes
    constants). We avoid numpy so the script has zero dependencies and is byte
    reproducible across machines. Returns floats in [0, 1)."""
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF

    def next(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state / 4294967296.0


def gaussian_blob(g: int, cz: float, sigma: float):
    """Render a G x G x G Gaussian density centered at (cx, cy, cz) and return it
    flattened to a length-G^3 list (row-major: index = (z*G + y)*G + x). The blob
    sits at the grid center in x and y; only its z-center cz varies per particle
    -- that single sliding axis IS the heterogeneity 3DVA must recover."""
    cx = (g - 1) / 2.0
    cy = (g - 1) / 2.0
    vol = []
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    for z in range(g):
        for y in range(g):
            for x in range(g):
                r2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
                vol.append(math.exp(-r2 * inv2s2))
    return vol


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic cryo-EM 3DVA sample.")
    ap.add_argument("--n", type=int, default=24, help="number of particle volumes")
    ap.add_argument("--g", type=int, default=6, help="grid edge length (cube is g^3)")
    ap.add_argument("--sigma", type=float, default=1.2, help="blob width (voxels)")
    ap.add_argument("--noise", type=float, default=0.01, help="per-voxel noise amplitude")
    ap.add_argument("--seed", type=int, default=12345, help="RNG seed (determinism)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    n, g = args.n, args.g
    d = g * g * g
    rng = LCG(args.seed)

    # Ground-truth conformational coordinate t[p] sweeps linearly from -1 to +1.
    # The blob center in z is the grid center plus t * (slide amplitude).
    center_z = (g - 1) / 2.0
    slide = (g - 1) / 4.0                       # how far the blob travels each way
    truth = [(-1.0 + 2.0 * p / (n - 1)) for p in range(n)]

    rows = []
    for p in range(n):
        cz = center_z + truth[p] * slide
        vol = gaussian_blob(g, cz, args.sigma)
        # Add small reproducible noise so the covariance is not exactly rank-1
        # (real data is noisy); PC1 should still dominate.
        vol = [val + args.noise * (rng.next() - 0.5) for val in vol]
        rows.append(vol)

    # Assemble the text file.
    lines = []
    lines.append("# SYNTHETIC cryo-EM 3DVA sample -- NOT real data (see data/README.md).")
    lines.append("# A Gaussian density blob slides along z; t[p] is the hidden coordinate.")
    lines.append(f"{n} {g} {d}    # N G D")
    for p in range(n):
        lines.append(" ".join(f"{v:.6f}" for v in rows[p]) + f"   # volume {p}")
    lines.append(" ".join(f"{t:.6f}" for t in truth) + "   # ground-truth t[p]")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {out}  (N={n}, G={g}, D={d}; SYNTHETIC)")


if __name__ == "__main__":
    main()
