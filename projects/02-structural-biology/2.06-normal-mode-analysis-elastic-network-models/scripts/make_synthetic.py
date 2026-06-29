#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic Cα protein structure
# ---------------------------------------------------------------------------
# Project 2.06 : Normal Mode Analysis / Elastic Network Models
#
# Builds a compact globular Cα backbone: a self-avoiding-ish chain with realistic
# ~3.8 A spacing, gently pulled toward the centre so it folds into a blob (so the
# elastic network is well-connected -> exactly 6 rigid-body modes). Deterministic
# from a seed. Real structures are Cα coordinates from PDB files (see download_data).
#
# OUTPUT (data/README.md format): "N cutoff" then N lines of "x y z".
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --N 120
# ===========================================================================
import argparse
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "protein_ca.txt"


def norm(v):
    m = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) or 1.0
    return (v[0] / m, v[1] / m, v[2] / m)


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic Cα protein structure.")
    ap.add_argument("--N", type=int, default=60, help="number of C-alpha atoms (residues)")
    ap.add_argument("--cutoff", type=float, default=13.0, help="ANM spring cutoff (A)")
    ap.add_argument("--step", type=float, default=3.8, help="Cα-Cα spacing (A)")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pos = [(0.0, 0.0, 0.0)]
    d = norm((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)))
    for _ in range(args.N - 1):
        last = pos[-1]
        # Pull toward the centre, scaled by how far out we are, so the chain
        # stays a tight, well-connected blob (no dangling, under-connected ends).
        r = math.sqrt(last[0]**2 + last[1]**2 + last[2]**2)
        pw = 0.35 + 0.05 * r                                   # stronger pull when far out
        pull = norm((-last[0], -last[1], -last[2]))
        rnd = (rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1))
        nd = norm((d[0] * 0.45 + pull[0] * pw + rnd[0] * 0.55,
                   d[1] * 0.45 + pull[1] * pw + rnd[1] * 0.55,
                   d[2] * 0.45 + pull[2] * pw + rnd[2] * 0.55))
        pos.append((last[0] + nd[0] * args.step,
                    last[1] + nd[1] * args.step,
                    last[2] + nd[2] * args.step))
        d = nd

    lines = [f"{args.N} {args.cutoff:g}"] + [f"{x:.3f} {y:.3f} {z:.3f}" for (x, y, z) in pos]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  (N={args.N} C-alpha atoms, cutoff={args.cutoff} A; "
          f"compact synthetic fold)")


if __name__ == "__main__":
    main()
