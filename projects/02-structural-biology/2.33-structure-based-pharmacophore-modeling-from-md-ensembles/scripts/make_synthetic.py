#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic pharmacophore screen
# ---------------------------------------------------------------------------
# Project 2.33 : Structure-Based Pharmacophore Modeling from MD Ensembles
#
# WHAT IT BUILDS  (all SYNTHETIC -- no real molecules, see data/README.md)
#   * ONE query pharmacophore: a handful of typed 3-D feature points standing in
#     for the consensus features an MD ensemble would yield (donor / acceptor /
#     hydrophobe / aromatic / charges).
#   * N library "molecules", each a small set of typed feature points:
#       - the PLANTED TARGET (index `--target`) is a near-copy of the query --
#         same feature types, positions jittered by a sub-angstrom amount and a
#         couple of extra decoy features -- so it should score near 1.0 and rank
#         #1. This embeds a KNOWN ANSWER the demo recovers (PATTERNS.md §6).
#       - every other molecule is RANDOM: random feature types at random
#         positions, so its ROCS color-Tanimoto to the query is low.
#   Feature counts VARY per molecule, which is the whole point of the flat CSR
#   layout the loader/kernel use.
#
# OUTPUT FORMAT  (data/README.md):
#   header:           "N n_query target"
#   query block:      n_query rows of "type x y z weight"
#   each molecule k:  a line "m" then m rows of "type x y z weight"
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --N 100000 --target 12
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "pharmacophore_sample.txt"

NUM_TYPES = 6   # must match FeatureType / FEAT_NUM_TYPES in src/pharmacophore.h


def rand_feature(rng, box):
    """A random typed feature point inside a cubic box of side `box` angstroms."""
    t = rng.randint(0, NUM_TYPES - 1)
    x = rng.uniform(0.0, box)
    y = rng.uniform(0.0, box)
    z = rng.uniform(0.0, box)
    w = round(rng.uniform(0.6, 1.0), 3)   # consensus-feature weight in [0.6,1]
    return (t, x, y, z, w)


def fmt(feat):
    t, x, y, z, w = feat
    return f"{t} {x:.3f} {y:.3f} {z:.3f} {w:.3f}"


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic pharmacophore screen.")
    ap.add_argument("--N", type=int, default=512, help="number of library molecules")
    ap.add_argument("--nquery", type=int, default=5, help="features in the query pharmacophore")
    ap.add_argument("--box", type=float, default=12.0, help="cubic box side [angstrom]")
    ap.add_argument("--jitter", type=float, default=0.25,
                    help="sub-angstrom position jitter applied to the planted target")
    ap.add_argument("--target", type=int, default=7, help="library index of the planted match")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    N = args.N

    # ---- the query pharmacophore (the "consensus" of a notional MD ensemble) ----
    query = [rand_feature(rng, args.box) for _ in range(args.nquery)]

    # ---- the library --------------------------------------------------------
    library = []
    for k in range(N):
        if k == args.target:
            # Planted hit: jitter each query feature slightly, keep its type, then
            # add 1-2 random decoy features so the count differs from the query.
            mol = []
            for (t, x, y, z, w) in query:
                jx = x + rng.uniform(-args.jitter, args.jitter)
                jy = y + rng.uniform(-args.jitter, args.jitter)
                jz = z + rng.uniform(-args.jitter, args.jitter)
                mol.append((t, jx, jy, jz, w))
            for _ in range(rng.randint(1, 2)):
                mol.append(rand_feature(rng, args.box))
        else:
            # Decoy: a random number (4-9) of random typed features.
            m = rng.randint(4, 9)
            mol = [rand_feature(rng, args.box) for _ in range(m)]
        library.append(mol)

    # ---- emit ----------------------------------------------------------------
    lines = [f"{N} {args.nquery} {args.target}"]
    lines += [fmt(f) for f in query]
    for mol in library:
        lines.append(str(len(mol)))
        lines += [fmt(f) for f in mol]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  (N={N} molecules, n_query={args.nquery}, "
          f"planted target {args.target}; SYNTHETIC)")


if __name__ == "__main__":
    main()
