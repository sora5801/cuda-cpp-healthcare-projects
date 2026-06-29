#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the tiny SYNTHETIC structure sample
# ---------------------------------------------------------------------------
# Project 1.9 : ML Interatomic Potentials (Neural Network Potentials)
#
# WHAT THIS MAKES
#   A small, fully SYNTHETIC molecular cluster (data/sample/water_cluster.xyzc)
#   that the offline demo runs on. It is a grid of "molecule" units, each a
#   water-like triangle of three atoms, jittered by a FIXED, seeded rule so the
#   coordinates are reproducible (every run -- in Python or by hand -- yields the
#   same file). Coordinates are in Angstrom.
#
#   IMPORTANT: this data is SYNTHETIC and carries no chemical meaning. It exists
#   only to exercise the descriptor + MLP pipeline on a realistic-looking spread
#   of interatomic distances. It is NOT a real molecule and NOT for any chemical
#   or clinical use (CLAUDE.md sec 8). The energies the demo prints are equally
#   synthetic (the MLP weights are manufactured, not trained).
#
#   The matching real datasets (ANI-1ccx, SPICE, rMD17) are pointed to by
#   scripts/download_data.{ps1,sh} and data/README.md.
#
# FILE FORMAT (see data/README.md):
#   line 1 : "<n>"                       number of atoms
#   next n : "<x> <y> <z>"               coordinates in Angstrom
#   '#' starts a comment line (ignored by the loader).
#
# USAGE
#   python scripts/make_synthetic.py                       # default 8 molecules
#   python scripts/make_synthetic.py --mols 27 --out foo.xyzc
# ===========================================================================
import argparse
import math
import os


def splitmix64(state):
    """The SAME tiny deterministic PRNG used in src/reference_cpu.cpp.
    Integer-only so it produces identical bits everywhere. Returns
    (next_value, new_state)."""
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return z, state


def next_uniform(state):
    """Next double in [0, 1), mirroring next_weight()'s [0,1) step in C++."""
    val, state = splitmix64(state)
    bits = val >> 11                       # top 53 bits
    return bits / 9007199254740992.0, state  # / 2^53


def build_cluster(n_mols):
    """Place n_mols water-like triangles on a coarse 3-D lattice and jitter each
    a little. Returns a flat list of (x, y, z) tuples (3 atoms per molecule).

    A 'water' triangle: one O at the unit center, two H at the canonical
    O-H bond length (0.9572 A) and H-O-H angle (104.52 deg). This is only to get
    a chemically plausible SPREAD of short and long distances -- the atoms are
    not typed, so the model treats them all identically."""
    OH = 0.9572                            # O-H bond length (Angstrom)
    half_angle = math.radians(104.52 / 2)  # half the H-O-H angle
    # Two H positions relative to the O, in the molecule's local frame.
    h1 = (OH * math.sin(half_angle),  OH * math.cos(half_angle), 0.0)
    h2 = (-OH * math.sin(half_angle), OH * math.cos(half_angle), 0.0)

    # Lay molecules out on the smallest cube that holds n_mols, ~3.1 A apart so
    # neighboring molecules fall inside the 5 A cutoff (giving inter-molecular
    # neighbors, not just the 3 intramolecular atoms).
    side = int(math.ceil(n_mols ** (1.0 / 3.0)))
    spacing = 3.1                          # lattice spacing (Angstrom)

    state = 0xC0FFEE123456789              # fixed seed -> reproducible jitter
    atoms = []
    placed = 0
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                if placed >= n_mols:
                    break
                # Lattice center for this molecule.
                cx, cy, cz = ix * spacing, iy * spacing, iz * spacing
                # Small jitter in [-0.15, 0.15) A on each axis (deterministic).
                js = []
                for _ in range(3):
                    u, state = next_uniform(state)
                    js.append((u - 0.5) * 0.3)
                cx += js[0]; cy += js[1]; cz += js[2]
                # O atom at the (jittered) center, then the two H atoms.
                atoms.append((cx, cy, cz))
                atoms.append((cx + h1[0], cy + h1[1], cz + h1[2]))
                atoms.append((cx + h2[0], cy + h2[1], cz + h2[2]))
                placed += 1
    return atoms


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic atomic cluster.")
    ap.add_argument("--mols", type=int, default=8,
                    help="number of water-like molecules (3 atoms each)")
    ap.add_argument("--out", default=None,
                    help="output path (default: data/sample/water_cluster.xyzc)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    out = args.out or os.path.join(root, "data", "sample", "water_cluster.xyzc")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    atoms = build_cluster(args.mols)
    n = len(atoms)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# SYNTHETIC atomic cluster -- generated by scripts/make_synthetic.py\n")
        f.write("# {} water-like molecules ({} atoms). Coordinates in Angstrom.\n".format(
            args.mols, n))
        f.write("# NOT a real molecule and NOT for chemical/clinical use (see data/README.md).\n")
        f.write("{}\n".format(n))
        for (x, y, z) in atoms:
            f.write("{:.6f} {:.6f} {:.6f}\n".format(x, y, z))
    print("wrote {} atoms ({} molecules) to {}".format(n, args.mols, out))


if __name__ == "__main__":
    main()
