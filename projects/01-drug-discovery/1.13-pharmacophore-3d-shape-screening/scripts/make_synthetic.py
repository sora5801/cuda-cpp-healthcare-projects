#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic conformer sample
# ---------------------------------------------------------------------------
# Project 1.13 : Pharmacophore & 3D Shape Screening
#
# WHY THIS EXISTS
#   The real conformer libraries this project targets (ZINC20, DUD-E, Enamine
#   REAL) are large and/or licensed, so we cannot commit them. To keep the demo
#   runnable OFFLINE we generate a tiny, clearly-SYNTHETIC stand-in that matches
#   the loader's format (see data/README.md). Synthetic data is labeled
#   synthetic everywhere (CLAUDE.md sec 8) -- these are NOT real molecules.
#
# WHAT WE BUILD (so the result is INTERPRETABLE -- PATTERNS.md sec 6)
#   * a QUERY shape: a small rigid cluster of 6 carbon-sized spheres arranged
#     as a hexagon (a stand-in for an aromatic ring) plus one substituent atom.
#   * library conformers engineered to span the similarity range, so the ranking
#     recovers a KNOWN answer:
#       - lib_00_self     : an exact copy of the query     -> ShapeTanimoto = 1.0
#       - lib_01_jitter   : the query with sub-angstrom noise -> very high (~0.95)
#       - lib_02_shift05  : the query shifted 0.5 A          -> high
#       - lib_03_shift15  : the query shifted 1.5 A          -> moderate
#       - lib_04_rot      : the query rotated about z by 30deg-> high (shape is
#                           rotation-invariant for THIS symmetric query)
#       - lib_05_grow     : query + 2 extra atoms (bigger)   -> moderate (size gap)
#       - lib_06_shrink   : query minus 2 atoms (smaller)    -> moderate
#       - lib_07_far      : the query translated 8 A away    -> ~0 (disjoint)
#       - lib_08_line     : a totally different (linear) shape -> low
#   The demo's top hit must be lib_00_self at exactly 1.000000 -- a built-in
#   correctness check anyone can eyeball.
#
#   Coordinates are written with FIXED rounding so the committed file -- and thus
#   the program's deterministic output -- never changes between regenerations.
#
# USAGE
#   python scripts/make_synthetic.py
#       -> writes data/sample/conformers_sample.txt
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "conformers_sample.txt"

# Van der Waals radius (angstrom) used for every atom. Carbon's vdW radius is
# ~1.70 A; we use one element so the shape (not the chemistry) is the variable.
R_C = 1.70
R_BIG = 1.90   # a slightly larger sphere for the "grow" variant's extras


def hexagon(cx=0.0, cy=0.0, cz=0.0, radius=1.39):
    """Six points on a regular hexagon in the z=cz plane (a benzene-like ring).
    1.39 A is the C-C aromatic bond length, so the ring is realistically sized."""
    pts = []
    for i in range(6):
        ang = math.pi / 3.0 * i                        # 0, 60, 120, ... degrees
        pts.append((cx + radius * math.cos(ang),
                    cy + radius * math.sin(ang),
                    cz))
    return pts


def query_atoms():
    """The reference shape: a hexagon ring + one substituent atom off the ring.
    Returns a list of (x, y, z) tuples (radius R_C is added by the writer)."""
    pts = hexagon()
    pts.append((2.6, 0.0, 0.0))                        # a substituent on one vertex
    return pts


def rotate_z(pts, deg):
    """Rotate points about the z-axis by `deg` degrees (rigid-body rotation)."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return [(c * x - s * y, s * x + c * y, z) for (x, y, z) in pts]


def shift(pts, dx, dy, dz):
    """Translate every point by (dx, dy, dz)."""
    return [(x + dx, y + dy, z + dz) for (x, y, z) in pts]


def jitter(pts, amp=0.15):
    """Add small DETERMINISTIC per-coordinate noise (no RNG, so the file is
    reproducible). Uses a fixed sinusoid pattern keyed to the atom index."""
    out = []
    for i, (x, y, z) in enumerate(pts):
        out.append((x + amp * math.sin(1.7 * i),
                    y + amp * math.cos(2.3 * i),
                    z + amp * math.sin(0.9 * i)))
    return out


def build_library():
    """Return a list of (label, [(x,y,z,radius), ...]) library conformers,
    engineered so the ranking recovers a known answer (see file header)."""
    q = query_atoms()
    lib = []

    def with_r(pts, r=R_C):
        return [(x, y, z, r) for (x, y, z) in pts]

    lib.append(("lib_00_self",   with_r(q)))                       # exact copy -> 1.0
    lib.append(("lib_01_jitter", with_r(jitter(q))))               # tiny noise -> ~0.95
    lib.append(("lib_02_shift05", with_r(shift(q, 0.5, 0.0, 0.0))))# +0.5 A     -> high
    lib.append(("lib_03_shift15", with_r(shift(q, 1.5, 0.0, 0.0))))# +1.5 A     -> moderate
    lib.append(("lib_04_rot",    with_r(rotate_z(q, 30.0))))       # rotated    -> high
    grow = q + [(0.0, 0.0, 1.6), (0.0, 0.0, -1.6)]                 # two atoms above/below
    lib.append(("lib_05_grow",   with_r(grow[:-2]) +
                                 [(0.0, 0.0, 1.6, R_BIG), (0.0, 0.0, -1.6, R_BIG)]))
    lib.append(("lib_06_shrink", with_r(q[:5])))                   # drop 2 atoms -> moderate
    lib.append(("lib_07_far",    with_r(shift(q, 8.0, 0.0, 0.0)))) # 8 A away   -> ~0
    line = [(float(i) * 1.5, 0.0, 0.0) for i in range(7)]          # a linear chain
    lib.append(("lib_08_line",   with_r(line)))                    # different shape -> low
    return lib


def fmt(v):
    """Fixed 4-decimal formatting so the file is byte-stable across machines."""
    return f"{v:.4f}"


def write_block(lines, label, atoms):
    lines.append(f"{len(atoms)}")
    lines.append(label)
    for (x, y, z, r) in atoms:
        lines.append(f"{fmt(x)} {fmt(y)} {fmt(z)} {fmt(r)}")


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic conformer sample.")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    q = [(x, y, z, R_C) for (x, y, z) in query_atoms()]
    lib = build_library()

    lines = []
    lines.append("# SYNTHETIC conformer set for Project 1.13 (NOT real molecules).")
    lines.append("# Format: N, then a QUERY block, then N library blocks.")
    lines.append("# Each block: M, label, then M lines of 'x y z radius' (angstrom).")
    lines.append(f"{len(lib)}")
    write_block(lines, "QUERY", q)
    for (label, atoms) in lib:
        write_block(lines, label, atoms)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(query={len(q)} atoms, {len(lib)} conformers; SYNTHETIC)")


if __name__ == "__main__":
    main()
