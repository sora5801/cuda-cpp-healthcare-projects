#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic molecule for SASA
# ---------------------------------------------------------------------------
# Project 1.31 : Solvent-Accessible Surface Area (SASA) on GPU
#
# WHY SYNTHETIC
#   Real input would be a PDB structure (see scripts/download_data.* and
#   data/README.md). To keep the demo offline, tiny, and INTERPRETABLE we build
#   a clearly-SYNTHETIC molecule with a KNOWN answer baked into its geometry
#   (PATTERNS.md sec 6 -- engineer the sample so the result is meaningful):
#
#       * one CENTRAL carbon at the origin, surrounded by an inner shell of 12
#         carbons at bonding distance. The inner shell collectively covers the
#         central atom, so it should have very FEW exposed test points (~buried).
#       * an OUTER shell of 12 carbons pushed far out -> highly EXPOSED, so they
#         top the "most exposed" ranking the program prints.
#       * two lone, well-separated atoms (O, N) -> FULLY exposed (every point).
#
#   "Central atom buried, lone atoms fully exposed" is a fact you can check by
#   eye, so the demo verifies the SCIENCE, not just CPU==GPU agreement.
#
#   Everything is deterministic (no RNG), so data/sample/ and the captured
#   demo/expected_output.txt are byte-for-byte reproducible.
#
# OUTPUT FORMAT (data/README.md):
#   line 1 : "<n>"                      (atom count)
#   next n : "<element> <x> <y> <z>"    (element letter + coords in Angstrom)
#   '#' lines are comments; blank lines are ignored.
#
# USAGE
#   python scripts/make_synthetic.py                 # writes the committed sample
#   python scripts/make_synthetic.py --out other.xyz
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "molecule_sample.xyz"


def icosahedron_vertices():
    """The 12 vertices of a regular icosahedron, normalized to the unit sphere.
    Built from the golden ratio phi; these directions are evenly spread on the
    sphere, so a shell placed on them surrounds the center symmetrically."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    raw = [
        (-1,  phi, 0), (1,  phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1,  phi), (0, 1,  phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
    ]
    norm = math.sqrt(1.0 + phi * phi)
    return [(x / norm, y / norm, z / norm) for (x, y, z) in raw]


def build_molecule():
    """Return a list of (element, x, y, z) atoms with a known exposure pattern."""
    atoms = []
    verts = icosahedron_vertices()

    # (1) Central carbon at the origin -- to be buried by the inner shell.
    atoms.append(("C", 0.0, 0.0, 0.0))

    # (2) Inner shell: 12 carbons at ~3.0 A. Two carbon vdW spheres inflated by
    #     the 1.4 A probe have radius 1.7+1.4 = 3.1 A, so neighbours at 3.0 A
    #     overlap heavily -> they collectively cover the central atom's surface.
    inner_r = 3.0
    for (vx, vy, vz) in verts:
        atoms.append(("C", vx * inner_r, vy * inner_r, vz * inner_r))

    # (3) Outer shell: 12 carbons far out at 12.0 A along the same directions.
    #     They are well clear of everything, so they are highly exposed.
    outer_r = 12.0
    for (vx, vy, vz) in verts:
        atoms.append(("C", vx * outer_r, vy * outer_r, vz * outer_r))

    # (4) Two lone atoms far from all others -> fully exposed (every test point).
    atoms.append(("O", 30.0, 0.0, 0.0))
    atoms.append(("N", 0.0, 30.0, 0.0))

    return atoms


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic molecule for SASA.")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    atoms = build_molecule()
    lines = [
        "# SYNTHETIC molecule for SASA demo (project 1.31) -- NOT a real structure.",
        "# Geometry is engineered so the central atom (index 0) is buried and the",
        "# outer-shell + lone atoms are highly exposed. Format: <element> x y z (Angstrom).",
        f"{len(atoms)}",
    ]
    for (el, x, y, z) in atoms:
        lines.append(f"{el} {x:.6f} {y:.6f} {z:.6f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  (n={len(atoms)} atoms; SYNTHETIC, deterministic)")


if __name__ == "__main__":
    main()
