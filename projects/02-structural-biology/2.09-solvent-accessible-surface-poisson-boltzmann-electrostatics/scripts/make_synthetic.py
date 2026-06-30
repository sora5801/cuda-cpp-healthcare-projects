#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write the synthetic PBE input (.pqr-style)
# ---------------------------------------------------------------------------
# Project 2.9 : Solvent-Accessible Surface & Poisson-Boltzmann Electrostatics
#
# The "data" for a continuum-electrostatics solve is a set of ATOMS, each with a
# position, a partial CHARGE (e), and a RADIUS (A) -- exactly what a .pqr file
# carries (and what PDB2PQR produces from a PDB structure for APBS). The grids
# (dielectric, screening, charge density) are then BUILT from these atoms by the
# solver (build_problem in reference_cpu.cpp), so the only thing we author here
# is the atom list plus the grid/physics parameters.
#
# We engineer a tiny, INTERPRETABLE synthetic molecule: a compact cluster of
# atoms with a built-in DIPOLE -- a few positive charges on one side and matching
# negative charges on the other. The PBE solution then shows a clear positive
# potential well near the + atoms and a negative well near the - atoms, decaying
# into the screened solvent. Net charge is zero, so the far field is dipolar.
#
# OUTPUT (data/README.md format):
#   line 1:  natoms n h eps_in eps_out kappa2 iters
#   then natoms lines:  x y z q radius
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --n 64 --iters 800   # finer/longer solve
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "molecule.pqr"


def build_atoms():
    """A small synthetic 'molecule' (~11 atoms) with a deliberate dipole.

    Two charged lobes separated along x, plus a few neutral 'scaffold' atoms in
    the middle so the low-dielectric interior is a connected blob (not two
    disconnected spheres). All coordinates in angstrom; charges in e; radii in A.
    Returns a list of (x, y, z, q, radius) tuples.
    """
    atoms = []
    # Positive lobe (x = -4): three +0.33 e charges -> net +1 on this side.
    atoms.append((-4.0,  0.0,  0.0,  0.33, 1.8))
    atoms.append((-4.0,  1.4,  0.0,  0.33, 1.8))
    atoms.append((-4.0, -1.4,  0.0,  0.34, 1.8))
    # Negative lobe (x = +4): three -0.33 e charges -> net -1 on this side.
    atoms.append(( 4.0,  0.0,  0.0, -0.33, 1.8))
    atoms.append(( 4.0,  1.4,  0.0, -0.33, 1.8))
    atoms.append(( 4.0, -1.4,  0.0, -0.34, 1.8))
    # Neutral scaffold bridging the two lobes (no charge; defines the interior).
    for x in (-2.0, 0.0, 2.0):
        atoms.append((x, 0.0, 0.0, 0.0, 1.7))
    # A couple of out-of-plane neutral atoms so the blob is 3-D, not a rod.
    atoms.append((0.0, 0.0,  1.6, 0.0, 1.7))
    atoms.append((0.0, 0.0, -1.6, 0.0, 1.7))
    return atoms


def main():
    ap = argparse.ArgumentParser(description="Write the synthetic PBE atom/grid input.")
    ap.add_argument("--n", type=int, default=48, help="cells per side (n^3 grid)")
    ap.add_argument("--h", type=float, default=0.6, help="grid spacing (angstrom)")
    ap.add_argument("--eps-in", type=float, default=2.0, help="protein dielectric")
    ap.add_argument("--eps-out", type=float, default=80.0, help="water dielectric")
    ap.add_argument("--kappa2", type=float, default=0.10,
                    help="squared inverse Debye length (1/A^2); ionic strength")
    ap.add_argument("--iters", type=int, default=600,
                    help="red-black Gauss-Seidel sweeps")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    atoms = build_atoms()
    header = (f"{len(atoms)} {args.n} {args.h:g} {args.eps_in:g} "
              f"{args.eps_out:g} {args.kappa2:g} {args.iters}")
    lines = [header]
    for (x, y, z, q, r) in atoms:
        lines.append(f"{x:g} {y:g} {z:g} {q:g} {r:g}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"({len(atoms)} atoms, {args.n}^3 grid, h={args.h} A, "
          f"kappa^2={args.kappa2}, {args.iters} sweeps -> dipolar potential)")


if __name__ == "__main__":
    main()
