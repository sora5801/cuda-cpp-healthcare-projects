#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic docking problem
# ---------------------------------------------------------------------------
# Project 1.3 : Molecular Docking Engine  (reduced-scope teaching version)
#
# WHY SYNTHETIC  (CLAUDE.md S8: committed data must be tiny + clearly synthetic)
#   Real docking inputs are a prepared receptor (PDBQT) turned into AutoGrid
#   energy maps plus a ligand (see scripts/download_data.* and data/README.md).
#   To keep the demo offline, reproducible, and INTERPRETABLE we synthesise a
#   problem with a KNOWN ANSWER, so the best pose the engine finds is verifiable
#   by eye (PATTERNS.md S6 "engineer the sample so the result is meaningful").
#
#   Construction:
#     * Energy grid: a smooth 3D Gaussian WELL (a single attractive pocket)
#       centred at the world point `well` with depth `depth` kcal/mol. Energy at
#       a grid point r is  -depth * exp(-|r-well|^2 / (2*sigma^2)).  The deepest
#       (most negative) energy is exactly at `well`.
#     * Ligand: a small rigid cluster of atoms whose CENTROID is the origin, so
#       placing the centroid at the well minimises the summed energy. We make the
#       cluster slightly anisotropic so ROTATION matters a little (a non-trivial
#       best rotation), but the dominant signal is "put the centroid in the well".
#     * Search space: a translation grid centred on the (approximate) pocket
#       centre plus a coarse rotation grid. The best pose should land the ligand
#       centroid at (or nearest grid point to) `well`.
#
#   Everything is deterministic (no RNG) so demo/expected_output.txt is stable.
#
# OUTPUT FORMAT  (full spec in data/README.md):
#     GRID  nx ny nz  ox oy oz  spacing
#     <nx*ny*nz energies, x fastest then y then z>
#     LIGAND  n_atoms
#     <n_atoms lines:  x y z weight>
#     SEARCH  n_trans n_rot trans_range  tcx tcy tcz
#
# USAGE
#   python scripts/make_synthetic.py                 # default tiny sample
#   python scripts/make_synthetic.py --n-grid 32 --n-trans 9 --n-rot 4
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "receptor_ligand_sample.txt"


def build(n_grid, spacing, depth, sigma, well, ligand, n_trans, n_rot,
          trans_range, pocket_center):
    """Return the dataset text. Pure function of its args -> deterministic."""
    nx = ny = nz = n_grid
    # Origin so the grid is centred on the world origin: point (0,0,0) sits at
    # -extent/2. extent = (n-1)*spacing is the physical width of the grid.
    half = (n_grid - 1) * spacing / 2.0
    ox = oy = oz = -half

    lines = []
    lines.append("# SYNTHETIC docking problem (project 1.3) -- NOT real receptor/ligand data.")
    lines.append(f"# Gaussian energy well at {well} kcal/mol depth {depth}, sigma {sigma} A.")
    lines.append("# Best pose should place the ligand centroid nearest the well.")
    lines.append(f"GRID {nx} {ny} {nz}  {ox:.6f} {oy:.6f} {oz:.6f}  {spacing:.6f}")

    # Energy values, x fastest then y then z (must match docking_core.h layout).
    wx, wy, wz = well
    two_sig2 = 2.0 * sigma * sigma
    for iz in range(nz):
        for iy in range(ny):
            row = []
            for ix in range(nx):
                px = ox + ix * spacing
                py = oy + iy * spacing
                pz = oz + iz * spacing
                d2 = (px - wx) ** 2 + (py - wy) ** 2 + (pz - wz) ** 2
                e = -depth * math.exp(-d2 / two_sig2)
                row.append(f"{e:.6f}")
            lines.append(" ".join(row))

    lines.append(f"# Rigid ligand: {len(ligand)} atoms in ligand-local coords (centroid ~ origin).")
    lines.append(f"LIGAND {len(ligand)}")
    for (x, y, z, w) in ligand:
        lines.append(f"{x:.6f} {y:.6f} {z:.6f} {w:.6f}")

    cx, cy, cz = pocket_center
    lines.append(f"# Search: {n_trans}^3 translations over +/-{trans_range} A, {n_rot}^3 rotations.")
    lines.append(f"SEARCH {n_trans} {n_rot} {trans_range:.6f}  {cx:.6f} {cy:.6f} {cz:.6f}")
    return "\n".join(lines) + "\n"


def make_ligand():
    """A tiny rigid 5-atom ligand, centroid at the origin. Slightly anisotropic
    (longer along x) so the best rotation is not entirely degenerate. Weights ~1
    (a generic probe). Coordinates in Angstrom."""
    atoms = [
        (0.0, 0.0, 0.0, 1.0),     # central atom
        (1.2, 0.0, 0.0, 1.0),     # +x arm (longer axis)
        (-1.2, 0.0, 0.0, 1.0),    # -x arm
        (0.0, 0.7, 0.0, 1.0),     # +y
        (0.0, -0.7, 0.0, 1.0),    # -y
    ]
    # Recentre exactly on the centroid so translation alone aligns it to the well.
    n = len(atoms)
    mx = sum(a[0] for a in atoms) / n
    my = sum(a[1] for a in atoms) / n
    mz = sum(a[2] for a in atoms) / n
    return [(x - mx, y - my, z - mz, w) for (x, y, z, w) in atoms]


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic docking problem.")
    ap.add_argument("--n-grid", type=int, default=16, help="grid points per axis")
    ap.add_argument("--spacing", type=float, default=0.5, help="grid spacing (A)")
    ap.add_argument("--depth", type=float, default=10.0, help="well depth (kcal/mol)")
    ap.add_argument("--sigma", type=float, default=1.5, help="well width (A)")
    ap.add_argument("--n-trans", type=int, default=7, help="translation samples/axis")
    ap.add_argument("--n-rot", type=int, default=3, help="rotation samples/axis")
    ap.add_argument("--trans-range", type=float, default=1.5, help="+/- A around pocket centre")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    # Put the well slightly OFF the pocket centre so the recovered best translation
    # is non-zero and interesting (the engine has to find it, not sit at index 0).
    well = (0.5, -0.5, 0.0)
    pocket_center = (0.0, 0.0, 0.0)
    ligand = make_ligand()

    text = build(args.n_grid, args.spacing, args.depth, args.sigma, well, ligand,
                 args.n_trans, args.n_rot, args.trans_range, pocket_center)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    n_poses = (args.n_trans ** 3) * (args.n_rot ** 3)
    print(f"[make_synthetic] wrote {args.out}")
    print(f"    grid {args.n_grid}^3 @ {args.spacing} A, well at {well} depth {args.depth}; "
          f"{n_poses} poses; SYNTHETIC (deterministic, no RNG).")


if __name__ == "__main__":
    main()
