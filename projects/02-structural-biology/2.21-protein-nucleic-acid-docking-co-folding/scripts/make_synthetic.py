#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic docking sample
# ---------------------------------------------------------------------------
# Project 2.21 : Protein-Nucleic Acid Docking & Co-Folding (reduced-scope).
#
# WHY THIS EXISTS
#   Real protein-nucleic-acid complexes (PDB, RNA-Puzzles) are large, need
#   parsing/cleanup, and some carry redistribution constraints. So the committed
#   demo runs on a TINY, fully SYNTHETIC complex this script generates -- always
#   LABELED synthetic (CLAUDE.md sec 8). It is engineered so the search has a
#   KNOWN, UNIQUE answer we can recover (PATTERNS.md sec 6), which makes the demo
#   interpretable: the best pose should be the planted "native" pose.
#
# THE PLANTED COMPLEX (all coordinates are fixed-point milli-Angstrom integers)
#   * PROTEIN: a flat 5x5 grid of atoms in the z=0 plane (a model binding
#     "surface"/groove). A 3x3 central patch carries an ASYMMETRIC, chiral
#     "L"-shaped pattern of formal charges (+/-1); the rim is neutral. This is
#     the electrostatic "lock". The asymmetry matters: a symmetric pattern (a
#     checkerboard) would be invariant under several cube rotations, giving
#     MANY tied best poses; the chiral pattern breaks that degeneracy so the
#     native pose is the UNIQUE global maximum (a real lesson about binding
#     specificity -- kept as an exercise in the README).
#   * LIGAND (nucleic-acid fragment): a 3x3 grid of atoms shaped to be the
#     charge-COMPLEMENT of that central patch (every ligand charge is the
#     opposite sign of the protein atom it will sit over). This is the "key".
#   * NATIVE POSE: identity rotation, translated so the ligand sits one contact
#     shell ABOVE the central patch (z = +CONTACT_GAP). There, all 9 ligand
#     atoms are in the contact shell of their partner, every charge pair is
#     opposite (maximally favourable electrostatics), and nothing clashes ->
#     the global maximum score. The pose grid is centred so this native pose is
#     one of the enumerated lattice points (so the search can actually find it).
#
#   Because the cube group includes the identity (rots[0]) and the lattice
#   includes the native translation, the exhaustive search WILL evaluate the
#   native pose and -- by construction -- rank it #1. The demo asserts that.
#
# USAGE
#   python scripts/make_synthetic.py            # writes data/sample/complex_sample.txt
#   python scripts/make_synthetic.py --spacing 3500
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "complex_sample.txt"

# Fixed-point scale: must match COORD_SCALE in src/docking_core.h (milli-A).
COORD_SCALE = 1000


def build(spacing_mA: int):
    """Return (protein, ligand, grid, params) for the planted complex.
       spacing_mA: lattice spacing between adjacent atoms, in milli-Angstrom.
    """
    s = spacing_mA                       # atom spacing (e.g. 3500 = 3.5 A)

    # The chiral "L" charge pattern for the inner 3x3 patch, indexed [row][col]
    # = [iy-1][ix-1]. It is NOT invariant under the cube rotations, so it pins
    # down a single native orientation (see the header note on degeneracy).
    PATCH = [[ 1,  0,  0],
             [ 1,  0,  0],
             [ 1, -1, -1]]

    # ---- protein: 5x5 plane at z=0, central 3x3 carries the "L" pattern -----
    protein = []
    for iy in range(5):
        for ix in range(5):
            x = (ix - 2) * s             # centre the grid on the origin
            y = (iy - 2) * s
            z = 0
            # Rim atoms (the outer ring) are neutral; the inner 3x3 is charged.
            if 1 <= ix <= 3 and 1 <= iy <= 3:
                q = PATCH[iy - 1][ix - 1]
            else:
                q = 0
            protein.append((x, y, z, q))

    # ---- ligand: 3x3 plane, charge = OPPOSITE of the protein patch below ----
    # The ligand atom at local (lx,ly) will sit over protein atom (ix=lx+1,
    # iy=ly+1); give it the opposite charge so the native pose is attractive.
    ligand = []
    for ly in range(3):
        for lx in range(3):
            x = (lx - 1) * s             # centred on the ligand's own origin
            y = (ly - 1) * s
            z = 0
            qp = PATCH[ly][lx]           # the protein atom it pairs with
            ligand.append((x, y, z, -qp))   # opposite sign -> attraction

    # ---- pose grid: centred so the native (0,0,+gap) translation is on it ---
    # The native vertical gap puts the ligand one contact shell above the patch.
    gap = s                              # one atom-spacing above the plane
    # Sweep a small box of translations around the origin in x,y and around the
    # native gap in z. We choose counts so the search stays tiny but non-trivial,
    # and so (tx,ty,tz) = (0,0,gap) is exactly an enumerated lattice point.
    half = s                             # +/- one step in x and y
    grid = dict(
        tx0=-half, ty0=-half, tz0=0,     # origins
        step=s,                          # lattice spacing = atom spacing
        nx=3, ny=3, nz=3,                # 3x3x3 translations
    )
    # With tz0=0, step=s, the z lattice points are {0, s, 2s}; the native gap=s
    # is the middle one -> the native pose (rot 0, tx 0, ty 0, tz s) is enumerated.

    # ---- scoring params (all fixed-point integers) --------------------------
    # Distances are squared milli-A. A contact at exactly one spacing s has
    # d2 = s^2; we set contact_r2 a bit above s^2 and clash_r2 well below it so
    # the native packing is "contact, not clash".
    contact_r2 = int(1.5 * s * s)        # within ~1.22*s counts as contact
    clash_r2 = int(0.25 * s * s)         # within 0.5*s counts as a clash
    params = dict(
        clash_r2=clash_r2,
        contact_r2=contact_r2,
        clash_pen=1000,                  # big penalty per clashing pair
        contact_w=10,                    # bonus per favourable contact
        elec_w=50,                       # electrostatic weight (dominant signal)
    )
    return protein, ligand, grid, params


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic docking sample.")
    ap.add_argument("--spacing", type=int, default=3500,
                    help="atom spacing in milli-Angstrom (default 3500 = 3.5 A)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    protein, ligand, grid, params = build(args.spacing)

    lines = []
    lines.append("# Project 2.21 synthetic protein-nucleic-acid complex (SYNTHETIC).")
    lines.append("# All coordinates are fixed-point milli-Angstrom integers "
                 "(1000 = 1 Angstrom).")
    lines.append("# Format: 'Np Nl' / pose-grid / scoring-params / Np protein atoms"
                 " / Nl ligand atoms.")
    lines.append("# Planted native pose: rotation 0 (identity), translation "
                 "(0, 0, +spacing).")
    lines.append(f"{len(protein)} {len(ligand)}")
    lines.append(f"{grid['tx0']} {grid['ty0']} {grid['tz0']} {grid['step']} "
                 f"{grid['nx']} {grid['ny']} {grid['nz']}")
    lines.append(f"{params['clash_r2']} {params['contact_r2']} "
                 f"{params['clash_pen']} {params['contact_w']} {params['elec_w']}")
    lines.append("# --- protein atoms: x y z charge ---")
    for (x, y, z, q) in protein:
        lines.append(f"{x} {y} {z} {q}")
    lines.append("# --- ligand atoms: x y z charge ---")
    for (x, y, z, q) in ligand:
        lines.append(f"{x} {y} {z} {q}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(Np={len(protein)}, Nl={len(ligand)}; SYNTHETIC)")


if __name__ == "__main__":
    main()
