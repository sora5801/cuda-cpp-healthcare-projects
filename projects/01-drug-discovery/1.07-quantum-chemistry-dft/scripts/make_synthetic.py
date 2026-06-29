#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the tiny molecule input files
# ---------------------------------------------------------------------------
# Project 1.7 : Quantum Chemistry / DFT  (reduced-scope RHF/SCF)
#
# WHAT THIS DOES
#   Emits the committed sample molecule(s) used by demo/run_demo. These are NOT
#   experimental data -- they are STANDARD, PUBLIC textbook geometries (equilibrium
#   bond lengths) written out in the project's simple input format. Nothing here is
#   patient-derived; everything is synthetic/public (CLAUDE.md section 8).
#
# THE FORMAT (see data/README.md)
#   line 1: "<natoms> <charge>"      charge = net molecular charge
#   then natoms lines: "<Z> <x> <y> <z>"   coordinates in BOHR (atomic units)
#   '#' comment lines and blank lines are ignored by the loader.
#
# USAGE
#   python scripts/make_synthetic.py                 # writes data/sample/h2.txt
#   python scripts/make_synthetic.py --mol heh+      # writes a HeH+ cation
#   python scripts/make_synthetic.py --mol he        # writes a He atom
#
# This teaching build ships only H and He basis sets, so only those elements work.
# ===========================================================================
import argparse
import os

# Equilibrium geometries in BOHR (1 Bohr = 0.52917721 Angstrom). These are common
# textbook values used to reproduce the canonical STO-3G energies.
MOLECULES = {
    # name : (charge, [(Z, x, y, z), ...], reference STO-3G total energy in Hartree)
    "h2":   (0, [(1, 0.0, 0.0, 0.0), (1, 0.0, 0.0, 1.4)], -1.1167),
    "heh+": (1, [(2, 0.0, 0.0, 0.0), (1, 0.0, 0.0, 1.4632)], -2.8606),
    "he":   (0, [(2, 0.0, 0.0, 0.0)], -2.8077),
}


def write_molecule(path: str, name: str) -> None:
    """Write one molecule file in the project's text format."""
    charge, atoms, e_ref = MOLECULES[name]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# SYNTHETIC sample molecule: {name.upper()} (standard equilibrium geometry).\n")
        f.write('# Line 1: "<natoms> <charge>"   charge = net molecular charge.\n')
        f.write('# Then one line per atom: "<Z> <x> <y> <z>"   coordinates in BOHR (atomic units).\n')
        f.write(f"# Reference STO-3G RHF total energy ~ {e_ref:.4f} Hartree (textbook). NOT patient data.\n")
        f.write(f"{len(atoms)} {charge}\n")
        for (Z, x, y, z) in atoms:
            f.write(f"{Z} {x} {y} {z}\n")
    print(f"[make_synthetic] wrote {path}  ({name.upper()}, ref ~ {e_ref:.4f} Ha)")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Generate a tiny synthetic molecule input.")
    ap.add_argument("--mol", default="h2", choices=sorted(MOLECULES.keys()),
                    help="which molecule to write (default: h2)")
    ap.add_argument("--out", default=None,
                    help="output path (default: data/sample/<mol>.txt)")
    args = ap.parse_args()
    out = args.out or os.path.join(here, "..", "data", "sample", f"{args.mol}.txt")
    write_molecule(os.path.normpath(out), args.mol)


if __name__ == "__main__":
    main()
