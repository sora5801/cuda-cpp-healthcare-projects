#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic covalent-docking sample
# ---------------------------------------------------------------------------
# Project 1.28 : Covalent Docking
#
# WHY THIS EXISTS
#   Real covalent-docking inputs come from PDB co-crystal structures and curated
#   benchmarks (CovDocker, ChEMBL/BindingDB covalent sets) -- several need
#   registration and many forbid redistribution (see data/README.md). So the
#   committed demo runs on a CLEARLY-SYNTHETIC stand-in generated here: a single
#   warhead atom covalently bonded to a cysteine sulfur, a short flexible ligand
#   chain (3 rotatable torsions), and a small rigid "pocket" of 6 atoms.
#
#   This is NOT real chemistry data and must never be presented as such. It is a
#   didactic system engineered so the lowest-energy pose is well-defined and the
#   GPU-vs-CPU agreement is exact.
#
#   The numbers below are the SAME system that src/main.cu hard-codes in
#   built_in_problem(), so the program prints identical results with or without
#   the file. The field order matches load_problem() in src/reference_cpu.cpp.
#
# USAGE
#   python scripts/make_synthetic.py            # writes data/sample/covalent_sample.txt
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent              # the project folder
OUT = ROOT / "data" / "sample" / "covalent_sample.txt"

# sp3 tetrahedral angle (~109.47 degrees) in radians -- the ideal valence /
# covalent-approach angle for a carbon warhead bonding to sulfur.
SP3 = math.acos(-1.0 / 3.0)   # = 1.9106332... rad

# Direction of the FIRST ligand segment from the anchor (unit vector). Chosen so
# the cysteine-S-gamma -- anchor -- first-ligand-atom angle is exactly the sp3
# tetrahedral 109.47 degrees, i.e. first_dir . (1,0,0) = cos(109.47) = -1/3.
# This zeroes the covalent ANGLE penalty for the ideal geometry, so the docking
# SCORE that varies across conformations is purely the ligand-pocket fit.
FIRST_DIR = (-1.0 / 3.0, math.sqrt(1.0 - 1.0 / 9.0), 0.0)

# The 6 fixed pocket atoms: (x, y, z, sigma, epsilon, charge). The positions are
# NOT arbitrary: each sits ~3.82 A (the Lennard-Jones energy minimum, 2^(1/6)*
# sigma) from the closest point the flexible ligand tip can reach. That makes
# every pocket atom ATTRACTIVE-but-never-clashing -- so the energy landscape is
# smooth (no r^-12 blow-up) and has a clear, deep, negative minimum. They were
# found by scanning a shell around the ligand's reachable volume (the same
# scan reproduced in THEORY.md and the exercises).
POCKET = [
    ( 0.382, -3.497,  4.476, 3.40, 0.20,  0.10),
    ( 5.194,  6.384,  0.686, 3.40, 0.20, -0.10),
    (-3.293, -0.819,  6.642, 3.40, 0.20,  0.10),
    ( 0.170,  0.809, -7.523, 3.40, 0.20, -0.10),
    (-7.918,  1.226, -1.574, 3.40, 0.20,  0.05),
    (-2.702,  7.121, -4.190, 3.40, 0.20, -0.05),
]


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic covalent-docking sample.")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# SYNTHETIC covalent-docking problem (NOT real data; teaching only).")
    lines.append("# Project 1.28 -- field order matches load_problem() in src/reference_cpu.cpp.")
    lines.append("# Lines beginning with '#' are comments stripped by the loader.")
    lines.append("#")
    lines.append("# anchor (warhead carbon, bonded to cysteine S-gamma)  x y z [A]")
    lines.append("0.0 0.0 0.0")
    lines.append("# cysteine S-gamma position  x y z [A]")
    lines.append("1.81 0.0 0.0")
    lines.append("# covalent constraint: bond_len_ideal[A]  angle_ideal[rad]  k_bond  k_angle")
    lines.append(f"1.81 {SP3:.6f} 300.0 100.0")
    lines.append("# ligand chain: seg_len[A]  bond_angle[rad]  first_dir(x y z)")
    lines.append(f"1.50 {SP3:.6f} {FIRST_DIR[0]:.6f} {FIRST_DIR[1]:.6f} {FIRST_DIR[2]:.6f}")
    lines.append("# ligand-atom nonbonded: lig_sigma[A]  lig_epsilon[kcal/mol]  lig_charge[e]")
    lines.append("3.40 0.10 -0.10")
    lines.append("# pocket atoms (6): x y z sigma[A] epsilon[kcal/mol] charge[e]")
    for (x, y, z, sig, eps, chg) in POCKET:
        lines.append(f"{x} {y} {z} {sig} {eps} {chg}")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {out}  ({len(POCKET)} pocket atoms, 3 torsions; SYNTHETIC)")


if __name__ == "__main__":
    main()
