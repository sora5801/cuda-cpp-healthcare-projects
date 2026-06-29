#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic AMOEBA ensemble sample
# ---------------------------------------------------------------------------
# Project 1.21 : Polarizable / AMOEBA Force Field MD
#
# WHY THIS EXISTS
#   Real AMOEBA parameter sets (Tinker .prm / .key files) and protein
#   geometries are large and license-encumbered; we cannot redistribute them.
#   So we ship a TINY, clearly-SYNTHETIC stand-in that exercises the exact same
#   solver (a self-consistent induced-dipole conjugate-gradient solve) and lets
#   the demo run fully offline. Synthetic data is always LABELED synthetic.
#
#   This script writes the SAME ensemble that make_synthetic_ensemble() builds
#   in src/reference_cpu.cpp, in the text format src/load_ensemble() reads:
#
#       M tol max_iter
#       <repeated M times:>
#          n
#          x y z  Ex Ey Ez  alpha          (one line per atom)
#
#   Geometry: a 3-atom "water-like" trio -- a central polarizable site at the
#   origin and two partners at (+/- d, 0, 0). The half-separation d sweeps from
#   4.0 down to 2.0 Angstrom across the M members, so the dipole-dipole coupling
#   (~1/d^3) strengthens and the induced dipoles / polarization energy grow.
#   A uniform external field (0.05 along +x) drives the polarization.
#
# USAGE
#   python scripts/make_synthetic.py                 # 8 members -> data/sample/
#   python scripts/make_synthetic.py --members 16    # bigger synthetic ensemble
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT  = ROOT / "data" / "sample" / "amoeba_ensemble.txt"


def build(members: int):
    """Return the list of output lines for `members` synthetic systems.

    Mirrors make_synthetic_ensemble() in src/reference_cpu.cpp exactly so the
    file-driven run and the built-in fallback produce identical results."""
    tol, max_iter = 1.0e-8, 64
    lines = [f"{members} {tol:.1e} {max_iter}"]
    dmax, dmin = 4.0, 2.0
    for m in range(members):
        frac = (m / (members - 1)) if members > 1 else 0.0
        d = dmax - (dmax - dmin) * frac                 # half-separation [Angstrom]
        n = 3
        lines.append(str(n))
        # central atom (alpha ~ water O = 1.40 A^3), then the two partners (1.10).
        atoms = [
            (0.0, 0.0, 0.0, 1.40),
            (d,   0.0, 0.0, 1.10),
            (-d,  0.0, 0.0, 1.10),
        ]
        for (x, y, z, alpha) in atoms:
            # x y z  Ex Ey Ez  alpha ; uniform permanent field 0.05 along +x.
            # Full repr() precision so the file-driven run reproduces the in-memory
            # builder (make_synthetic_ensemble in reference_cpu.cpp) to the last
            # bit -- otherwise a 6-decimal truncation of the swept separation would
            # shift the induced dipoles in the final printed digit.
            lines.append(f"{x!r} {y!r} {z!r}  0.05 0.0 0.0  {alpha!r}")
    return lines


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic AMOEBA induced-dipole ensemble.")
    ap.add_argument("--members", type=int, default=8, help="number of polarization systems")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    lines = build(max(1, args.members))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {out}  ({args.members} members; SYNTHETIC)")


if __name__ == "__main__":
    main()
