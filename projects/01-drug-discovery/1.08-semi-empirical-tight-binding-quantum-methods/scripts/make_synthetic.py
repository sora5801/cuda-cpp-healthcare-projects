#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic molecule batch
# ---------------------------------------------------------------------------
# Project 1.8 : Semi-Empirical & Tight-Binding Quantum Methods
#
# WHY THIS EXISTS
#   The catalog datasets (ANI-1, QM9, GMTKN55) are large 3-D-geometry sets that
#   need downloading and a chemistry toolkit to turn into pi-system graphs. For a
#   self-contained, offline, instantly-interpretable demo we instead generate a
#   tiny batch of TEXTBOOK conjugated hydrocarbons whose Huckel (tight-binding)
#   eigenvalues are known in CLOSED FORM. That lets the demo double-check itself
#   against analytic chemistry, not just CPU==GPU agreement. The data is fully
#   SYNTHETIC (idealised connectivity graphs, no real coordinates) and is labeled
#   synthetic everywhere (see data/README.md).
#
#   Each molecule is described purely by its pi-system CONNECTIVITY (which sp2
#   carbons are bonded). That graph IS the tight-binding model input.
#
# SAMPLE FORMAT (consumed by load_batch() in src/reference_cpu.cpp):
#   # comment lines start with '#'
#   NUM_MOL
#   for each molecule:
#     NAME  N  NBONDS
#     NBONDS lines of "i j"   (0-based bond between pi-atoms i and j)
#
# USAGE
#   python scripts/make_synthetic.py          # writes data/sample/molecules_sample.txt
#   python scripts/make_synthetic.py --out other.txt
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent              # the project folder
OUT = ROOT / "data" / "sample" / "molecules_sample.txt"


def chain(n):
    """Linear polyene: atoms 0-1-2-...-(n-1). Bonds between consecutive atoms."""
    return [(i, i + 1) for i in range(n - 1)]


def ring(n):
    """Monocyclic ring: a chain plus the closing bond (n-1)-0."""
    return chain(n) + [(n - 1, 0)]


# Standard naphthalene pi-graph: two fused 6-rings sharing one edge (10 carbons,
# 11 bonds). Ring A = atoms 0-1-2-3-4-5 (closed), and the SHARED (fusion) edge is
# 0-5 whose endpoints are the two bridgehead carbons. Ring B reuses those
# bridgeheads and adds atoms 6,7,8,9: 0-6-7-8-9-5. This is the canonical
# adjacency used in Huckel textbooks; its 10-electron total pi energy is exactly
# 13.6832 |beta| (a strong analytic check -- see THEORY.md "How we verify").
NAPHTHALENE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),   # ring A (incl. fusion edge 0-5)
    (0, 6), (6, 7), (7, 8), (8, 9), (9, 5),           # ring B sharing bridgeheads 0,5
]


# The batch: (name, atom-count, bond-list). Chosen so several have closed-form
# Huckel spectra (see THEORY.md "How we verify correctness").
MOLECULES = [
    ("ethylene",        2,  chain(2)),    # E_pi = 2.000 |beta|, gap = 2.000
    ("allyl",           3,  chain(3)),    # radical (3 e-); MO energies 0, +-sqrt(2)
    ("butadiene",       4,  chain(4)),    # E_pi = 4.472 |beta|
    ("benzene",         6,  ring(6)),     # E_pi = 8.000 |beta|, gap = 2.000 (aromatic)
    ("cyclobutadiene",  4,  ring(4)),     # gap = 0  (antiaromatic; two NB orbitals)
    ("hexatriene",      6,  chain(6)),    # E_pi = 6.988 |beta|
    ("cyclopentadienyl",5,  ring(5)),     # 5-ring (as neutral radical here)
    ("naphthalene",     10, NAPHTHALENE_EDGES),  # E_pi = 13.683 |beta|
]


def build_text():
    lines = []
    lines.append("# SYNTHETIC tight-binding (Huckel) molecule batch -- project 1.8")
    lines.append("# Each molecule is an idealised pi-system CONNECTIVITY graph (no real")
    lines.append("# coordinates). Format: NUM_MOL, then per molecule 'NAME N NBONDS'")
    lines.append("# followed by NBONDS '<i> <j>' 0-based pi-atom bonds. SYNTHETIC DATA.")
    lines.append(str(len(MOLECULES)))
    for name, n, bonds in MOLECULES:
        # sanity: bond indices in range
        for (i, j) in bonds:
            assert 0 <= i < n and 0 <= j < n and i != j, f"{name}: bad bond {i},{j}"
        lines.append(f"{name} {n} {len(bonds)}")
        for (i, j) in bonds:
            lines.append(f"{i} {j}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic Huckel molecule batch.")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(build_text(), encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  ({len(MOLECULES)} molecules; SYNTHETIC)")


if __name__ == "__main__":
    main()
