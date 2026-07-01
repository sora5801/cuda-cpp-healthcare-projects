#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic toy metabolic model
# ---------------------------------------------------------------------------
# Project 6.12 : Metabolic Flux / Constraint-Based Modeling
#
# WHY THIS EXISTS
#   Real genome-scale metabolic models (Recon3D, the BiGG models) are large SBML
#   files and, while public, are far too big to commit and would bury the teaching
#   point. So we hand-craft a TINY, clearly-SYNTHETIC toy network whose knockout
#   answers can be read off by eye -- and label it synthetic everywhere it appears
#   (CLAUDE.md section 8). scripts/download_data.* points at the real models for
#   anyone who wants to scale up.
#
# THE TOY NETWORK  (4 internal metabolites A,B,C,D; 8 reactions)
#   Substrate A is taken up, flows through a branched pathway to biomass. The
#   network is engineered so a single-reaction knockout screen produces ALL THREE
#   interesting outcomes, each teaching a metabolic concept:
#
#     R0 uptake_A   ( -> A)      the sole carbon source        ESSENTIAL
#     R1 A->B_1     (A -> B)     main step, has an isozyme      neutral (isozyme covers)
#     R2 A->B_2iso  (A -> B)     isozyme of R1 (redundant)     neutral
#     R3 B->C       (B -> C)     main trunk                     REDUCED (only bypass left)
#     R4 A->C_byp   (A -> C)     low-capacity bypass (ub=3)     neutral (spare capacity)
#     R5 C->D       (C -> D)     sole route C->D                ESSENTIAL
#     R6 D->biomass (D -> )      the OBJECTIVE reaction          ESSENTIAL
#     R7 A->waste   (A -> )      overflow sink                   neutral
#
#   Uptake caps growth at 10 (ub of R0). Knocking out R3 forces carbon through the
#   ub=3 bypass R4, so growth falls to 3 -- a "growth-reducing" (not lethal) gene.
#   Isozyme R1/R2 are mutually redundant, so deleting either alone does nothing.
#   The three ESSENTIAL reactions (uptake, C->D, biomass) have no alternative
#   route, so deleting any of them collapses growth to 0.
#
#   This mirrors real essentiality biology: essential genes are candidate drug
#   targets; synthetic-lethal PAIRS (both isozymes) are a known combination-therapy
#   idea (left as an exercise in README.md).
#
# OUTPUT FORMAT (parsed by src/reference_cpu.cpp :: load_model):
#   '#'-lines are comments; a '#names:' line supplies reaction labels; the numeric
#   stream is  nmet nrxn / S (nmet rows) / lb / ub / c.
#
# USAGE
#   python scripts/make_synthetic.py                    # -> data/sample/toy_core_model.txt
#   python scripts/make_synthetic.py --out other.txt
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent            # the project folder
OUT = ROOT / "data" / "sample" / "toy_core_model.txt"

# Reaction labels (order defines column order of S, lb, ub, c).
NAMES = ["uptake_A", "A->B_1", "A->B_2iso", "B->C", "A->C_byp", "C->D", "D->biomass", "A->waste"]

# Dense stoichiometry S: rows = metabolites A,B,C,D; cols = reactions above.
#   +1 = produced by the reaction, -1 = consumed, 0 = not involved.
S = [
    #  R0  R1  R2  R3  R4  R5  R6  R7
    [  1, -1, -1,  0, -1,  0,  0, -1],   # A
    [  0,  1,  1, -1,  0,  0,  0,  0],   # B
    [  0,  0,  0,  1,  1, -1,  0,  0],   # C
    [  0,  0,  0,  0,  0,  1, -1,  0],   # D
]
# Flux bounds. R0 uptake is capped at 10 (nutrient supply); R4 bypass at 3 (low
# enzyme capacity); everything else effectively unbounded (1000 ~ "infinity").
LB = [0, 0, 0, 0, 0, 0, 0, 0]
UB = [10, 1000, 1000, 1000, 3, 1000, 1000, 1000]
# Objective: maximise the biomass reaction R6 only.
C = [0, 0, 0, 0, 0, 0, 1, 0]


def fmt_row(vals):
    """Space-separated integer/float row with aligned width for readability."""
    return " ".join(f"{v:>5g}" for v in vals)


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic toy FBA model.")
    ap.add_argument("--out", default=str(OUT), help="output model path")
    args = ap.parse_args()

    nmet, nrxn = len(S), len(NAMES)
    lines = []
    lines.append("# SYNTHETIC toy metabolic model for FBA teaching (project 6.12).")
    lines.append("# NOT a real organism -- hand-crafted so the knockout screen is")
    lines.append("# readable by eye. See scripts/make_synthetic.py for the biology.")
    lines.append("# Format: nmet nrxn / S (nmet rows) / lb / ub / c.  Metabolites: A B C D.")
    lines.append("#names: " + " ".join(NAMES))
    lines.append(f"{nmet} {nrxn}")
    lines.append("# --- stoichiometry S (rows = metabolites A,B,C,D) ---")
    for row in S:
        lines.append(fmt_row(row))
    lines.append("# --- lower bounds lb ---")
    lines.append(fmt_row(LB))
    lines.append("# --- upper bounds ub (R0 uptake<=10, R4 bypass<=3) ---")
    lines.append(fmt_row(UB))
    lines.append("# --- objective c (maximise biomass reaction R6) ---")
    lines.append(fmt_row(C))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"({nmet} metabolites x {nrxn} reactions; SYNTHETIC toy model)")


if __name__ == "__main__":
    main()
