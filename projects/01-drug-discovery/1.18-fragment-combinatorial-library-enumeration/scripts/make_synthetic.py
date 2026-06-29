#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic synthon catalog
# ---------------------------------------------------------------------------
# Project 1.18 : Fragment / Combinatorial Library Enumeration
#
# WHY SYNTHETIC
#   Real building-block catalogs (Enamine, ChemSpace) are large commercial files
#   that require registration and cannot be redistributed here (CLAUDE.md sec.8).
#   To keep the demo OFFLINE and REPRODUCIBLE we generate a clearly-SYNTHETIC
#   three-slot catalog whose product descriptors are engineered so that a
#   well-defined, deterministic FRACTION of products passes the Lipinski + Veber
#   drug-likeness filter -- making the result interesting and verifiable.
#
#   This is a teaching stand-in. The descriptor numbers are physically plausible
#   (an amine donates HBD/HBA + a little MW/cLogP/TPSA, a lipophilic cap pushes
#   cLogP/MW up so that the heaviest combinations FAIL the filter) but they are
#   INVENTED, not measured. Real synthon descriptors come from RDKit run on the
#   actual building-block SMILES -- see scripts/download_data.* and data/README.md.
#
# OUTPUT FORMAT (full spec in data/README.md):
#   <N_SLOTS>                                     # = 3
#   SLOT 0 <size0>                                # header for slot 0
#   <name> <MW> <cLogP> <TPSA> <HBD> <HBA>        # size0 synthon rows
#   SLOT 1 <size1>
#   ...
#   SLOT 2 <size2>
#   ...
#   ('#' lines and blank lines are comments.)
#
# USAGE
#   python scripts/make_synthetic.py                 # default 6 x 6 x 6 = 216
#   python scripts/make_synthetic.py --per-slot 40   # 40^3 = 64000 products
# ===========================================================================
import argparse
from pathlib import Path

N_SLOTS = 3                                   # MUST match src/product_core.h
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "synthons_sample.txt"

# Per-slot "chemistry": each slot contributes a base descriptor vector that grows
# with the synthon index. The three slots model an Ugi-like 3-component reaction:
#   slot 0 = amine head      (adds polar HBD/HBA + modest MW)
#   slot 1 = aldehyde/acid   (adds MW + some cLogP)
#   slot 2 = lipophilic cap  (adds the most MW + cLogP -> heavy combos FAIL)
# Descriptor order is [MW, cLogP, TPSA, HBD, HBA] (see DescIndex in product_core.h).
SLOT_PROFILES = [
    # (prefix, base[MW,cLogP,TPSA,HBD,HBA], per-index step[MW,cLogP,TPSA,HBD,HBA])
    ("A", [ 60.0, 0.20, 26.0, 1, 1], [12.0, 0.15,  3.0, 0, 0]),  # amine head
    ("B", [ 90.0, 0.60, 17.0, 0, 1], [18.0, 0.30,  4.0, 0, 1]),  # aldehyde/acid
    ("C", [110.0, 1.40,  9.0, 0, 0], [26.0, 0.55,  2.0, 0, 0]),  # lipophilic cap
]


def synthon_row(slot, j):
    """Return (name, [MW, cLogP, TPSA, HBD, HBA]) for synthon j of `slot`.
    Values increase linearly with j so that low-index combinations are small and
    drug-like while high-index combinations are heavy/greasy and fail Lipinski.
    The arithmetic is exact decimals so the written file round-trips precisely."""
    prefix, base, step = SLOT_PROFILES[slot]
    desc = [base[d] + step[d] * j for d in range(5)]
    return f"{prefix}{j:02d}", desc


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic synthon catalog.")
    ap.add_argument("--per-slot", type=int, default=6,
                    help="synthons per slot (library = per_slot^3 products)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    s = args.per_slot

    lines = []
    lines.append("# SYNTHETIC synthon catalog for project 1.18 (NOT real chemistry).")
    lines.append("# Format: N_SLOTS, then per slot 'SLOT k size' + size rows of")
    lines.append("#   <name> <MW> <cLogP> <TPSA> <HBD> <HBA>")
    lines.append(f"{N_SLOTS}")
    for k in range(N_SLOTS):
        lines.append(f"SLOT {k} {s}")
        for j in range(s):
            name, d = synthon_row(k, j)
            # Print MW/cLogP/TPSA with fixed decimals; HBD/HBA as integers.
            lines.append(f"{name} {d[0]:.3f} {d[1]:.3f} {d[2]:.3f} {int(d[3])} {int(d[4])}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_products = s ** N_SLOTS
    print(f"[make_synthetic] wrote {args.out}  "
          f"({s} x {s} x {s} = {n_products} products; SYNTHETIC)")


if __name__ == "__main__":
    main()
