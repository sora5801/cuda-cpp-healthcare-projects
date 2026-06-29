#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic ligand library
# ---------------------------------------------------------------------------
# Project 1.4 : Ultra-Large Virtual Screening
#
# WHY SYNTHETIC
#   Real virtual-screening libraries come from ChEMBL/ZINC/Enamine with 2-D
#   descriptors + pharmacophore features computed by RDKit (see download_data.*
#   and data/README.md). To keep the demo OFFLINE and REPRODUCIBLE we generate a
#   clearly-SYNTHETIC library whose surrogate-dock scores span a wide range, so
#   the top-K hit list is interesting AND deterministic.
#
#   The library is engineered (PATTERNS.md sec 6) to embed a KNOWN ANSWER:
#     * a handful of "designed binders" that present every pharmacophore feature
#       the target rewards AND sit right on the target's ideal size/logP/PSA ->
#       they should top the ranking;
#     * "decoys" with random properties inside drug-like ranges -> middling
#       scores;
#     * "filter-fail" ligands that deliberately violate Lipinski/Veber (too
#       heavy, too greasy, too polar, too floppy) -> rejected by the cascade and
#       never scored.
#   A fixed RNG seed makes the output byte-for-byte reproducible (so the demo's
#   expected_output.txt is stable). Scores are chemically MEANINGLESS -- this is
#   a synthetic teaching sample, labelled as such everywhere.
#
# OUTPUT FORMAT  (must match src/reference_cpu.cpp::load_library; data/README.md):
#   n                                            # ligand count
#   TARGET mw_opt logp_opt_x100 psa_opt feat_required_hex
#   mw logp_x100 hbd hba rotb psa feat_hex       # one line per ligand (n lines)
#   ('#'-prefixed comment lines and blank lines are ignored by the loader)
#
# USAGE
#   python scripts/make_synthetic.py                 # default n=64
#   python scripts/make_synthetic.py --n 1000000     # a "campaign scale" set
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "ligands_sample.txt"

# The screening TARGET (binding-site wish list). Chosen so designed binders sit
# squarely inside the drug-like window and the feature mask is non-trivial.
TARGET_MW_OPT        = 350       # ideal molecular weight (Da)
TARGET_LOGP_OPT_X100 = 250       # ideal logP = 2.50
TARGET_PSA_OPT       = 75        # ideal polar surface area (A^2)
TARGET_FEAT_REQUIRED = 0x00A5B3  # pharmacophore features the pocket rewards


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic ligand library.")
    ap.add_argument("--n", type=int, default=64, help="number of ligands")
    ap.add_argument("--seed", type=int, default=7, help="RNG seed (determinism)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    req_bits = bin(TARGET_FEAT_REQUIRED).count("1")  # how many features to match

    rows = []
    # How many of each engineered class. The rest are decoys. We cap the special
    # classes so they stay a small, fixed fraction even for large n.
    n_binders = min(4, args.n)                    # near-perfect designed binders
    n_fail    = min(max(args.n // 8, 1), args.n)  # ligands that fail the cascade

    for i in range(args.n):
        if i < n_binders:
            # DESIGNED BINDER: present all rewarded features, sit near the target
            # optimum (tiny deterministic jitter so the 4 binders rank distinctly).
            mw   = TARGET_MW_OPT + (i - 1) * 5
            logp = TARGET_LOGP_OPT_X100 + (i - 1) * 8
            psa  = TARGET_PSA_OPT + (i - 1) * 2
            hbd, hba, rotb = 2, 5, 4
            feat = TARGET_FEAT_REQUIRED           # full pharmacophore overlap
        elif i < n_binders + n_fail:
            # FILTER-FAIL: deliberately violate a drug-likeness rule (rotate which
            # rule fails so all cascade branches get exercised).
            which = i % 3
            mw   = 650 if which == 0 else 300     # too heavy (Lipinski MW > 500)
            logp = 720 if which == 1 else 200     # too greasy (logP > 5)
            psa  = 180 if which == 2 else 60      # too polar  (Veber PSA > 140)
            hbd, hba, rotb = 3, 6, (14 if which == 0 else 5)  # which==0 also floppy
            feat = rng.getrandbits(24)
        else:
            # DECOY: random but inside drug-like ranges -> passes the cascade,
            # scores in the middle of the pack (partial feature overlap by chance).
            mw   = rng.randint(180, 480)
            logp = rng.randint(-50, 480)
            psa  = rng.randint(20, 130)
            hbd  = rng.randint(0, 5)
            hba  = rng.randint(1, 9)
            rotb = rng.randint(0, 9)
            feat = rng.getrandbits(24)
        rows.append((mw, logp, hbd, hba, rotb, psa, feat))

    # Shuffle so the best hit is NOT trivially ligand[0] (the top-K must be earned).
    rng.shuffle(rows)

    lines = [
        "# Synthetic ligand library for project 1.4 (Ultra-Large Virtual Screening).",
        "# SYNTHETIC DATA -- scores are chemically meaningless; teaching use only.",
        f"# target rewards {req_bits} pharmacophore features (mask "
        f"0x{TARGET_FEAT_REQUIRED:06X}); designed binders match them all.",
        "# columns:  mw  logp_x100  hbd  hba  rotb  psa  feat_hex",
        f"{args.n}",
        f"TARGET {TARGET_MW_OPT} {TARGET_LOGP_OPT_X100} {TARGET_PSA_OPT} "
        f"0x{TARGET_FEAT_REQUIRED:06X}",
    ]
    for (mw, logp, hbd, hba, rotb, psa, feat) in rows:
        lines.append(f"{mw} {logp} {hbd} {hba} {rotb} {psa} 0x{feat:06X}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {out}  (n={args.n}; SYNTHETIC, seed={args.seed})")


if __name__ == "__main__":
    main()
