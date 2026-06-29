#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic kinase-panel sample
# ---------------------------------------------------------------------------
# Project 1.29 : Kinase Selectivity Panel Scoring
#
# WHY THIS EXISTS
#   The real datasets this project mirrors (KLIFS interaction fingerprints,
#   KINOMEscan S-scores, ChEMBL kinase activities) either require registration or
#   cannot be redistributed wholesale. So the committed demo runs on a TINY,
#   clearly-SYNTHETIC panel generated here. It is labeled synthetic everywhere
#   (data/README.md, the file header below, this script). It is NOT real
#   pharmacology and must never be used for any decision (CLAUDE.md sec 8).
#
# WHAT IT ENCODES  (so the demo result is meaningful and verifiable)
#   * NFEAT = 8 toy pharmacophore channels (must match selectivity_core.h):
#       0 donor  1 acceptor  2 hydrophobic  3 aromatic
#       4 ionic+ 5 ionic-    6 halogen      7 HINGE motif (highest weight)
#   * ONE query compound, modeled as a typical ATP-competitive inhibitor:
#       strong hinge binder, aromatic, a couple of H-bonds, mild hydrophobicity.
#   * A small panel of kinases with hand-set pocket requirement vectors so that:
#       - the INTENDED TARGET (ABL1) is the single strongest hit (rank #1),
#       - a few related kinases (the off-targets that make selectivity hard) also
#         clear the pK>=6 "hit" threshold -> a small but non-zero S-score, and
#       - most of the panel stays below threshold (selective-ish compound).
#   This embeds a known answer the demo recovers (PATTERNS.md sec 6).
#
# FILE FORMAT (whitespace-separated; '#'-comment lines are ignored by the loader)
#     N  NFEAT
#     LIGAND  f0 f1 ... f7
#     <name> <bias> r0 r1 ... r7        (one line per kinase)
#
# USAGE
#   python scripts/make_synthetic.py                 # writes the committed sample
#   python scripts/make_synthetic.py --out other.txt # custom path
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "kinase_panel_sample.txt"

NFEAT = 8   # MUST equal NFEAT in src/selectivity_core.h

# Per-channel weights, kept in sync with feature_weight() in selectivity_core.h.
# Documented here only so the comments below can show the arithmetic; the C++
# build is the source of truth for the actual scoring.
WEIGHTS = [3, 3, 2, 2, 4, 4, 2, 6]   # idx 7 (hinge) dominates affinity

# The query compound's per-channel feature OFFERS (what it can satisfy).
#   donor=1, acceptor=2, hydrophobic=3, aromatic=2, ionic+=0, ionic-=0,
#   halogen=1, hinge=2  -> a classic ATP-competitive scaffold (strong hinge).
LIGAND = [1, 2, 3, 2, 0, 0, 1, 2]

# The panel: (name, bias, requirement-vector). Requirements are "how much of each
# feature the pocket wants"; the score uses min(offer, need) per channel (you can
# only form the overlap). Hand-tuned so ABL1 is the clear #1 and the S-score is
# small. Raw score = bias + sum_f min(LIGAND[f], req[f]) * WEIGHTS[f];
# predicted pK*1000 = 4000 + 50 * raw; "hit" iff pK*1000 >= 6000  (raw >= 40).
PANEL = [
    # name      bias  donor acc hyd aro i+  i-  hal hinge
    ("ABL1",      8,  [ 2,   2,  4,  2,  0,  0,  2,  3]),  # intended target  -> strongest
    ("SRC",       6,  [ 1,   2,  3,  2,  0,  0,  1,  3]),  # close off-target -> hit
    ("KIT",       5,  [ 2,   1,  3,  1,  0,  0,  1,  3]),  # off-target       -> hit
    ("PDGFRA",    4,  [ 1,   2,  2,  2,  0,  0,  1,  2]),  # borderline
    ("EGFR",      3,  [ 1,   1,  2,  2,  0,  0,  0,  2]),  # below threshold
    ("LCK",       4,  [ 1,   1,  2,  1,  0,  0,  1,  2]),  # below threshold
    ("BRAF",      2,  [ 0,   1,  3,  2,  0,  0,  0,  1]),  # below threshold
    ("MAPK1",     1,  [ 1,   1,  1,  1,  0,  0,  0,  1]),  # below threshold
    ("CDK2",      2,  [ 0,   2,  2,  1,  0,  0,  0,  1]),  # below threshold
    ("AURKA",     1,  [ 1,   0,  2,  1,  0,  0,  0,  1]),  # below threshold
    ("PLK1",      0,  [ 0,   1,  1,  2,  0,  0,  0,  0]),  # non-binder
    ("CHEK1",     1,  [ 1,   1,  1,  0,  0,  0,  0,  1]),  # non-binder
    ("GSK3B",     0,  [ 0,   1,  2,  1,  0,  0,  0,  0]),  # non-binder
    ("PIM1",      2,  [ 1,   1,  1,  1,  0,  1,  0,  1]),  # non-binder
    ("ROCK1",     1,  [ 0,   2,  1,  1,  0,  0,  0,  1]),  # non-binder
    ("MET",       3,  [ 1,   1,  2,  1,  0,  0,  1,  2]),  # borderline off-target
]


def raw_score(bias, req):
    """Mirror score_kinase() so the script can sanity-print expected pK values."""
    acc = bias
    for f in range(NFEAT):
        acc += min(LIGAND[f], req[f]) * WEIGHTS[f]
    return acc


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic kinase panel sample.")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    lines = []
    lines.append("# SYNTHETIC kinase selectivity panel -- NOT real pharmacology.")
    lines.append("# Project 1.29. Generated by scripts/make_synthetic.py. See data/README.md.")
    lines.append("# Format:  N NFEAT / LIGAND f0..f7 / <name> <bias> r0..r7")
    lines.append(f"{len(PANEL)} {NFEAT}")
    lines.append("# query compound feature offers (donor acc hyd aro ion+ ion- halogen hinge):")
    lines.append("LIGAND " + " ".join(str(v) for v in LIGAND))
    lines.append("# kinase pockets:  name  bias  req[0..7]")
    for name, bias, req in PANEL:
        lines.append(f"{name} {bias} " + " ".join(str(v) for v in req))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Print a small expected-answer summary so a human regenerating the sample can
    # eyeball that ABL1 is the strongest hit and the S-score stays small.
    print(f"[make_synthetic] wrote {args.out}  ({len(PANEL)} kinases, NFEAT={NFEAT}; SYNTHETIC)")
    ranked = sorted(PANEL, key=lambda kp: raw_score(kp[1], kp[2]), reverse=True)
    hits = 0
    for name, bias, req in ranked:
        raw = raw_score(bias, req)
        pk = 4000 + 50 * raw
        hit = pk >= 6000
        hits += 1 if hit else 0
        print(f"    {name:8s} raw={raw:3d}  pK={pk/1000:.3f}  {'HIT' if hit else ''}")
    print(f"    S-score(pK>=6.000) = {hits}/{len(PANEL)} = {hits/len(PANEL):.3f}")


if __name__ == "__main__":
    main()
