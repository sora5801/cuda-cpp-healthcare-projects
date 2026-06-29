#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic FASTA sample
# ---------------------------------------------------------------------------
# Project 1.34 : Amyloid / Aggregation Propensity Prediction
#
# WHY THIS EXISTS
#   The real aggregation databases (AmyPro, WALTZ-DB, ThT kinetics sets) cannot
#   be redistributed wholesale and some need registration. So the committed
#   demo runs on a TINY, CLEARLY-SYNTHETIC FASTA generated here. The point of
#   the sample is to be *interpretable*: we engineer sequences whose correct
#   aggregation ranking is known by construction, so a learner can confirm the
#   pipeline recovers it (PATTERNS.md §6). Everything is labeled SYNTHETIC.
#
# WHAT IT BUILDS  (4 short, deterministic, designed sequences)
#   * SYNTH_AGG_core        -- a soluble flank with a strong hydrophobic/beta
#                              core (V/I/L/F) buried in the middle. This is the
#                              designed "obvious hot spot" -> should rank #1.
#   * SYNTH_AGG_polyV       -- an even longer pure poly-(V/I) stretch -> very
#                              high, broad APR; competes for the top.
#   * SYNTH_MIXED           -- alternating prone/soluble residues: a peak that
#                              hovers near the threshold (teaches the cutoff).
#   * SYNTH_SOLUBLE_charged -- charged/Pro/Gly-rich: low propensity throughout,
#                              the negative control -> ranks last.
#
#   These are INVENTED constructs in the *spirit* of known amyloidogenic motifs
#   (aliphatic/aromatic cores drive beta-aggregation); they are NOT real protein
#   sequences and carry no biological identity or clinical meaning. The famous
#   experimental hexapeptides (Abeta KLVFFA, IAPP NFGAIL) are described in
#   THEORY.md for context but deliberately NOT reproduced here.
#
# USAGE
#   python scripts/make_synthetic.py           # writes data/sample/amyloid_sample.fasta
#   python scripts/make_synthetic.py --out other.fasta
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "amyloid_sample.fasta"

# Each entry: (header, sequence). Sequences use one-letter codes. Designed so
# the per-residue intrinsic-propensity scale + 7-residue window mean produces a
# known ranking (see scale in src/propensity.h). All SYNTHETIC.
SEQUENCES = [
    # Soluble charged/turn flanks (D,E,K,R,G,P,S -> low) wrapping a buried
    # aliphatic/aromatic beta-core (I,L,V,F,Y -> high): a textbook designed APR.
    ("SYNTH_AGG_core  [synthetic; buried hydrophobic beta-core]",
     "DEKRGSPDEK" "GSDEKPGSDE" "VFILVFILVF" "ILVFYWILVF" "DEKRGSPDEK" "GSDEKPGSDE"),

    # A long, broad poly-(V/I/L/F) stretch: very high, wide APR across most of
    # the chain. Competes with the core sequence for the #1 rank.
    ("SYNTH_AGG_polyV [synthetic; broad aliphatic stretch]",
     "GSDEK" "VIVIVILILILVFVFVIVIVILILIL" "GSDEK"),

    # Alternating prone/soluble residues: smoothed peak sits NEAR the threshold,
    # so this sequence teaches how the cutoff carves APRs out of a noisy profile.
    ("SYNTH_MIXED [synthetic; near-threshold alternating]",
     "VKVKVKVKVK" "IEIEIEIEIE" "LDLDLDLDLD" "VKVKVKVKVK"),

    # Charged / proline / glycine rich throughout: the soluble negative control.
    # Low propensity everywhere -> no APR -> ranks last.
    ("SYNTH_SOLUBLE_charged [synthetic; negative control]",
     "DEKRDEKRPG" "SDEKRPGSDE" "KRPGSDEKRP" "GSDEKRPGSD" "EKRPGSDEKR"),
]


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic amyloid FASTA sample.")
    ap.add_argument("--out", default=str(OUT), help="output FASTA path")
    args = ap.parse_args()

    lines = []
    for header, seq in SEQUENCES:
        lines.append(">" + header)
        # Wrap the sequence at 60 cols (standard FASTA), purely cosmetic.
        for i in range(0, len(seq), 60):
            lines.append(seq[i:i + 60])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    total = sum(len(s) for _, s in SEQUENCES)
    print(f"[make_synthetic] wrote {out}  "
          f"({len(SEQUENCES)} SYNTHETIC proteins, {total} residues total)")


if __name__ == "__main__":
    main()
