#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic motif-finding dataset
# ---------------------------------------------------------------------------
# Project 3.29 : Motif Finding in Genomic Sequences
#
# WHY SYNTHETIC
#   Real input is a set of ChIP-seq peak sequences in FASTA (see
#   scripts/download_data.* and data/README.md). To keep the demo OFFLINE,
#   reproducible, and INTERPRETABLE, we generate a clearly-SYNTHETIC FASTA in
#   which a single known motif is PLANTED into random-background DNA at a random
#   offset in each sequence. Because the answer is known, the demo can show that
#   MEME EM recovers it -- the whole point of an instructive sample
#   (PATTERNS.md sec 6: "embed a known answer").
#
#   Method:
#     * Background: each base i.i.d. from a fixed composition (slightly AT-rich,
#       like much mammalian intergenic DNA), drawn from a SEEDED RNG so the
#       output is byte-for-byte reproducible (keeping expected_output.txt stable).
#     * Planted motif: a fixed 8 bp consensus (TGACGTCA, the palindromic
#       CRE/AP-1-like element) with a small per-base mutation rate, inserted at a
#       random offset in each sequence -- exactly the "degenerate motif in noise"
#       that real motif finders face.
#
# OUTPUT FORMAT (data/README.md): standard FASTA --
#     >seq_<i>  synthetic planted_at=<offset> motif=<consensus>
#     ACGT...                                  (one wrapped sequence per record)
#
# USAGE
#   python scripts/make_synthetic.py                       # default: 12 seqs x 60 bp
#   python scripts/make_synthetic.py --n 5000 --len 200    # a "ChIP-seq scale" set
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "sequences_sample.fasta"

BASES = "ACGT"
# Background composition (every base must be > 0 so the null model is valid).
BG = {"A": 0.30, "C": 0.20, "G": 0.20, "T": 0.30}

# The motif we plant. TGACGTCA is the palindromic CRE / AP-1-like consensus --
# a real transcription-factor binding site, used here purely as a known target.
PLANTED_MOTIF = "TGACGTCA"


def draw_background_base(rng):
    """Sample one base from the background composition BG (inverse-CDF)."""
    r = rng.random()
    acc = 0.0
    for b in BASES:
        acc += BG[b]
        if r < acc:
            return b
    return "T"  # numerical safety net (acc just under 1.0)


def mutated_motif(rng, mut_rate):
    """Return PLANTED_MOTIF with each base independently mutated to a different
    random base with probability `mut_rate` -- this degeneracy is what makes
    motif finding non-trivial (the planted sites are not identical)."""
    out = []
    for c in PLANTED_MOTIF:
        if rng.random() < mut_rate:
            out.append(rng.choice([x for x in BASES if x != c]))
        else:
            out.append(c)
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic planted-motif FASTA.")
    ap.add_argument("--n", type=int, default=12, help="number of sequences")
    ap.add_argument("--len", type=int, default=60, help="length of each sequence (bp)")
    ap.add_argument("--mut", type=float, default=0.10, help="per-base motif mutation rate")
    ap.add_argument("--seed", type=int, default=20240517, help="RNG seed (determinism)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    if args.len < len(PLANTED_MOTIF):
        ap.error(f"--len must be >= motif width {len(PLANTED_MOTIF)}")

    rng = random.Random(args.seed)
    records = []
    for i in range(args.n):
        # Random-background sequence, then overwrite a random window with a
        # (mutated) copy of the planted motif.
        seq = [draw_background_base(rng) for _ in range(args.len)]
        offset = rng.randint(0, args.len - len(PLANTED_MOTIF))
        site = mutated_motif(rng, args.mut)
        for k, c in enumerate(site):
            seq[offset + k] = c
        records.append((i, offset, "".join(seq)))

    # Write FASTA: the header records the planted offset (truth) for the learner.
    lines = []
    for i, offset, s in records:
        lines.append(f">seq_{i}  synthetic planted_at={offset} motif={PLANTED_MOTIF}")
        for j in range(0, len(s), 60):       # wrap at 60 cols (FASTA convention)
            lines.append(s[j:j + 60])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(n={args.n}, len={args.len}, motif={PLANTED_MOTIF}, mut={args.mut}; "
          f"SYNTHETIC, seed={args.seed})")


if __name__ == "__main__":
    main()
