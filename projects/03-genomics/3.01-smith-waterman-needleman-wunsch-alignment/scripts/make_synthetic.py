#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic DNA alignment problem
# ---------------------------------------------------------------------------
# Project 3.01 : Smith-Waterman / Needleman-Wunsch Alignment
#
# Builds two DNA sequences that SHARE a mutated motif, so there is a clear,
# high-scoring LOCAL alignment for Smith-Waterman to find:
#     query  = randPrefix + motif            + randSuffix
#     target = randPrefix + mutate(motif)     + randSuffix
# A fixed RNG seed makes the output reproducible (so expected_output.txt is
# stable). Real data is FASTA from UniProt/NCBI (see download_data.*).
#
# OUTPUT: two lines (query, target) of A/C/G/T -- the format the loader expects.
#
# USAGE
#   python scripts/make_synthetic.py                       # default sizes
#   python scripts/make_synthetic.py --motif 400 --mut 0.2 # harder, longer
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "sequences_sample.txt"
BASES = "ACGT"


def rand_seq(rng, k):
    return "".join(rng.choice(BASES) for _ in range(k))


def mutate(rng, seq, rate):
    out = []
    for c in seq:
        if rng.random() < rate:
            out.append(rng.choice([b for b in BASES if b != c]))  # substitution
        else:
            out.append(c)
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic SW alignment problem.")
    ap.add_argument("--motif", type=int, default=80, help="shared motif length")
    ap.add_argument("--qflank", type=int, default=20, help="random flank length on the query")
    ap.add_argument("--tpre", type=int, default=30, help="random prefix length on the target")
    ap.add_argument("--tsuf", type=int, default=40, help="random suffix length on the target")
    ap.add_argument("--mut", type=float, default=0.12, help="motif mutation rate in the target")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    motif = rand_seq(rng, args.motif)
    query = rand_seq(rng, args.qflank) + motif + rand_seq(rng, args.qflank)
    target = rand_seq(rng, args.tpre) + mutate(rng, motif, args.mut) + rand_seq(rng, args.tsuf)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(query + "\n" + target + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  (M={len(query)}, N={len(target)}; "
          f"motif={args.motif}, mut={args.mut}; SYNTHETIC, seed={args.seed})")


if __name__ == "__main__":
    main()
