#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic MSA sample dataset
# ---------------------------------------------------------------------------
# Project 3.8 : Multiple Sequence Alignment (MSA)
#
# WHY THIS EXISTS
#   Real MSA benchmarks (BAliBASE, HomFam, Pfam seeds) are large and/or have
#   licenses that complicate redistribution. For a self-contained, offline,
#   deterministic demo we generate a small, clearly-SYNTHETIC family of DNA
#   sequences that share a common ancestral motif and then diverge by point
#   mutations and small indels -- exactly the situation MSA is built to recover.
#   Everything here is SYNTHETIC and labeled as such (CLAUDE.md §8).
#
# WHAT THE DATA LOOKS LIKE
#   We start from one ancestral sequence (a fixed conserved "core" motif flanked
#   by short random arms). Each output sequence is a mutated descendant: with a
#   fixed PRNG seed we apply a few substitutions and occasionally delete/insert a
#   base. Because all descendants share the conserved core, a correct MSA lines
#   that core up into starred (fully conserved) columns -- a result you can eyeball
#   in the demo output. Determinism: a fixed seed => byte-identical file every run
#   => a stable expected_output.txt.
#
#   Output format is multi-FASTA (what src/reference_cpu.cpp::load_fasta reads):
#     >seq0
#     ACGT...
#     >seq1
#     ACGT...
#
# USAGE
#   python scripts/make_synthetic.py                       # default tiny sample
#   python scripts/make_synthetic.py --n 12 --sub 0.10     # bigger / noisier
#   python scripts/make_synthetic.py --out other.fasta
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "sequences_sample.fasta"

BASES = "ACGT"

# A fixed, human-recognisable conserved core motif (the "gene" all sequences
# descend from). Kept short so the committed sample is tiny.
CORE = "ACGTACGTTGCAACGT"


def mutate(seq, rng, sub_rate, indel_rate):
    """Return a mutated copy of `seq`: per-base substitution with prob sub_rate,
    and per-position single-base insertion/deletion with prob indel_rate. This is
    a crude but adequate model of sequence divergence for a teaching sample."""
    out = []
    for ch in seq:
        r = rng.random()
        if r < indel_rate * 0.5:
            continue                                   # deletion: drop this base
        if r < indel_rate:
            out.append(rng.choice(BASES))              # insertion before this base
        if rng.random() < sub_rate:
            # substitute to a DIFFERENT base so the mutation is visible
            out.append(rng.choice([b for b in BASES if b != ch]))
        else:
            out.append(ch)
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic MSA sample (multi-FASTA).")
    ap.add_argument("--n", type=int, default=6, help="number of sequences")
    ap.add_argument("--sub", type=float, default=0.08, help="per-base substitution rate")
    ap.add_argument("--indel", type=float, default=0.06, help="per-position indel rate")
    ap.add_argument("--arm", type=int, default=4, help="random flank length each side")
    ap.add_argument("--seed", type=int, default=20260628, help="PRNG seed (determinism)")
    ap.add_argument("--out", default=str(OUT), help="output FASTA path")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Ancestral sequence = random left arm + conserved CORE + random right arm.
    def arm():
        return "".join(rng.choice(BASES) for _ in range(args.arm))
    ancestor = arm() + CORE + arm()

    records = []
    for i in range(args.n):
        # seq0 is the (almost) ancestral sequence so the center-star pick is
        # interpretable; the rest diverge more.
        if i == 0:
            seq = mutate(ancestor, rng, args.sub * 0.3, args.indel * 0.3)
        else:
            seq = mutate(ancestor, rng, args.sub, args.indel)
        if not seq:                                    # never emit an empty record
            seq = CORE
        records.append((f"seq{i}", seq))

    lines = []
    for name, seq in records:
        lines.append(f">{name}")
        lines.append(seq)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  (n={args.n}, seed={args.seed}; SYNTHETIC)")
    for name, seq in records:
        print(f"    {name}: {seq}")


if __name__ == "__main__":
    main()
