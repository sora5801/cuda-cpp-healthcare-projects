#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic DNA text sample
# ---------------------------------------------------------------------------
# Project 3.27 -- Suffix Array / BWT / FM-Index Construction
#
# WHY THIS EXISTS
#   The real targets for BWT/FM-index construction are whole genomes (GRCh38 is
#   ~3 Gb) and read collections -- far too large to commit, and several are
#   licensed/credentialed (see data/README.md and scripts/download_data.*). So
#   the committed demo runs on a TINY, clearly-SYNTHETIC DNA string instead.
#
#   We engineer the sample so the result is *interpretable* (PATTERNS.md section 6):
#     * The text is short DNA over the alphabet {A,C,G,T}.
#     * It contains a planted, REPEATED motif ("ACGT") several times, so the
#       FM-index backward-search demo has a known, non-trivial hit count we can
#       eyeball: counting "ACG" should recover every planted occurrence.
#   The loader (src/main.cu) appends the '$' sentinel itself, so this file holds
#   ONLY the {A,C,G,T} characters -- no sentinel, no header, one line.
#
#   Synthetic data is LABELED synthetic everywhere (CLAUDE.md section 8). Nothing
#   here is a real genome or patient sequence.
#
# USAGE
#   python scripts/make_synthetic.py                 # writes data/sample/dna_sample.txt
#   python scripts/make_synthetic.py --n 200 --seed 7
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "dna_sample.txt"

ALPHABET = "ACGT"
MOTIF = "ACGT"                                          # planted, repeated motif


def make_text(n: int, seed: int) -> str:
    """Build a length-n DNA string with the MOTIF planted at regular spots.

    The background is uniform random over {A,C,G,T}; every ~10 bases we overwrite
    a window with MOTIF so the pattern "ACG" recurs a known number of times. The
    fixed seed makes the output deterministic (so expected_output.txt is stable).
    """
    rng = random.Random(seed)
    chars = [rng.choice(ALPHABET) for _ in range(n)]
    # Plant the motif at every multiple of 10 that fits, overwriting in place.
    for start in range(0, n - len(MOTIF) + 1, 10):
        for j, c in enumerate(MOTIF):
            chars[start + j] = c
    return "".join(chars)


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic DNA sample for project 3.27.")
    ap.add_argument("--n", type=int, default=60, help="text length (number of bases)")
    ap.add_argument("--seed", type=int, default=20260628, help="RNG seed (determinism)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    text = make_text(args.n, args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # One line of DNA, newline-terminated. The C++ loader trims whitespace and
    # appends the '$' sentinel; it accepts only A/C/G/T (case-insensitive).
    Path(args.out).write_text(text + "\n", encoding="utf-8")
    occ = sum(1 for i in range(len(text) - 2) if text[i:i + 3] == "ACG")
    print(f"[make_synthetic] wrote {args.out}  (n={len(text)} bases; SYNTHETIC DNA)")
    print(f"[make_synthetic] planted motif '{MOTIF}'; substring 'ACG' occurs {occ} time(s)")


if __name__ == "__main__":
    main()
