#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic fingerprint dataset
# ---------------------------------------------------------------------------
# Project 1.12 : Molecular Fingerprint Similarity Search
#
# WHY SYNTHETIC
#   Real fingerprints come from RDKit Morgan/ECFP generation over ChEMBL/ZINC
#   (see scripts/download_data.* and data/README.md). To keep the demo offline
#   and reproducible we generate a clearly-SYNTHETIC set whose Tanimoto scores
#   span a wide range, so the top-K result is interesting and deterministic.
#
#   Method: pick a random query bit-vector, then build each library fingerprint
#   by flipping a controlled fraction of the query's bits (small fraction ->
#   very similar; large fraction -> dissimilar). A fixed RNG seed makes the
#   output byte-for-byte reproducible (so demo/expected_output.txt is stable).
#
# OUTPUT FORMAT (data/README.md):
#   line 1 : "<n> <FP_WORDS>"
#   line 2 : query as FP_WORDS space-separated 16-hex-digit 64-bit words
#   next n : each library fingerprint, same encoding
#
# USAGE
#   python scripts/make_synthetic.py                 # default n=64
#   python scripts/make_synthetic.py --n 1000000     # a "library scale" set
# ===========================================================================
import argparse
import random
from pathlib import Path

FP_WORDS = 32                 # MUST match src/reference_cpu.h
FP_BITS = FP_WORDS * 64       # 2048
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "fingerprints_sample.txt"


def bits_to_words(bits):
    """Pack a list of FP_BITS 0/1 ints into FP_WORDS 64-bit words (bit b of
    word w is bit index w*64 + b)."""
    words = []
    for w in range(FP_WORDS):
        val = 0
        for b in range(64):
            if bits[w * 64 + b]:
                val |= (1 << b)
        words.append(val)
    return words


def fmt(words):
    return " ".join(f"{w:016x}" for w in words)


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic Tanimoto dataset.")
    ap.add_argument("--n", type=int, default=64, help="number of library fingerprints")
    ap.add_argument("--density", type=float, default=0.25, help="fraction of query bits set")
    ap.add_argument("--seed", type=int, default=7, help="RNG seed (determinism)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    # Query: each bit set with probability `density` (ECFP fingerprints are sparse).
    query = [1 if rng.random() < args.density else 0 for _ in range(FP_BITS)]

    lib = []
    for i in range(args.n):
        # Flip fraction f of the query's bits: f sweeps 0.05 -> 0.65 across i,
        # giving a spread of similarities (near-duplicates down to unrelated).
        f = 0.05 + 0.60 * (i / max(1, args.n - 1))
        fp = query[:]
        for b in range(FP_BITS):
            if rng.random() < f:
                fp[b] ^= 1
        lib.append(fp)
    rng.shuffle(lib)   # so the best hit is not trivially index 0

    lines = [f"{args.n} {FP_WORDS}", fmt(bits_to_words(query))]
    lines += [fmt(bits_to_words(fp)) for fp in lib]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  (n={args.n}, {FP_BITS}-bit; SYNTHETIC, seed={args.seed})")


if __name__ == "__main__":
    main()
