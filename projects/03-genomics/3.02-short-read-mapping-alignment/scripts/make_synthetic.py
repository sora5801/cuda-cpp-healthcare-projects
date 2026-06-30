#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic short-read sample
# ---------------------------------------------------------------------------
# Project 3.2 : Short-Read Mapping / Alignment
#
# WHY THIS EXISTS
#   Real WGS datasets (1000 Genomes, GiaB) are large and/or need registration
#   (see scripts/download_data.*), so they cannot be committed. To keep the demo
#   runnable OFFLINE we generate a tiny, clearly-SYNTHETIC reference + reads with
#   a KNOWN answer baked in, so the result is interpretable and verifiable:
#
#     * The reference is a random ACGT string of length REF_LEN (fixed seed ->
#       reproducible, deterministic file).
#     * Each "mapped" read is copied from the reference at a KNOWN start position,
#       then given a small number of point mutations (substitutions). The
#       aligner should recover that start position; with the leading SEED_K bases
#       kept mutation-free the exact seed is guaranteed to hit.
#     * One read is pure random noise -> its leading k-mer is (almost surely)
#       absent from the reference -> it should map NOWHERE (UNMAPPED). This
#       exercises the "no seed hit" path.
#
#   This is SYNTHETIC data, labelled synthetic everywhere (CLAUDE.md section 8).
#   It is NOT real sequencing data and carries no biological meaning.
#
# FILE FORMAT (consumed by load_problem() in src/reference_cpu.cpp)
#   line 1            : the reference sequence (ACGT)
#   each later line   : one read (ACGT), all reads the same length
#
# USAGE
#   python scripts/make_synthetic.py                 # default tiny sample
#   python scripts/make_synthetic.py --ref-len 400 --n-reads 16
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent           # the project folder
OUT  = ROOT / "data" / "sample" / "reads_sample.txt"

# MUST match SEED_K in src/reference_cpu.h: we keep the first SEED_K bases of
# each mapped read mutation-free so its exact-match seed always hits.
SEED_K = 12
BASES  = "ACGT"


def random_seq(rng: random.Random, length: int) -> str:
    """A length-`length` random ACGT string (each base uniform i.i.d.)."""
    return "".join(rng.choice(BASES) for _ in range(length))


def mutate(rng: random.Random, read: str, n_mut: int, protect: int) -> str:
    """Apply n_mut point substitutions to `read`, never touching the first
    `protect` bases (so the seed k-mer stays exact). Substitutions pick a
    DIFFERENT base so each is a real mismatch."""
    chars = list(read)
    positions = list(range(protect, len(chars)))
    rng.shuffle(positions)
    for p in positions[:n_mut]:
        alt = [b for b in BASES if b != chars[p]]
        chars[p] = rng.choice(alt)
    return "".join(chars)


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic short-read sample.")
    ap.add_argument("--ref-len",  type=int, default=240, help="reference length (bases)")
    ap.add_argument("--read-len", type=int, default=40,  help="read length (bases)")
    ap.add_argument("--n-reads",  type=int, default=10,  help="number of MAPPED reads")
    ap.add_argument("--max-mut",  type=int, default=3,   help="max substitutions per read")
    ap.add_argument("--seed",     type=int, default=20260630, help="PRNG seed (reproducible)")
    ap.add_argument("--out",      default=str(OUT), help="output path")
    args = ap.parse_args()

    if args.read_len < SEED_K:
        raise SystemExit(f"read-len ({args.read_len}) must be >= SEED_K ({SEED_K})")
    if args.ref_len < args.read_len:
        raise SystemExit("ref-len must be >= read-len")

    rng = random.Random(args.seed)                      # fixed seed => deterministic file
    ref = random_seq(rng, args.ref_len)

    reads = []
    # Spread the mapped reads' true start positions across the reference so the
    # demo output shows a spread of mapping positions (not all clustered).
    span = args.ref_len - args.read_len                 # last valid start offset
    for i in range(args.n_reads):
        start = (i * span) // max(1, args.n_reads - 1)  # evenly spaced starts
        start = min(start, span)
        window = ref[start:start + args.read_len]
        n_mut = i % (args.max_mut + 1)                  # 0,1,2,3,0,1,... mutations
        reads.append(mutate(rng, window, n_mut, protect=SEED_K))

    # One deliberately UNMAPPABLE read: random noise whose leading k-mer is
    # (with overwhelming probability) not present in the reference.
    reads.append(random_seq(rng, args.read_len))

    lines = [ref] + reads
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {out_path}")
    print(f"  reference length = {args.ref_len}, read length = {args.read_len}")
    print(f"  {args.n_reads} mapped reads + 1 unmappable read (SYNTHETIC)")


if __name__ == "__main__":
    main()
