#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic reads sample (FASTA)
# ---------------------------------------------------------------------------
# Project 3.5 : De Novo Genome Assembly  (all-vs-all read-overlap stage)
#
# WHY THIS EXISTS
#   Real long-read datasets (PacBio HiFi / ONT from SRA, T2T-CHM13, GenomeArk)
#   are large and need downloads/credentials (see scripts/download_data.*). To
#   keep the demo runnable OFFLINE with a KNOWN answer, we generate a tiny,
#   clearly-SYNTHETIC FASTA: a fixed pseudo-genome, tiled by overlapping reads.
#   The sequence is arbitrary -- it has NO biological meaning and is labelled
#   synthetic everywhere (CLAUDE.md sec.8).
#
# THE CONSTRUCTION (must match make_synthetic_reads() in src/main.cu exactly,
# so running on this file reproduces the built-in fallback bit-for-bit):
#   * GENOME : a fixed 120-base string over {A,C,G,T}.
#   * reads  : every `--read-len`-base window, sliding by `--step` bases. With
#     the defaults (len=60, step=12) that is 6 reads covering [0,60),[12,72),
#     ... ,[60,120). Consecutive reads overlap by 48 bases -> they SHARE
#     minimizers; the expected overlap graph is a single chain 0-1-2-3-4-5
#     (one contig), which the demo recovers.
#   * To make it look more like real data you can inject point "errors" with
#     --error-rate (a fixed RNG seed keeps it deterministic); the demo's default
#     uses 0 so the committed expected_output.txt is stable.
#
# USAGE
#   python scripts/make_synthetic.py                 # writes data/sample/reads_sample.fasta
#   python scripts/make_synthetic.py --read-len 80 --step 8
#   python scripts/make_synthetic.py --genome-len 5000 --read-len 500 --step 100
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "reads_sample.fasta"

# The fixed 120-base pseudo-genome used by the built-in fallback in main.cu.
# Keep these two in lockstep: if you change one, change the other (and rebuild,
# then regenerate expected_output.txt). SYNTHETIC -- not from any organism.
GENOME_120 = (
    "ACGTTGCAAGCTAGGCATCGATCGGATCCAACGTAGCTAGCATGCATGCTAGCTAGGCAT"
    "CGATCGATTACGGCATCCAGTACGTAGCATCGATCGTAGCTAGCATCGGATCCAACGTAG"
)

BASES = "ACGT"


def make_genome(length: int, seed: int) -> str:
    """Return GENOME_120 when length==120 (the demo default), else a random one."""
    if length == 120:
        return GENOME_120
    rng = random.Random(seed)
    return "".join(rng.choice(BASES) for _ in range(length))


def mutate(seq: str, error_rate: float, rng: random.Random) -> str:
    """Apply per-base substitution noise (models sequencing error)."""
    if error_rate <= 0.0:
        return seq
    out = []
    for ch in seq:
        if rng.random() < error_rate:
            out.append(rng.choice(BASES))     # random substitution
        else:
            out.append(ch)
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic reads FASTA.")
    ap.add_argument("--genome-len", type=int, default=120, help="pseudo-genome length in bases")
    ap.add_argument("--read-len", type=int, default=60, help="read length in bases")
    ap.add_argument("--step", type=int, default=12, help="start offset between consecutive reads")
    ap.add_argument("--error-rate", type=float, default=0.0, help="per-base substitution rate")
    ap.add_argument("--seed", type=int, default=1, help="RNG seed (deterministic output)")
    ap.add_argument("--out", default=str(OUT), help="output FASTA path")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    genome = make_genome(args.genome_len, args.seed)

    lines = []
    idx = 0
    s = 0
    while s + args.read_len <= len(genome):
        read = mutate(genome[s:s + args.read_len], args.error_rate, rng)
        lines.append(f">read{idx} pos={s} len={args.read_len} SYNTHETIC")
        lines.append(read)
        idx += 1
        s += args.step

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  ({idx} reads, "
          f"genome={len(genome)} bp, read_len={args.read_len}, step={args.step}; SYNTHETIC)")


if __name__ == "__main__":
    main()
