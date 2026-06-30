#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic DNA-read sample
# ---------------------------------------------------------------------------
# Project 3.6 -- k-mer Counting & Minimiser Sketching
#
# WHY THIS EXISTS
#   Real WGS read sets (NA12878, GenomeTrakr, ...) are huge and/or require SRA
#   download (see scripts/download_data.*). For an offline, instantly-runnable
#   demo we deterministically synthesize a TINY two-set DNA sample with a KNOWN
#   structure so the output is interpretable and verifiable. Synthetic data is
#   always LABELED synthetic (CLAUDE.md section 8).
#
# WHAT WE BUILD (engineered so the result recovers a known answer)
#   * A shared "genome" string G over {A,C,G,T} from a fixed PRNG seed.
#   * A planted MOTIF inserted into several of set A's reads, so its canonical
#     k-mer is the clear top-count k-mer (the demo prints it).
#   * Set A = reads sampled from the FIRST part of G (+ the motif).
#   * Set B = reads sampled from a window of G that OVERLAPS A's window by a
#     controlled fraction, so the MinHash Jaccard estimate is non-trivial and
#     reflects the true sequence overlap (the demo prints it).
#
# OUTPUT FORMAT (parsed by src/reference_cpu.cpp::load_reads)
#   line 1 : "k w s"   (k-mer length, minimiser window in k-mers, sketch size)
#   ">A" then set-A reads (one per line), ">B" then set-B reads.
#
# USAGE
#   python scripts/make_synthetic.py            # writes data/sample/kmer_sample.txt
#   python scripts/make_synthetic.py --seed 7   # a different (still deterministic) sample
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "kmer_sample.txt"

BASES = "ACGT"


def random_dna(rng, n):
    """A length-n DNA string from the given seeded PRNG (deterministic)."""
    return "".join(rng.choice(BASES) for _ in range(n))


def sample_reads(rng, genome, start, span, n_reads, read_len):
    """Sample n_reads substrings of length read_len from genome[start:start+span]."""
    reads = []
    hi = start + span - read_len
    for _ in range(n_reads):
        p = rng.randint(start, hi)
        reads.append(genome[p:p + read_len])
    return reads


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic k-mer/minimiser sample.")
    ap.add_argument("--k", type=int, default=11, help="k-mer length (<=31)")
    ap.add_argument("--w", type=int, default=5, help="minimiser window (in k-mers)")
    ap.add_argument("--s", type=int, default=16, help="MinHash sketch size")
    ap.add_argument("--seed", type=int, default=2026, help="PRNG seed (determinism)")
    ap.add_argument("--genome", type=int, default=400, help="reference genome length")
    ap.add_argument("--reads", type=int, default=12, help="reads per set")
    ap.add_argument("--readlen", type=int, default=40, help="read length")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    genome = random_dna(rng, args.genome)

    # A distinctive planted motif, length k, that is NOT all-symmetric so its
    # canonical form is itself; inserting it repeatedly makes it the top k-mer.
    motif = ("ACGTACGTACGTACGT")[:args.k]
    if len(motif) < args.k:                              # pad if k > 16
        motif = (motif * ((args.k // len(motif)) + 1))[:args.k]

    # Set A: reads from the first half of the genome; plant the motif into the
    # first few reads so it occurs many times.
    half = args.genome // 2
    a_reads = sample_reads(rng, genome, 0, half, args.reads, args.readlen)
    for i in range(min(5, len(a_reads))):               # plant motif near read start
        r = a_reads[i]
        a_reads[i] = motif + r[len(motif):]

    # Set B: reads from a window that OVERLAPS A's window by ~50% (start at
    # quarter genome, span half). The shared region drives the Jaccard estimate.
    quarter = args.genome // 4
    b_reads = sample_reads(rng, genome, quarter, half, args.reads, args.readlen)

    lines = [f"{args.k} {args.w} {args.s}", ">A"]
    lines += a_reads
    lines += [">B"]
    lines += b_reads

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(k={args.k} w={args.w} s={args.s}, seed={args.seed}; SYNTHETIC DNA)")


if __name__ == "__main__":
    main()
