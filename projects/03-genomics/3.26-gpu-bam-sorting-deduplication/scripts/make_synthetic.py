#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic aligned-read set
# ---------------------------------------------------------------------------
# Project 3.26 : GPU BAM Sorting & Deduplication
#
# WHY THIS EXISTS
#   Real aligned reads live in BAM files that are large and often credentialed
#   (TCGA, ICGC) -- they cannot be redistributed here. So we deterministically
#   generate a clearly-SYNTHETIC stand-in shaped like post-alignment reads, with
#   a KNOWN duplicate structure so the demo's result is interpretable and the
#   GPU-vs-CPU comparison is verifiable. Synthetic data is labeled synthetic
#   everywhere (data/README.md, demo output).
#
# WHAT WE EMBED (so the answer is checkable, PATTERNS.md §6)
#   * Reads are scattered across `--refs` chromosomes and random positions, so
#     coordinate sorting genuinely reorders them.
#   * A controlled fraction of reads are PCR/optical DUPLICATES: we pick a
#     fragment signature (ref, pos, strand, mate_pos) and emit it `copies` times
#     with DIFFERENT base-quality sums, so exactly (copies-1) of each cluster are
#     duplicates of the single highest-quality original. The script prints the
#     exact number of duplicates it planted -- which must equal the demo's count.
#
# OUTPUT (data/README.md format):
#   line 1:  "<n> <num_refs>"
#   n lines: "<ref_id> <pos> <strand> <mate_pos> <base_qual_sum>"
#
# Field ranges are kept inside the bit budgets in src/bam.h:
#   ref_id  < num_refs   pos in [0, 2^24)   strand in {0,1}
#   mate_pos in [0, 2^15)   base_qual_sum >= 0
#
# USAGE
#   python scripts/make_synthetic.py                  # writes data/sample/reads_sample.txt
#   python scripts/make_synthetic.py --n 1048576      # ~1M reads, bigger demo
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "reads_sample.txt"

POS_MAX  = 1 << 24    # 24-bit positions (matches coord_key in bam.h)
MATE_MAX = 1 << 15    # 15-bit mate positions (matches dup_key in bam.h)


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic aligned-read set.")
    ap.add_argument("--n", type=int, default=2000, help="approximate number of reads")
    ap.add_argument("--refs", type=int, default=4, help="number of chromosomes")
    ap.add_argument("--dup-frac", type=float, default=0.25,
                    help="fraction of reads that belong to duplicate clusters")
    ap.add_argument("--seed", type=int, default=326)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    records = []          # list of (ref, pos, strand, mate, qual)
    planted_dups = 0      # number of reads that are non-best copies (ground truth)

    # ---- Duplicate clusters: signature emitted `copies` times -------------
    n_dup_reads_target = int(args.n * args.dup_frac)
    emitted_dup_reads = 0
    while emitted_dup_reads < n_dup_reads_target:
        ref    = rng.randrange(args.refs)
        pos    = rng.randrange(POS_MAX)
        strand = rng.randrange(2)
        mate   = rng.randrange(MATE_MAX)
        copies = rng.randint(2, 5)                    # 2..5 PCR copies of one fragment
        quals  = rng.sample(range(100, 4000), copies)  # DISTINCT qualities -> unique best
        for q in quals:
            records.append((ref, pos, strand, mate, q))
        planted_dups += (copies - 1)                  # all but the best are duplicates
        emitted_dup_reads += copies

    # ---- Unique reads (no duplicates): random, mostly distinct signatures --
    n_unique = max(0, args.n - len(records))
    for _ in range(n_unique):
        ref    = rng.randrange(args.refs)
        pos    = rng.randrange(POS_MAX)
        strand = rng.randrange(2)
        mate   = rng.randrange(MATE_MAX)
        qual   = rng.randint(100, 4000)
        records.append((ref, pos, strand, mate, qual))

    # Shuffle so the input is NOT already coordinate-sorted (the sort has work).
    rng.shuffle(records)
    n = len(records)

    lines = [f"{n} {args.refs}"]
    lines += [f"{r} {p} {s} {m} {q}" for (r, p, s, m, q) in records]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(n={n} reads, refs={args.refs}; ~{planted_dups} planted duplicates; SYNTHETIC)")


if __name__ == "__main__":
    main()
