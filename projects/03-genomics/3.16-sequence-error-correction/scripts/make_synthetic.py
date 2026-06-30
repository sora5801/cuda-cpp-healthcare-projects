#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic sequencing dataset
# ---------------------------------------------------------------------------
# Project 3.16 : Sequence Error Correction  (k-mer spectrum / trusted-k-mer)
#
# WHY SYNTHETIC  (CLAUDE.md sec 8)
#   Real reads come from sequencers (Illumina/ONT/PacBio) over real genomes and
#   need download + credentials (see scripts/download_data.* and data/README.md).
#   To keep the demo OFFLINE, REPRODUCIBLE, and INTERPRETABLE, we synthesize the
#   whole experiment and KEEP THE GROUND TRUTH so the program can report how many
#   errors correction removed:
#     1. Build a random "genome" string over {A,C,G,T} (fixed RNG seed).
#     2. Sample many overlapping reads (the "truth" reads -- error free).
#     3. Make a noisy copy of each read by flipping a small fraction of bases
#        (the substitution-error model). These are the "raw" reads the corrector
#        sees. Coverage is high enough that every true k-mer recurs many times
#        (trusted) while error k-mers are rare (untrusted) -- exactly the signal
#        the k-mer spectrum exploits.
#
#   A fixed seed makes the output byte-for-byte reproducible, so the committed
#   sample (and demo/expected_output.txt) is stable.
#
# OUTPUT FORMAT (data/README.md is canonical):
#   line 1            : "<n> <has_truth>"    (has_truth = 1 here)
#   then per read (n times):
#     raw read line   : the noisy observed read (A/C/G/T)
#     truth read line : the error-free read
#   '#'-comment lines and blank lines are allowed and ignored by the loader.
#
# USAGE
#   python scripts/make_synthetic.py                       # default small sample
#   python scripts/make_synthetic.py --reads 100000        # a bigger set
# ===========================================================================
import argparse
import random
from pathlib import Path

BASES = "ACGT"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "reads_sample.txt"


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic error-correction dataset.")
    ap.add_argument("--genome-len", type=int, default=400, help="length of the synthetic genome")
    ap.add_argument("--reads", type=int, default=120, help="number of reads to sample")
    ap.add_argument("--read-len", type=int, default=60, help="length of each read")
    ap.add_argument("--error-rate", type=float, default=0.02,
                    help="per-base substitution error probability")
    ap.add_argument("--seed", type=int, default=20260628, help="RNG seed (determinism)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # 1. A random genome. (Real genomes have structure; a random string is the
    #    simplest substrate that still demonstrates the spectrum cleanly.)
    genome = "".join(rng.choice(BASES) for _ in range(args.genome_len))

    # 2/3. Sample reads at random start positions, then add substitution noise.
    #      coverage ~= reads * read_len / genome_len; with the defaults that is
    #      120*60/400 = 18x, so true k-mers are seen ~18x (well above T=3).
    raw_reads, truth_reads = [], []
    max_start = args.genome_len - args.read_len
    for _ in range(args.reads):
        start = rng.randint(0, max_start)
        truth = genome[start:start + args.read_len]
        # Noisy copy: flip each base to a DIFFERENT base with prob error_rate.
        noisy = []
        for b in truth:
            if rng.random() < args.error_rate:
                alt = rng.choice([x for x in BASES if x != b])  # a real substitution
                noisy.append(alt)
            else:
                noisy.append(b)
        raw_reads.append("".join(noisy))
        truth_reads.append(truth)

    lines = [f"# SYNTHETIC sequencing reads -- Project 3.16 (NOT real patient data)",
             f"# genome_len={args.genome_len} reads={args.reads} read_len={args.read_len} "
             f"error_rate={args.error_rate} seed={args.seed}",
             f"{args.reads} 1"]
    for raw, truth in zip(raw_reads, truth_reads):
        lines.append(raw)
        lines.append(truth)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(reads={args.reads}, read_len={args.read_len}, SYNTHETIC, seed={args.seed})")


if __name__ == "__main__":
    main()
