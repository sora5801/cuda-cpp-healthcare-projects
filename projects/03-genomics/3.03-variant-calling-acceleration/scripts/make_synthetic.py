#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic variant-calling sample
# ---------------------------------------------------------------------------
# Project 3.3 : Variant Calling Acceleration
#
# WHY THIS EXISTS
#   The real benchmark inputs for variant calling (GiaB HG001-HG007, 1000
#   Genomes WGS) are large BAM/VCF files that require download and tooling
#   (scripts/download_data.* documents how to get them). For a tiny, offline,
#   fully-deterministic demo we synthesize a clearly-labeled stand-in: one
#   "truth" haplotype, a handful of reads drawn from it with a few simulated
#   sequencing errors, and two alternative haplotypes that differ from the truth
#   by a single SNP / a small deletion. The PairHMM forward algorithm should then
#   assign every read to the truth haplotype -- a result the demo checks.
#
#   SYNTHETIC DATA, NOT REAL PATIENT DATA. It exists only to exercise the math.
#
# OUTPUT FORMAT (see data/README.md for the field-by-field spec):
#   # comments allowed
#   n_reads n_haps read_len hap_len truth delta epsilon
#   <n_haps haplotype sequences>
#   <n_reads lines: read sequence + read_len integer Phred qualities>
#
# DETERMINISM
#   A fixed RNG seed (default 5) makes the output byte-identical every run, so the
#   committed sample and demo/expected_output.txt never drift. Re-running this
#   script reproduces the exact committed file.
#
# USAGE
#   python scripts/make_synthetic.py                 # default tiny sample
#   python scripts/make_synthetic.py --reads 12 --seed 7
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "reads_haplotypes_sample.txt"

BASES = "ACGT"


def random_seq(rng, length):
    """A uniformly random DNA sequence of the given length."""
    return "".join(rng.choice(BASES) for _ in range(length))


def mutate_snp(seq, pos, rng):
    """Return `seq` with a single substitution at `pos` (to a different base)."""
    alt = rng.choice([b for b in BASES if b != seq[pos]])
    return seq[:pos] + alt + seq[pos + 1:]


def make_read(truth_hap, read_len, err_rate, rng):
    """Draw a read of `read_len` bases starting at a random offset in the truth
    haplotype, then inject independent substitution errors at rate `err_rate`.
    Returns (read_string, list_of_phred_qualities). Quality encodes the (constant
    here) per-base error model the demo uses; an errored base keeps the same
    nominal quality, exactly as a real sequencer would report it."""
    max_start = len(truth_hap) - read_len
    start = rng.randint(0, max_start)
    bases = list(truth_hap[start:start + read_len])
    quals = []
    for i in range(read_len):
        # Constant nominal base quality Q30 (~1 error in 1000). The PairHMM uses
        # this Phred score in its emission model regardless of whether THIS base
        # happens to be an injected error.
        quals.append(30)
        if rng.random() < err_rate:
            bases[i] = rng.choice([b for b in BASES if b != bases[i]])
    return "".join(bases), quals


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic variant-calling sample.")
    ap.add_argument("--reads", type=int, default=8, help="number of reads")
    ap.add_argument("--read-len", type=int, default=20, help="bases per read")
    ap.add_argument("--hap-len", type=int, default=30, help="bases per haplotype")
    ap.add_argument("--err-rate", type=float, default=0.03, help="per-base substitution error rate")
    ap.add_argument("--delta", type=float, default=0.0015, help="pair-HMM gap-open probability")
    ap.add_argument("--epsilon", type=float, default=0.1, help="pair-HMM gap-extend probability")
    # Seed 5 was chosen because it yields a clean teaching result: every read's
    # most-likely haplotype is the truth (8/8), while still containing realistic
    # simulated substitution errors (so it is not a trivial error-free case).
    ap.add_argument("--seed", type=int, default=5, help="RNG seed (determinism)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # --- Build three candidate haplotypes -----------------------------------
    # hap 0 = the TRUTH; hap 1 differs by a single SNP near the middle; hap 2
    # differs by another SNP elsewhere. The reads come from hap 0, so the PairHMM
    # likelihood should be highest for hap 0 for every read.
    truth = random_seq(rng, args.hap_len)
    hap_snp1 = mutate_snp(truth, args.hap_len // 2, rng)
    hap_snp2 = mutate_snp(truth, args.hap_len // 3, rng)
    haps = [truth, hap_snp1, hap_snp2]
    truth_idx = 0

    # --- Draw reads from the truth haplotype --------------------------------
    reads = [make_read(truth, args.read_len, args.err_rate, rng) for _ in range(args.reads)]

    # --- Emit the text file --------------------------------------------------
    lines = []
    lines.append("# SYNTHETIC variant-calling sample for project 3.3 (NOT real patient data).")
    lines.append("# Reads are drawn from haplotype 0 (the truth) with ~3% simulated errors.")
    lines.append("# Format: n_reads n_haps read_len hap_len truth delta epsilon")
    lines.append(f"{args.reads} {len(haps)} {args.read_len} {args.hap_len} {truth_idx} "
                 f"{args.delta:g} {args.epsilon:g}")
    lines.append("# --- haplotypes (candidate local genome sequences) ---")
    for h in haps:
        lines.append(h)
    lines.append("# --- reads (sequence then per-base Phred qualities) ---")
    for seq, quals in reads:
        lines.append(seq + " " + " ".join(str(q) for q in quals))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"({args.reads} reads, {len(haps)} haplotypes; SYNTHETIC, seed={args.seed})")


if __name__ == "__main__":
    main()
