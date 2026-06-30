#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic SV sample dataset
# ---------------------------------------------------------------------------
# Project 3.21 : Structural Variant (SV) Calling   (REDUCED-SCOPE teaching version)
#
# WHY THIS EXISTS
#   Real SV benchmarks (GiaB HG002, PacBio sv-benchmark) are large BAM/VCF files
#   that cannot be committed (size + provenance), so we deterministically generate
#   a clearly-SYNTHETIC stand-in matching the loader's text layout. The result is
#   small enough to read by eye and contains a KNOWN planted deletion so the demo
#   can report "did the caller recover it?".
#
# WHAT WE BUILD  (a single biallelic DELETION on one reference window)
#   1. A random reference window of length REF_LEN (bases A/C/G/T, fixed seed).
#   2. A planted heterozygous deletion: bases [TRUE_BP, TRUE_BP+TRUE_LEN) are
#      "deleted" in the variant haplotype.
#   3. SUPPORTING reads (variant haplotype): each carries the SV_FLANK reference
#      bases ending exactly at the breakpoint, plus a NOISY raw breakpoint guess
#      (jittered by a few bp, as a real aligner's split-read estimate would be).
#      The realignment step (banded SW) is what removes that jitter.
#   4. A few REFERENCE / NOISE reads whose guesses scatter elsewhere and whose
#      flanks do NOT realign cleanly to the breakpoint -- these should fall below
#      the support floor and NOT be called (teaching the noise floor).
#
#   Because supporting reads all carry the TRUE breakpoint flank, banded SW pulls
#   every jittered guess back to TRUE_BP, they pile into one histogram bin, clear
#   MIN_SUPPORT, and produce exactly one DEL call at TRUE_BP. Deterministic seed
#   => identical file => stable expected_output.txt.
#
# OUTPUT FORMAT (parsed by reference_cpu.cpp::load_dataset; see data/README.md):
#   REF   <reference_sequence>
#   TRUTH <true_breakpoint> <true_deletion_length>
#   N     <num_reads>
#   <raw_guess> <del_len> <flank_sequence>      (one line per read; flank = SV_FLANK bases)
#
# USAGE
#   python scripts/make_synthetic.py                 # writes data/sample/sv_sample.txt
#   python scripts/make_synthetic.py --reads 200000  # bigger synthetic problem
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "sv_sample.txt"

# These MUST match the constants in src/sv.h (SV_FLANK) so the loader accepts the
# flanks and the realignment window makes sense.
SV_FLANK = 24          # bases of read flank carried left of the breakpoint
BASES = "ACGT"


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic SV sample (one planted deletion).")
    ap.add_argument("--ref-len", type=int, default=240, help="reference window length (bp)")
    ap.add_argument("--true-bp", type=int, default=120, help="planted left breakpoint (ref coord)")
    ap.add_argument("--true-len", type=int, default=50, help="planted deletion length (bp)")
    ap.add_argument("--support", type=int, default=18, help="number of supporting (variant) reads")
    ap.add_argument("--noise", type=int, default=6, help="number of scattered noise reads")
    ap.add_argument("--jitter", type=int, default=6, help="+/- bp jitter on the raw breakpoint guess")
    ap.add_argument("--seed", type=int, default=3, help="RNG seed (determinism)")
    ap.add_argument("--reads", type=int, default=0,
                    help="if >0, scale support up to this many total reads (large synthetic run)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    ref_len = args.ref_len
    true_bp = args.true_bp
    true_len = args.true_len
    assert true_bp >= SV_FLANK, "breakpoint must leave room for a full left flank"
    assert true_bp < ref_len, "breakpoint must lie inside the reference"

    # (1) Random reference window.
    ref = "".join(rng.choice(BASES) for _ in range(ref_len))

    # The TRUE left flank: the SV_FLANK reference bases ending exactly at true_bp.
    true_flank = ref[true_bp - SV_FLANK: true_bp]

    support = args.support
    if args.reads > 0:
        # Scale up supporting reads for a large synthetic stress run (keeps the
        # same planted SV; the call is unchanged, only the histogram mass grows).
        support = max(args.support, args.reads - args.noise)

    reads = []  # each: (raw_guess, del_len, flank_sequence)

    # (3) Supporting reads: true flank, true length, jittered raw breakpoint guess.
    for _ in range(support):
        jitter = rng.randint(-args.jitter, args.jitter)
        guess = true_bp + jitter
        # The deletion-length estimate also jitters slightly, as a real read would.
        dlen = true_len + rng.randint(-2, 2)
        reads.append((guess, dlen, true_flank))

    # (4) Noise / reference reads: a random flank (won't realign to true_bp well)
    #     and a guess scattered far from the true breakpoint, with low coherence so
    #     they spread across bins and stay below MIN_SUPPORT.
    for _ in range(args.noise):
        guess = rng.randint(SV_FLANK + 1, ref_len - 1)
        dlen = rng.randint(50, 300)
        flank = "".join(rng.choice(BASES) for _ in range(SV_FLANK))
        reads.append((guess, dlen, flank))

    # Shuffle so supporting and noise reads interleave (the order must not matter:
    # integer atomic voting is order-independent -- that's the whole point).
    rng.shuffle(reads)

    lines = [f"REF {ref}",
             f"TRUTH {true_bp} {true_len}",
             f"N {len(reads)}"]
    for (guess, dlen, flank) in reads:
        lines.append(f"{guess} {dlen} {flank}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(ref_len={ref_len}, true_bp={true_bp}, true_len={true_len}, "
          f"reads={len(reads)} [{support} support + {args.noise} noise]; SYNTHETIC)")


if __name__ == "__main__":
    main()
