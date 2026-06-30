#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic BQSR alignment
# ---------------------------------------------------------------------------
# Project 3.25 : Base Quality Score Recalibration (BQSR)
#
# WHAT THIS BUILDS
#   A tiny aligned "tile": one reference string, a known-variant mask, and many
#   fixed-length reads. The reads are copies of the reference with errors injected
#   so that the EMPIRICAL error rate differs from the REPORTED quality -- which is
#   exactly the systematic miscalibration BQSR exists to correct.
#
#   Two teaching effects are engineered in (deterministically, seeded):
#     1. MISCALIBRATION. Every base is *reported* at a single nominal quality
#        (REPORTED_Q, e.g. Q30 ~ 1-in-1000). But we inject errors at a HIGHER true
#        rate (TRUE_ERR), so the empirical quality the table recovers is LOWER than
#        Q30. That gap (reported Q30 -> empirical ~Q20) is the headline result.
#     2. KNOWN-VARIANT MASKING. A handful of reference positions are marked as
#        known variants. At those columns every read carries the alternate allele
#        (a real biological difference, not a machine error). Because BQSR SKIPS
#        known sites, those systematic "mismatches" must NOT inflate the error
#        count -- the demo shows the error tally stays at the injected machine rate.
#
#   Output is whitespace text (format documented in data/README.md): a REF line, a
#   KNOWN line, a READS header, then one line per read.
#
# USAGE
#   python scripts/make_synthetic.py                 # default tiny committed sample
#   python scripts/make_synthetic.py --reads 5000    # bigger synthetic set
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "bqsr_sample.txt"

BASES = "ACGT"

# Reads are fixed length L (must be <= MAX_CYCLE=16 in bqsr.h). The reference is a
# bit longer than a read so reads can start at a few offsets.
READ_LEN = 12
REF_LEN = 24

# Reported quality stamped on every base, and the (higher) TRUE machine error rate
# we actually inject. P(Q30)=1e-3; injecting ~1.2e-2 makes empirical Q ~ 19-20.
REPORTED_Q = 30
TRUE_ERR = 0.012      # ~1.2% real errors -> empirical quality well below Q30

# Reference positions we flag as KNOWN variants (0-based). At these columns every
# covering read carries a fixed alternate allele, so the column is all-"mismatch"
# -- which BQSR must mask out rather than count as machine error.
KNOWN_SITES = [7, 16]


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic BQSR alignment.")
    ap.add_argument("--reads", type=int, default=1200, help="number of reads")
    ap.add_argument("--seed", type=int, default=7, help="PRNG seed (determinism)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # ---- the reference: a fixed pseudo-random ACGT string -------------------
    ref = "".join(rng.choice(BASES) for _ in range(REF_LEN))

    # ---- alternate alleles at the known sites (different from the ref base) --
    alt = {}
    for p in KNOWN_SITES:
        choices = [b for b in BASES if b != ref[p]]
        alt[p] = rng.choice(choices)

    # ---- build reads --------------------------------------------------------
    # Each read starts at one of a few offsets so reads tile the reference and the
    # known sites land at varying cycles (positions) across reads.
    max_start = REF_LEN - READ_LEN
    rows = []
    for _ in range(args.reads):
        start = rng.randint(0, max_start)
        bases = []
        for c in range(READ_LEN):
            refp = start + c
            true_base = ref[refp]
            if refp in alt:
                # Known variant: this read genuinely carries the alternate allele.
                called = alt[refp]
            elif rng.random() < TRUE_ERR:
                # Inject a real machine error: a different base than the reference.
                called = rng.choice([b for b in BASES if b != true_base])
            else:
                called = true_base
            bases.append(called)
        quals = [REPORTED_Q] * READ_LEN
        rows.append((start, "".join(bases), quals))

    # ---- write the file -----------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="ascii") as f:
        f.write(f"REF {ref}\n")
        f.write("KNOWN " + " ".join(str(p) for p in KNOWN_SITES) + "\n")
        f.write(f"READS {len(rows)} {READ_LEN}\n")
        for start, seq, quals in rows:
            f.write(f"{start} {seq} " + " ".join(str(q) for q in quals) + "\n")

    print(f"[make_synthetic] wrote {out_path}  (SYNTHETIC)")
    print(f"  reference   : {ref} ({REF_LEN} bp)")
    print(f"  known sites : {KNOWN_SITES}  (alts {alt})")
    print(f"  reads       : {len(rows)} x {READ_LEN} bp, reported Q{REPORTED_Q}, "
          f"true error rate ~{TRUE_ERR:.3f}")


if __name__ == "__main__":
    main()
