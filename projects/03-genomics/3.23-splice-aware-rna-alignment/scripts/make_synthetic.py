#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic splice-aware sample
# ---------------------------------------------------------------------------
# Project 3.23 : Splice-Aware RNA Alignment   (REDUCED-SCOPE teaching version)
#
# WHY THIS EXISTS
#   Real RNA-seq data (ENCODE / GTEx / SRA) is large and sometimes credential-
#   gated; scripts/download_data.* point you there. For an OFFLINE, byte-stable
#   demo we generate a TINY, clearly-SYNTHETIC "gene model" with known exons and
#   canonical GT-AG introns, plus reads that we KNOW cross specific junctions.
#   Because the answer is engineered in, the demo's output is interpretable: a
#   junction read should produce a CIGAR like "<exon>M<intron>N<exon>M".
#
#   The file layout the C++ loader expects:
#       line 1            : the reference gene model (genomic order: exons+introns)
#       subsequent lines  : one read per line ('#'-comment and blank lines allowed)
#   Bases are ACGT (the loader also accepts RNA 'U' as 'T').
#
#   DETERMINISTIC: a fixed RNG seed so re-running reproduces the same sample and
#   the committed expected_output.txt stays valid.
#
# USAGE
#   python scripts/make_synthetic.py                 # writes data/sample/reads_sample.txt
#   python scripts/make_synthetic.py --seed 7        # a different synthetic batch
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "reads_sample.txt"

BASES = "ACGT"


def rand_seq(rng, length):
    """A random ACGT string of the given length (exon/intron body filler)."""
    return "".join(rng.choice(BASES) for _ in range(length))


def make_intron(rng, length):
    """A synthetic intron that obeys the canonical GT-AG rule: it STARTS with the
    GT donor dinucleotide and ENDS with the AG acceptor. The interior filler is
    drawn ONLY from {A, C} so it can never contain another 'GT' donor or 'AG'
    acceptor -- that keeps the TRUE splice boundaries the only canonical sites,
    so each junction read has one unambiguous best CIGAR (clean demo output).
    Real introns of course have random interiors; the kernel handles those too,
    it just may report an equally-scoring alternative boundary."""
    assert length >= 4
    body = "".join(rng.choice("AC") for _ in range(length - 4))
    return "GT" + body + "AG"


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic splice sample.")
    ap.add_argument("--seed", type=int, default=2026, help="RNG seed (determinism)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # ---- Build a 3-exon gene model: E1 - I1 - E2 - I2 - E3 ------------------
    # Exon lengths are chosen so junction reads land cleanly; intron lengths sit
    # comfortably inside MAX_INTRON (=64 in reference_cpu.h) so the N move finds
    # them, and above MIN_INTRON (=4) so they are called introns not gaps.
    e1 = rand_seq(rng, 30)
    e2 = rand_seq(rng, 24)
    e3 = rand_seq(rng, 30)
    i1 = make_intron(rng, 40)     # canonical GT...AG
    i2 = make_intron(rng, 48)     # canonical GT...AG

    reference = e1 + i1 + e2 + i2 + e3        # genomic order (introns present)
    mrna = e1 + e2 + e3                        # mature transcript (introns spliced out)

    # ---- Generate reads from the mature mRNA -------------------------------
    # The mRNA has NO introns; a read taken from it that straddles an exon-exon
    # boundary must, when mapped back to the genomic reference, SKIP the intron.
    # We craft reads at known offsets so the expected CIGAR is predictable.
    reads = []

    # (a) Exon-internal reads (no junction -> pure "M" CIGAR).
    reads.append(("exon1_internal", e1[5:5 + 20]))
    reads.append(("exon3_internal", e3[3:3 + 22]))

    # (b) Reads spanning the E1|E2 junction (must skip intron I1).
    reads.append(("junc_E1E2_a", e1[-12:] + e2[:12]))
    reads.append(("junc_E1E2_b", e1[-8:] + e2[:16]))

    # (c) Reads spanning the E2|E3 junction (must skip intron I2).
    reads.append(("junc_E2E3_a", e2[-10:] + e3[:14]))

    # (d) A read spanning BOTH junctions (skips I1 then I2): the hard case.
    reads.append(("junc_double", e1[-6:] + e2 + e3[:6]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# SYNTHETIC splice-aware RNA alignment sample (project 3.23).")
    lines.append("# Line 1 below = reference GENE MODEL in genomic order:")
    lines.append("#   E1(30) GT-intron(40)-AG  E2(24)  GT-intron(48)-AG  E3(30).")
    lines.append("# Remaining lines = reads from the mature mRNA (introns spliced).")
    lines.append("# NOT REAL DATA -- engineered so junction reads yield M..N..M CIGARs.")
    lines.append(reference)
    for name, seq in reads:
        lines.append(f"# read: {name}")
        lines.append(seq)

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {out}")
    print(f"[make_synthetic]   reference N={len(reference)}  mRNA={len(mrna)}  reads={len(reads)}  (SYNTHETIC, seed={args.seed})")
    print(f"[make_synthetic]   introns: I1 len={len(i1)} (GT..AG), I2 len={len(i2)} (GT..AG)")


if __name__ == "__main__":
    main()
