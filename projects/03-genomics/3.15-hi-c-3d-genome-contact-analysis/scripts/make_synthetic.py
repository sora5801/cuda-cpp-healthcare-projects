#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic Hi-C contact sample
# ---------------------------------------------------------------------------
# Project 3.15 : Hi-C / 3D Genome Contact Analysis
#
# WHY THIS EXISTS
#   Real Hi-C matrices (4DN, ENCODE, GSE63525) are large and license-gated, so
#   this script deterministically generates a TINY, clearly-SYNTHETIC stand-in
#   that the demo can run offline. The data is engineered so the expected answer
#   is KNOWN, which is what makes the demo verifiable (docs/PATTERNS.md §6):
#
#     * TAD STRUCTURE: the genome is split into BLOCKS (domains). Pairs inside a
#       block contact each other strongly; pairs across a block border contact
#       weakly. So the insulation score should DIP at the block borders, and the
#       TAD-boundary caller should recover exactly those border bins.
#
#     * COVERAGE BIAS: each bin is given a multiplicative "visibility" factor
#       (some bins are over-sequenced). Every raw count is multiplied by
#       bias_i * bias_j. ICE balancing should RECOVER these factors (up to a
#       global scale) and flatten every row sum -- that is the thing ICE exists
#       to undo, so embedding a known bias lets us see ICE work.
#
#   The matrix is SYMMETRIC; we store only the upper triangle (i <= j), the same
#   convention the loader (reference_cpu.cpp) and the GPU kernel expect.
#
#   Counts are deterministic INTEGERS (no RNG) so expected_output.txt is stable.
#
# OUTPUT FORMAT (see data/README.md):
#   line 1 : "n nnz"
#   next nnz lines : "i j count"   with 0 <= i <= j < n, count > 0
#
# USAGE
#   python scripts/make_synthetic.py                 # writes the committed sample
#   python scripts/make_synthetic.py --out other.txt
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "hic_sample.txt"

# Genome model: 12 bins split into three TADs. Block borders (the first bin of
# each new block) are at 4 and 8 -> the insulation score should bottom out near
# there. Keeping it 12 bins keeps the printed report short and human-checkable.
N_BINS = 12
BLOCKS = [(0, 4), (4, 8), (8, 12)]    # half-open [start, end) bin ranges

# Per-bin coverage bias (visibility). Deliberately uneven so ICE has work to do.
# Length must equal N_BINS. Values are O(1); ICE recovers them up to global scale.
BIAS = [1.0, 1.6, 0.8, 1.2,           # block 0
        1.4, 0.9, 1.1, 1.3,           # block 1
        0.7, 1.5, 1.0, 1.2]           # block 2


def base_count(i, j):
    """Bias-free 'true' contact frequency between bins i and j.

    Polymer intuition: contact frequency falls with genomic distance, and is
    much higher WITHIN a TAD than across a TAD border. We encode that with a
    simple integer model (no randomness, so the sample is reproducible):
      * within the same block: strong, distance-decaying contacts
      * across a block border: a weak background only
    """
    d = abs(i - j)
    same_block = any(s <= i < e and s <= j < e for (s, e) in BLOCKS)
    if i == j:
        return 50                      # strong diagonal (self/adjacent ligation)
    if same_block:
        # Distance decay inside a domain: 40 at d=1 down to a small floor.
        return max(40 - 8 * (d - 1), 6)
    else:
        # Cross-domain background: weak, decays fast with distance.
        return max(8 - d, 1)


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic Hi-C sample.")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    lines = []
    nnz = 0
    for i in range(N_BINS):
        for j in range(i, N_BINS):     # upper triangle only (i <= j)
            base = base_count(i, j)
            if base <= 0:
                continue
            # Apply the multiplicative coverage bias and round to an integer count
            # (real Hi-C counts are integers). int() truncation is deterministic.
            count = int(round(base * BIAS[i] * BIAS[j]))
            if count <= 0:
                continue
            lines.append(f"{i} {j} {count}")
            nnz += 1

    text = f"{N_BINS} {nnz}\n" + "\n".join(lines) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(n={N_BINS}, nnz={nnz}; SYNTHETIC, 3 TADs, borders at bins 4 and 8)")


if __name__ == "__main__":
    main()
