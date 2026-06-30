#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic posterior dataset
# ---------------------------------------------------------------------------
# Project 3.4 : Nanopore Basecalling  (REDUCED-SCOPE: CTC greedy decode)
#
# WHY THIS EXISTS
#   Real nanopore posterior matrices come out of a trained neural network run on
#   raw squiggle signal (Dorado/Guppy) -- not redistributable here, and producing
#   them is the research-grade part this project deliberately leaves out
#   (CLAUDE.md sec 13). So we synthesize the network's OUTPUT directly: for each
#   read we plant a KNOWN DNA sequence and build a posterior matrix that decodes
#   back to it under greedy CTC. The demo then RECOVERS the planted sequence,
#   which is a human-checkable proof the decoder works (PATTERNS.md sec 6).
#   The data is SYNTHETIC and labeled as such everywhere.
#
# HOW WE ENCODE A KNOWN SEQUENCE INTO POSTERIORS
#   CTC alphabet (must match src/ctc_core.h):  0=blank, 1=A, 2=C, 3=G, 4=T.
#   For each base in the target sequence we emit:
#       * a BLANK separator step  (so that two identical adjacent bases -- a
#         homopolymer like "AA" -- survive the collapse instead of merging), then
#       * `dwell` steps whose argmax is that base's class (a base physically
#         dwells in the pore for several signal samples -> several time steps).
#   Greedy decode (argmax -> merge repeats -> drop blanks) of this matrix is
#   exactly the original sequence. We make the correct class clearly dominant so
#   the argmax is unambiguous and the result is deterministic.
#
#   This mirrors the real model's behavior: blanks separate events, and one base
#   spans many steps. We are NOT claiming these probabilities are realistic
#   network outputs -- only that they exercise the SAME decode logic.
#
# DETERMINISM
#   A fixed RNG seed makes the file byte-reproducible, so committing it and
#   regenerating it give identical bytes (and a stable expected_output.txt).
#
# USAGE
#   python scripts/make_synthetic.py                 # default tiny committed sample
#   python scripts/make_synthetic.py --reads 64      # a bigger synthetic batch
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT  = ROOT / "data" / "sample" / "reads_sample.txt"

NUM_CLASSES = 5                                        # {blank, A, C, G, T}
BASE_TO_CLASS = {"A": 1, "C": 2, "G": 3, "T": 4}       # must match ctc_core.h

# A few hand-picked target sequences for the committed sample. They include a
# homopolymer ("AA", "TT", "GG") on purpose -- the case where the blank separator
# matters -- so the demo visibly exercises that subtlety.
DEFAULT_TARGETS = [
    "ACGTACGT",        # simple, all distinct neighbours
    "AACCGGTT",        # homopolymers: each base doubled
    "GATTACA",         # a classic; note the "TT" homopolymer
    "TTTTGGGG",        # long homopolymer runs
]


def emit_read(target, dwell, p_correct, rng):
    """Build a [T x NUM_CLASSES] posterior matrix that greedily decodes to
    `target`. Returns (T, rows) where rows is a list of NUM_CLASSES-float lists.

    For each base: one BLANK-dominant step, then `dwell` steps dominant in that
    base's class. The dominant class gets probability p_correct; the rest split
    the remainder (with a little deterministic jitter so it looks less synthetic
    but the argmax never changes -- the dominant class stays clearly on top)."""
    rows = []
    remainder = (1.0 - p_correct) / (NUM_CLASSES - 1)   # split among the others

    def make_row(dominant_class):
        # Start every class at the shared "remainder" mass, then lift the
        # dominant one to p_correct. Add tiny jitter to the non-dominant classes
        # (bounded well below the gap to p_correct, so argmax is unchanged).
        row = [remainder] * NUM_CLASSES
        row[dominant_class] = p_correct
        for c in range(NUM_CLASSES):
            if c != dominant_class:
                row[c] = max(1e-4, remainder + rng.uniform(-remainder / 4, remainder / 4))
        return row

    for base in target:
        cls = BASE_TO_CLASS[base]
        rows.append(make_row(0))                 # a blank separator step
        for _ in range(dwell):
            rows.append(make_row(cls))           # the base dwells `dwell` steps
    return len(rows), rows


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic CTC posterior sample.")
    ap.add_argument("--reads", type=int, default=len(DEFAULT_TARGETS),
                    help="number of reads (cycles through / extends the default targets)")
    ap.add_argument("--dwell", type=int, default=3, help="time steps each base dwells")
    ap.add_argument("--p-correct", type=float, default=0.80,
                    help="probability mass on the correct class per step (0..1)")
    ap.add_argument("--seed", type=int, default=20260628, help="RNG seed (reproducible)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Build the list of target sequences. For more reads than we have hand-picked
    # targets, generate extra random ones deterministically.
    targets = list(DEFAULT_TARGETS)
    while len(targets) < args.reads:
        L = rng.randint(6, 14)
        targets.append("".join(rng.choice("ACGT") for _ in range(L)))
    targets = targets[:args.reads]

    # Serialize in the loader's format:
    #   line 1: "<n_reads> <C>"
    #   per read: a line "<T>" then T lines of C floats.
    lines = [f"{len(targets)} {NUM_CLASSES}"]
    for tgt in targets:
        T, rows = emit_read(tgt, args.dwell, args.p_correct, rng)
        lines.append(str(T))
        for row in rows:
            lines.append(" ".join(f"{v:.4f}" for v in row))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {out_path}")
    print(f"[make_synthetic] {len(targets)} SYNTHETIC reads; planted sequences:")
    for i, t in enumerate(targets):
        print(f"  read {i}: {t}")


if __name__ == "__main__":
    main()
