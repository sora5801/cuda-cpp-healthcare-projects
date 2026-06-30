#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic protein sample
# ---------------------------------------------------------------------------
# Project 3.18 : Protein Language Model Inference
#
# WHY THIS EXISTS
#   Real protein language models (ESM-2) ship hundreds of megabytes of TRAINED
#   weights that we deliberately do NOT redistribute. This project instead
#   GENERATES every weight deterministically from an integer hash inside the C++
#   code (see src/attention_math.h), so the only "data" it needs is a protein
#   SEQUENCE and the model shape. This script writes that tiny sample file.
#
#   The sample is a short, clearly-SYNTHETIC peptide -- it is a real-looking
#   amino-acid string but carries no biological meaning; it exists so the
#   attention demo has a concrete, reproducible input. Synthetic data is always
#   LABELED synthetic (CLAUDE.md §8).
#
# OUTPUT FORMAT (see data/README.md):
#   line 1 : "<d_model> <n_heads>"
#   line 2 : the amino-acid sequence (letters from the 20 canonical residues)
#
# USAGE
#   python scripts/make_synthetic.py                 # default 24-residue sample
#   python scripts/make_synthetic.py --len 64        # a longer synthetic peptide
#   python scripts/make_synthetic.py --d-model 64 --heads 8
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "protein_sample.txt"

# The 20 canonical amino acids, matching AA_ALPHABET in src/attention_math.h.
AA = "ACDEFGHIKLMNPQRSTVWY"

# A fixed, hand-picked 24-residue peptide used as the committed sample. It looks
# like a plausible protein fragment (it is NOT one) and gives the attention block
# a varied set of residues to mix. Kept short so the CPU verifies in microseconds.
DEFAULT_SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLE"   # 24 residues, SYNTHETIC


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic protein sample.")
    ap.add_argument("--d-model", type=int, default=32, help="embedding width (must be a multiple of --heads)")
    ap.add_argument("--heads", type=int, default=4, help="number of attention heads")
    ap.add_argument("--len", type=int, default=0,
                    help="if >0, generate a random peptide of this length instead of the default")
    ap.add_argument("--seed", type=int, default=3, help="RNG seed for --len peptides")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    if args.d_model % args.heads != 0:
        ap.error(f"d_model ({args.d_model}) must be divisible by heads ({args.heads})")
    if args.d_model // args.heads > 64:
        ap.error("d_head = d_model/heads must be <= 64 (the teaching kernel's register cap)")

    if args.len > 0:
        rng = random.Random(args.seed)
        seq = "".join(rng.choice(AA) for _ in range(args.len))
    else:
        seq = DEFAULT_SEQUENCE

    lines = [f"{args.d_model} {args.heads}", seq]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(L={len(seq)} residues, d_model={args.d_model}, heads={args.heads}; SYNTHETIC)")


if __name__ == "__main__":
    main()
