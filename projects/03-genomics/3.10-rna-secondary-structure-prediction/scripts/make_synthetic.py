#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic RNA sample (FASTA)
# ---------------------------------------------------------------------------
# Project 3.10 : RNA Secondary-Structure Prediction  (Nussinov base-pair DP)
#
# WHY THIS EXISTS
#   Real RNA-structure databases (Rfam, RNAcentral, the PDB, ArchiveII) are
#   public but large, and we want the demo to RUN OFFLINE with zero downloads.
#   So we commit a tiny, clearly-SYNTHETIC sequence and generate it here. A
#   synthetic sequence is fine for teaching the algorithm: Nussinov maximises
#   base pairs and does not care whether the molecule is real -- and we LABEL it
#   synthetic everywhere (CLAUDE.md §8). For real structures, see download_data.*.
#
# WHAT IT WRITES
#   A FASTA file: a '>' header (marked SYNTHETIC) and one RNA sequence line over
#   the alphabet {A,C,G,U}. The default sequence is an 18-nt designed HAIRPIN:
#   a 6-bp GC-rich stem closing a 4-base "AAAA" loop, with two unpaired 3' bases.
#   It folds (Nussinov, MIN_LOOP=3) to exactly 6 base pairs, "((((((....))))))..".
#   This known answer is what demo/expected_output.txt encodes, so the demo is a
#   real correctness check, not a tautology.
#
# USAGE
#   python scripts/make_synthetic.py                       # default hairpin
#   python scripts/make_synthetic.py --seq GGGAAACCC       # custom sequence
#   python scripts/make_synthetic.py --random 40 --seed 7  # random length-40 RNA
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent             # the project folder
OUT = ROOT / "data" / "sample" / "rna_sample.fasta"

# The default committed sequence: a designed hairpin with a known optimal fold.
DEFAULT_SEQ = "GGGCGCAAAAGCGCCCAU"


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic RNA FASTA sample.")
    ap.add_argument("--seq", default=None,
                    help="explicit RNA sequence (A/C/G/U); overrides --random")
    ap.add_argument("--random", type=int, default=0, metavar="LEN",
                    help="generate a random RNA of this length instead")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for --random")
    ap.add_argument("--out", default=str(OUT), help="output FASTA path")
    args = ap.parse_args()

    if args.seq:
        seq = args.seq.strip().upper().replace("T", "U")
    elif args.random > 0:
        # A random sequence still folds; it just has no designed structure. Handy
        # for stress-testing the wavefront on longer inputs (an exercise).
        rng = random.Random(args.seed)
        seq = "".join(rng.choice("ACGU") for _ in range(args.random))
    else:
        seq = DEFAULT_SEQ

    bad = set(seq) - set("ACGU")
    if bad:
        raise SystemExit(f"[make_synthetic] non-ACGU characters in sequence: {sorted(bad)}")

    header = ">synthetic_rna_%dnt | SYNTHETIC teaching RNA (not a real transcript)" % len(seq)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(header + "\n" + seq + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  (length={len(seq)}; SYNTHETIC)")


if __name__ == "__main__":
    main()
