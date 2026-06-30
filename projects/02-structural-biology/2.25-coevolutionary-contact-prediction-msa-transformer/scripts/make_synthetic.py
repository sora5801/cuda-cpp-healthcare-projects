#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic protein MSA with planted
#                                 coevolving (contact) column pairs
# ---------------------------------------------------------------------------
# Project 2.25 : Coevolutionary Contact Prediction & MSA Transformer
#
# WHY SYNTHETIC
#   Real coevolution needs a deep MSA of a real protein family (UniRef/Pfam),
#   which is large and license-encumbered (see scripts/download_data.*). For a
#   self-contained, offline, VERIFIABLE demo we instead generate an MSA in which
#   we KNOW the ground truth: a handful of column PAIRS are made to coevolve
#   (mimicking residues that touch in 3-D and mutate together), while all other
#   columns are independent (conserved or freely variable). A correct method
#   must rank the planted pairs at the very top -- which is exactly what the demo
#   checks. This makes the result interpretable AND the GPU/CPU comparison
#   meaningful. The data is SYNTHETIC and carries no biological/clinical meaning.
#
# HOW WE PLANT COEVOLUTION
#   For each "contact" pair (i, j) we draw, per sequence, a matched pair of amino
#   acids from a shared codebook (e.g. a salt bridge: Asp<->Lys). Because column
#   i's residue determines column j's, the two columns share information -> high
#   Mutual Information -> high APC score. Non-contact columns are emitted
#   independently (some conserved, some variable), so they carry little pairwise
#   information and must NOT rank at the top.
#
# OUTPUT  (FASTA alignment, the format load_msa() in reference_cpu.cpp expects):
#   >seq0
#   <L residues>
#   >seq1
#   <L residues>
#   ... (N records, every sequence exactly L columns)
#
# USAGE
#   python scripts/make_synthetic.py                 # default tiny sample
#   python scripts/make_synthetic.py --n 4000 --seed 7
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "coevolution_msa.fasta"

# Alignment length (columns / alignment positions). Small so the committed
# sample stays tiny and the demo is instant, but big enough that the planted
# contacts must out-rank many decoy pairs.
L = 24

# The PLANTED CONTACTS: pairs of 0-based columns that we force to coevolve. The
# demo expects these (as 1-based columns) to dominate the top of the ranking.
CONTACT_PAIRS = [
    (2, 21),    # long-range "clamp": near-N-terminus residue touches near-C-term
    (5, 18),
    (8, 15),
    (3, 4),     # a local i,i+1 backbone contact
]

# Coevolution "codebook": each contact pair, per sequence, picks one of these
# matched residue pairs at random. The two members covary perfectly (knowing the
# residue in column i tells you the residue in column j), creating mutual info.
# Chosen to look like real complementary contacts (charge / size complementarity).
COEVO_ALPHABET = [
    ("D", "K"),   # Asp(-)  <-> Lys(+)   salt bridge
    ("E", "R"),   # Glu(-)  <-> Arg(+)   salt bridge
    ("K", "D"),   # Lys(+)  <-> Asp(-)   (swap) salt bridge
    ("W", "G"),   # big aromatic <-> tiny Gly (size complementarity)
    ("I", "V"),   # hydrophobic packing pair
]

# Conserved columns keep one fixed residue for the whole alignment (low entropy),
# so near-zero MI with anything -> they must NOT rank.
CONSERVED = "ACDEFGHIKLMNPQRSTVWY"

# Standard 20 amino acids for the freely-variable background columns.
AA20 = "ACDEFGHIKLMNPQRSTVWY"


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic coevolution MSA (SYNTHETIC).")
    ap.add_argument("--n", type=int, default=400, help="number of sequences (MSA depth)")
    ap.add_argument("--seed", type=int, default=2025, help="RNG seed (determinism)")
    ap.add_argument("--noise", type=float, default=0.03,
                    help="per-residue substitution noise on conserved/contact columns")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Decide each column's ROLE up front:
    #   - columns that are a member of a contact pair  -> driven by the codebook
    #   - a few fixed columns                          -> conserved (low entropy)
    #   - everything else                              -> freely variable background
    contact_cols = set()
    for (i, j) in CONTACT_PAIRS:
        contact_cols.add(i)
        contact_cols.add(j)
    # A deterministic handful of conserved columns from the leftovers.
    conserved_cols = {0, 7, 12, 20}
    conserved_cols -= contact_cols          # contacts win if they overlap
    # Each conserved column keeps one fixed residue for the whole alignment.
    conserved_res = {c: CONSERVED[(c * 7) % len(CONSERVED)] for c in conserved_cols}

    def maybe_mutate(residue):
        # With probability `noise`, replace a residue with a random amino acid
        # (adds realistic alignment noise without destroying the signal).
        if rng.random() < args.noise:
            return rng.choice(AA20)
        return residue

    records = []
    for s in range(args.n):
        col = ["A"] * L   # will be overwritten for every column

        # 1) Fill the contact pairs from the shared codebook (this is the signal).
        for (i, j) in CONTACT_PAIRS:
            ai, aj = rng.choice(COEVO_ALPHABET)
            col[i] = maybe_mutate(ai)
            col[j] = maybe_mutate(aj)

        # 2) Fill conserved columns with their fixed residue (+ a little noise).
        for c, res in conserved_res.items():
            col[c] = maybe_mutate(res)

        # 3) Fill everything else with an independent random amino acid.
        for c in range(L):
            if c in contact_cols or c in conserved_cols:
                continue
            col[c] = rng.choice(AA20)

        records.append((f"seq{s}", "".join(col)))

    text = []
    for (name, seq) in records:
        text.append(f">{name}")
        text.append(seq)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(text) + "\n", encoding="utf-8")

    planted = ", ".join(f"({i + 1},{j + 1})" for (i, j) in CONTACT_PAIRS)
    print(f"[make_synthetic] wrote {args.out}")
    print(f"[make_synthetic]   N={args.n} sequences, L={L} columns  (SYNTHETIC)")
    print(f"[make_synthetic]   planted coevolving contacts (1-based cols): {planted}")


if __name__ == "__main__":
    main()
