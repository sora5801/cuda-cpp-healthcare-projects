#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic clinical-note sample
# ---------------------------------------------------------------------------
# Project 7.3 -- Clinical NLP over Notes & Records
#
# WHY THIS EXISTS
#   The real datasets named in the catalog (MIMIC-IV Notes, i2b2/n2c2) are
#   CREDENTIALED and CANNOT be redistributed. So the committed demo runs on a
#   clearly-SYNTHETIC stand-in generated here. Everything below is invented; it
#   contains NO real patient text and implies NO clinical validity.
#
# WHAT WE GENERATE  (a tiny "tokenized clinical-note batch" for one transformer
# self-attention encoder block -- see ../THEORY.md):
#
#   * A small VOCABULARY of clinical tokens (word-piece stand-ins). Token 0 is a
#     special [CLS] summary token, token 1 is [PAD] (used to fill short notes).
#   * A batch of B NOTES, each a sequence of token ids of length <= S (padded
#     with [PAD] to exactly S). Real notes are word-piece id streams; ours are a
#     handful of ids so a human can read the attention pattern by eye.
#   * Deterministic token EMBEDDINGS: row t of a [V x D] table, filled by a fixed
#     integer recipe (no RNG, no training) so the demo output is byte-stable and
#     reproducible on any machine. A real model would LEARN these on billions of
#     tokens; we fake them so the *mechanics* of attention are exact and legible.
#   * Deterministic projection matrices Wq, Wk, Wv (each [D x D]) built the same
#     fixed way. Multi-head attention splits the D columns into H heads.
#
#   The batch embeds a PLANTED pattern so the result is interpretable
#   (PATTERNS.md §6): in every note the pronoun token "he" is authored to share
#   its query/key direction with the subject token "patient", so a correct
#   attention block makes "he" attend most strongly to "patient" -- a toy
#   stand-in for the COREFERENCE resolution the catalog lists as a task. The demo
#   reports whether that link is recovered.
#
# FILE FORMAT (whitespace/'#'-comment text; parsed by src/reference_cpu.cpp):
#     line: V D H S B                # vocab, model dim, heads, seq len, notes
#     V lines:  tok: <id> <string>   # vocabulary (id then human-readable token)
#     B blocks, each:
#        note: <len> <id0> <id1> ...        # `len` real tokens, then padded to S
#     D lines:  emb: <V doubles>     # transposed? NO -- row r = embedding dim r
#                                    #   value for every vocab token (see loader)
#     (Wq, Wk, Wv are regenerated in code from a fixed seed, NOT stored, to keep
#      the sample small; the loader rebuilds them with the SAME recipe.)
#
#   To keep the committed file tiny and the math auditable we DO store the
#   embedding table explicitly but generate Wq/Wk/Wv from the shared integer
#   recipe in both the file header and the code. The header records the recipe
#   parameters so the loader is self-contained.
#
# USAGE
#   python scripts/make_synthetic.py            # writes data/sample/notes_sample.txt
#   python scripts/make_synthetic.py --seq 12 --notes 6
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "notes_sample.txt"

# ---------------------------------------------------------------------------
# The toy clinical vocabulary. Ids 0/1 are the special tokens; the rest are
# invented "word-piece" stand-ins for clinical terms. Order fixes the ids.
# ---------------------------------------------------------------------------
VOCAB = [
    "[CLS]",      # 0  sequence-summary token (its output row is the note vector)
    "[PAD]",      # 1  padding token (masked out of attention)
    "patient",    # 2  the subject noun (coreference antecedent)
    "he",         # 3  pronoun -> should attend to "patient" (planted link)
    "denies",     # 4
    "chest",      # 5
    "pain",       # 6
    "fever",      # 7
    "cough",      # 8
    "prescribed", # 9
    "aspirin",    # 10
    "and",        # 11
    "history",    # 12
    "of",         # 13
    "diabetes",   # 14
]

# A few hand-authored "notes" as token-id sequences (before padding). Each is a
# terse synthetic snippet; "he" always appears with "patient" so the planted
# coreference link is testable. [CLS] leads every note (BERT convention).
NOTE_STRINGS = [
    ["[CLS]", "patient", "denies", "chest", "pain", "he", "denies", "fever"],
    ["[CLS]", "patient", "history", "of", "diabetes", "he", "prescribed", "aspirin"],
    ["[CLS]", "patient", "cough", "and", "fever", "he", "denies", "chest"],
    ["[CLS]", "patient", "prescribed", "aspirin", "he", "denies", "pain"],
]

D_DEFAULT = 8     # model / embedding dimension (small so the file is tiny)
H_DEFAULT = 2     # attention heads (D must be divisible by H)


def embedding_recipe(tok_id: int, dim: int, D: int) -> float:
    """Deterministic 'pretend-learned' embedding value for (token, dim).

    A fixed trigonometric-ish integer recipe (no RNG) so the table is identical
    on every machine and the demo output is byte-stable. The PLANTED coreference
    link lives here: token "he" (id 3) is given the SAME embedding as "patient"
    (id 2) in the dimensions that dominate the query/key dot product, so after
    projection "he" points at "patient".
    """
    src = 2 if tok_id == 3 else tok_id          # "he" borrows "patient"'s vector
    # Integer-stable pseudo-features: bounded, smooth, distinct per (token,dim).
    v = ((src * 13 + dim * 7 + 1) % 17) - 8     # in [-8, 8]
    return v / 8.0                               # scale into [-1, 1]


def proj_recipe(i: int, j: int, kind: int, D: int) -> float:
    """Deterministic [D x D] projection-matrix entry (kind: 0=Wq,1=Wk,2=Wv).

    Near-identity plus a small structured perturbation so Q,K,V are distinct but
    the "he"~"patient" alignment survives projection. Same recipe is mirrored in
    src/attn_core.h so CPU and GPU build identical matrices.
    """
    base = 1.0 if i == j else 0.0                # start from identity
    pert = (((i * 3 + j * 5 + kind * 11) % 7) - 3) / 24.0   # small, in ~[-.125,.125]
    return base + pert


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic clinical-note sample.")
    ap.add_argument("--dim", type=int, default=D_DEFAULT, help="model dim D")
    ap.add_argument("--heads", type=int, default=H_DEFAULT, help="attention heads H")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    D, H = args.dim, args.heads
    if D % H != 0:
        raise SystemExit(f"D ({D}) must be divisible by H ({H})")
    V = len(VOCAB)
    tok2id = {t: i for i, t in enumerate(VOCAB)}
    notes = [[tok2id[t] for t in ns] for ns in NOTE_STRINGS]
    B = len(notes)
    S = max(len(n) for n in notes)               # sequence length (pad to this)

    lines = []
    lines.append("# SYNTHETIC clinical-note batch for project 7.3 (NOT real patient data).")
    lines.append("# One transformer self-attention encoder block; see ../THEORY.md.")
    lines.append("# header: V D H S B   (vocab, model dim, heads, seq len, notes)")
    lines.append(f"{V} {D} {H} {S} {B}")

    lines.append("# vocabulary:  tok: <id> <string>")
    for i, t in enumerate(VOCAB):
        lines.append(f"tok: {i} {t}")

    lines.append("# notes:  note: <len> <id0> <id1> ...   ([PAD]=1 fills to S)")
    for n in notes:
        ids = " ".join(str(x) for x in n)
        lines.append(f"note: {len(n)} {ids}")

    lines.append("# embeddings: emb row = dim d, columns = all V tokens")
    for d in range(D):
        row = " ".join(f"{embedding_recipe(t, d, D):+.6f}" for t in range(V))
        lines.append(f"emb: {row}")

    # Record the projection recipe parameters so the loader can rebuild Wq/Wk/Wv
    # without storing DxD*3 numbers (keeps the file tiny + auditable).
    lines.append("# proj: Wq/Wk/Wv are rebuilt in code from the shared integer recipe")
    lines.append("proj: recipe-v1")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(V={V}, D={D}, H={H}, S={S}, B={B}; SYNTHETIC)")


if __name__ == "__main__":
    main()
