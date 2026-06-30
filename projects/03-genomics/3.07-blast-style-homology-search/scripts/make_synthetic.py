#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic protein FASTA dataset
# ---------------------------------------------------------------------------
# Project 3.7 : BLAST-Style Homology Search
#
# WHY SYNTHETIC
#   Real protein databases (UniRef, NCBI nr, PDB70 -- see data/README.md) are
#   huge and licensed. To keep the demo OFFLINE, REPRODUCIBLE, and INTERPRETABLE
#   we generate a tiny, clearly-SYNTHETIC FASTA whose homology relationships are
#   KNOWN by construction, so the top-K result is meaningful and deterministic:
#
#     record 0 (query)  : a random "query" protein.
#     hit_close         : the query with ~8% of residues mutated  -> strong hit.
#     hit_medium        : the query with ~25% mutated             -> medium hit.
#     hit_domain        : an unrelated protein with the query's MIDDLE 40 residues
#                         spliced in (a shared "domain")          -> partial hit.
#     decoy_1..k        : fully random proteins (no homology)     -> low/zero.
#
#   So the EXPECTED ranking is: hit_close > hit_medium > hit_domain > decoys.
#   A fixed RNG seed makes the FASTA byte-for-byte reproducible, which keeps
#   demo/expected_output.txt stable. SYNTHETIC is stated in every header.
#
# OUTPUT: data/sample/proteins_sample.fasta  (FASTA; first record = query).
#
# USAGE
#   python scripts/make_synthetic.py                 # default small sample
#   python scripts/make_synthetic.py --decoys 50     # a bigger decoy set
# ===========================================================================
import argparse
import random
from pathlib import Path

# The 20 standard amino acids (single-letter codes). We deliberately do NOT emit
# B/Z/X/* ambiguity codes in synthetic data so every k-mer is a valid seed.
AMINO = "ACDEFGHIKLMNPQRSTVWY"

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "proteins_sample.fasta"


def random_protein(rng, length):
    """A length-L protein of i.i.d. uniform amino acids."""
    return "".join(rng.choice(AMINO) for _ in range(length))


def mutate(rng, seq, frac):
    """Return a copy of seq with ~frac of positions substituted to a random
    (possibly different) amino acid. Models point mutations / divergence."""
    out = list(seq)
    for i in range(len(out)):
        if rng.random() < frac:
            out[i] = rng.choice(AMINO)
    return "".join(out)


def wrap(seq, width=60):
    """Wrap a sequence to FASTA line width (cosmetic; our loader handles any)."""
    return "\n".join(seq[i:i + width] for i in range(0, len(seq), width))


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic protein FASTA for BLAST.")
    ap.add_argument("--qlen", type=int, default=120, help="query length (residues)")
    ap.add_argument("--decoys", type=int, default=6, help="number of unrelated decoy sequences")
    ap.add_argument("--seed", type=int, default=7, help="RNG seed (determinism)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # The query protein.
    query = random_protein(rng, args.qlen)

    # Build the database records as (header, sequence) in a FIXED order so the
    # output is deterministic. Headers carry the SYNTHETIC label and the designed
    # relationship, so a reader knows exactly what each sequence is.
    records = [(f"QUERY synthetic_query len={args.qlen}", query)]

    # Strong homolog: a near-duplicate (low divergence).
    records.append(("hit_close synthetic ~8pct_diverged_homolog",
                    mutate(rng, query, 0.08)))

    # Medium homolog: noticeably diverged but still clearly related.
    records.append(("hit_medium synthetic ~25pct_diverged_homolog",
                    mutate(rng, query, 0.25)))

    # Domain sharer: an unrelated scaffold with the query's middle 40 residues
    # spliced in -> one strong local HSP over the shared "domain", flanks random.
    mid_start = args.qlen // 2 - 20
    domain = query[mid_start:mid_start + 40]
    scaffold = random_protein(rng, 50) + domain + random_protein(rng, 50)
    records.append(("hit_domain synthetic shared_40aa_domain", scaffold))

    # Decoys: fully random proteins of varied length (no designed homology).
    for d in range(args.decoys):
        L = rng.randint(90, 150)
        records.append((f"decoy_{d+1} synthetic random_protein", random_protein(rng, L)))

    # Emit FASTA. First record is the query (the loader's convention).
    lines = []
    for header, seq in records:
        lines.append(">" + header)
        lines.append(wrap(seq))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(query len={args.qlen}, {len(records)-1} DB seqs; SYNTHETIC, seed={args.seed})")


if __name__ == "__main__":
    main()
