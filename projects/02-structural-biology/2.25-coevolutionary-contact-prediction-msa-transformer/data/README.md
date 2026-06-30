# Data — 2.25 Coevolutionary Contact Prediction & MSA Transformer

## Committed sample (`sample/coevolution_msa.fasta`)

| Field | Value |
|---|---|
| File | `sample/coevolution_msa.fasta` |
| Origin | **Synthetic** MSA with planted coevolving pairs (`scripts/make_synthetic.py`, seed 2025) |
| License | Public domain (CC0) — it is synthetic |
| Size | ~14 KB (400 sequences × 24 columns) |
| Format | FASTA: `>header` lines alternating with one sequence line each |

This tiny file lets `demo/run_demo` run **offline, with zero downloads**, which
is a hard requirement for every project (CLAUDE.md §8).

### File format (what the loader expects)

```
>seq0
AVIWGETLWDHHFAIGPRRNAVTA      # exactly L=24 residues, one-letter amino-acid code
>seq1
ARIIVDVLERPDFWQRWAKMAVYI
... (N=400 records, every sequence the same length)
```

`load_msa()` (in `src/reference_cpu.cpp`) maps each letter to a token in
`[0, 21)` via `cv_token_of_aa` (`src/coevolution.h`): the 20 standard amino acids
`ACDEFGHIKLMNPQRSTVWY` → `0..19`, and gap `-` / unknown `X` / anything else → `20`.
Every sequence **must** have the same length (a valid alignment) or the loader
throws.

### What is planted (the ground truth)

The synthetic MSA hides a **known** answer so the result is verifiable. Four
column pairs are forced to **coevolve** (one column's residue determines the
other's, as for two residues that touch in 3-D and mutate together):

| Planted contact (1-based columns) | Role |
|---|---|
| (3, 22) | long-range "clamp" (N-term ↔ C-term) |
| (6, 19) | medium-range contact |
| (9, 16) | medium-range contact |
| (4, 5)  | local backbone contact (i, i+1) |

A correct coevolution method must rank these **four pairs at the top**. The demo
confirms exactly that: in `expected_output.txt`, contacts #1–#4 are the four
planted pairs (APC ≈ 1.3–1.4), an order of magnitude above the best decoy
(APC ≈ 0.13). A few columns are deliberately **conserved** (near-zero entropy)
and the rest are **independent random** background, so they carry little pairwise
information and stay out of the top ranks.

Bigger synthetic set (deeper MSA, sharper signal):
`python scripts/make_synthetic.py --n 4000 --seed 7`.

## Full dataset (real protein families)

Real coevolution analysis needs a **deep MSA** of a real protein family. Nothing
here requires credentials, but the files are large and better fetched on demand —
`scripts/download_data.*` print the pointers (they do not auto-download):

- **UniRef50 / UniRef90** — <https://www.uniprot.org/help/uniref> — the sequence
  databases you search (with `jackhmmer`/`HHblits`) to build an MSA for a query.
- **Pfam** — <http://pfam.xfam.org> — precomputed family MSAs (Stockholm format;
  convert to aligned FASTA, then to the format above).
- **EVcouplings benchmark contacts** — <https://github.com/debbiemarkslab/EVcouplings>
  — families with known PDB contacts to score predictions against.
- **CASP14 contact targets** — <https://predictioncenter.org> — community
  benchmark of contact/distance prediction.

To use a real MSA: download a Pfam family or build one with `jackhmmer`, save it
as aligned FASTA (one record per sequence, equal length), and pass the path:
`...\coevolutionary-contact-prediction-msa-transformer.exe path\to\family.fasta`.

## Provenance & honesty

The committed sample is **synthetic** (Gaussian-free, codebook-driven covariation
of planted column pairs), **not** a real protein family, and carries **no
biological or clinical meaning**. It exists to make the contact-prediction result
interpretable (the four planted contacts are recovered) and the GPU/CPU
comparison verifiable. Synthetic data is labeled synthetic everywhere it appears
(CLAUDE.md §8).
