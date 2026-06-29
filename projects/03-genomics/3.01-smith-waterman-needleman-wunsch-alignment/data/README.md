# Data — 3.01 Smith-Waterman / Needleman-Wunsch Alignment

## Committed sample (`sample/sequences_sample.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** (`scripts/make_synthetic.py`, seed 11) |
| License | Public domain (CC0) — synthetic |
| Contents | Two DNA sequences (line 1 = query, line 2 = target) |
| Design | The target embeds a **mutated copy** of a motif from the query, so there is a clear high-scoring local alignment to find. |

### File format

```
<query sequence on line 1>      # A/C/G/T, length M
<target sequence on line 2>     # A/C/G/T, length N
```

Sequence lengths `M`, `N` are taken from the line lengths (stray whitespace is
ignored). Only `A/C/G/T` are accepted (`src/reference_cpu.cpp::load_sequences`).

## Full dataset

Real alignments use protein/nucleotide FASTA from public databases:

- **UniProtKB/Swiss-Prot** — curated proteins (~570k): <https://www.uniprot.org/downloads>
- **NCBI nr** — non-redundant protein (100M+): <https://ftp.ncbi.nlm.nih.gov/blast/db/>
- **NCBI RefSeq** — reference nucleotide/protein: <https://ftp.ncbi.nlm.nih.gov/refseq/>
- **PDB sequences** — for benchmarking: <https://www.rcsb.org/downloads>

`scripts/download_data.ps1` / `.sh` print how to extract two sequences from a
FASTA file into this format. For a larger synthetic problem:

```
python scripts/make_synthetic.py --motif 400 --mut 0.2
```

> This teaching project uses a **DNA** alphabet and **linear** gap scoring.
> Protein alignment adds a substitution matrix (BLOSUM/PAM) and affine gaps —
> see THEORY.md "Where this sits in the real world".

## Provenance & honesty

The sample is **synthetic** and labeled as such. The alignment is a software
test, not a biological finding.
