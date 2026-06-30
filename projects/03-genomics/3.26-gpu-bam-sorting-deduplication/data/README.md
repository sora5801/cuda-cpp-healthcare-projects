# Data — 3.26 GPU BAM Sorting & Deduplication

## Committed sample (`sample/reads_sample.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** aligned reads (`scripts/make_synthetic.py`, seed 326) |
| License | Public domain (CC0) — synthetic |
| Size | ~40 KB |
| Contents | 2,000 reads across 4 references, with ~358 planted PCR duplicates |

This tiny file lets `demo/run_demo` run **offline, with zero downloads**, a hard
requirement for every project (CLAUDE.md §8).

### File format

```
<n> <num_refs>                              # number of reads, number of references
<ref_id> <pos> <strand> <mate_pos> <base_qual_sum>   # one line per read, n lines
...
```

Each read line carries exactly the fields the sort + dedup look at (a real BAM
record has 11+ fields; we keep only these — see `src/bam.h`):

| Field | Meaning | Range in the sample |
|---|---|---|
| `ref_id` | reference/chromosome index (BAM `RNAME` as a 0-based index) | `0 .. num_refs-1` |
| `pos` | 0-based leftmost mapped coordinate (BAM `POS`) | `[0, 2^24)` |
| `strand` | `0` = forward (+), `1` = reverse (−) (from BAM `FLAG` bit `0x10`) | `{0,1}` |
| `mate_pos` | mate / fragment-end coordinate (BAM `PNEXT`); part of the dup signature | `[0, 2^15)` |
| `base_qual_sum` | sum of base qualities — Picard's duplicate **score** (higher = keep) | `≥ 0` |

The read's original line index becomes its `id` (the total-order tie-breaker, so
both the sort and the dedup are deterministic — see `THEORY.md`).

### What is engineered into the sample (so the answer is checkable)

- Reads are **scattered** across 4 references at random positions, so coordinate
  sorting genuinely reorders them (the input is shuffled, not pre-sorted).
- A controlled fraction are **PCR/optical duplicates**: a fragment signature
  `(ref, pos, strand, mate_pos)` is emitted 2–5 times with *distinct*
  base-quality sums, so exactly one copy per cluster is the original and the rest
  are duplicates. The generator prints the planted duplicate count (**358**),
  which must equal the demo's reported count.

## Full dataset

Real aligned reads live in **BAM** files (compressed BGZF). `scripts/download_data.*`
prints where to get them and how to convert a BAM into the text format above with
`samtools view` + `awk`:

- **1000 Genomes** WGS BAMs — <https://www.internationalgenome.org/data> (open)
- **ENCODE** ChIP-seq BAMs — <https://www.encodeproject.org/> (open)
- **TCGA** cancer WGS BAMs — <https://portal.gdc.cancer.gov/> (controlled access)
- **ICGC PCAWG** BAMs — <https://dcc.icgc.org/> (controlled access)

The controlled-access sets require their own registration; the download script
**does not** bypass it — it prints instructions and links only (CLAUDE.md §8).

## Provenance & honesty

The sample is **synthetic** — randomly generated reads with a deliberately
planted duplicate structure, **not** real patient sequencing data, and it carries
no clinical meaning. It exists to make the sort + dedup result interpretable (the
duplicate count is known) and the GPU-vs-CPU comparison verifiable.
