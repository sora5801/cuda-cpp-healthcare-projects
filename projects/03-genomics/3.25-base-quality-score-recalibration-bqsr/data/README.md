# Data — 3.25 Base Quality Score Recalibration (BQSR)

## Committed sample (`sample/bqsr_sample.txt`)

| Field | Value |
|---|---|
| File | `sample/bqsr_sample.txt` |
| Origin | **Synthetic** (`scripts/make_synthetic.py`, seed 7) |
| License | Public domain (CC0) — it is synthetic |
| Size | ~30 KB |
| Contents | 24 bp reference, 2 known-variant sites, 1200 reads × 12 bp at reported Q30 |

### File format

```
REF <reference-string>                       # the reference bases (ACGTN)
KNOWN <p1> <p2> ...                          # 0-based known-variant positions (masked)
READS <R> <L>                                # R reads, each L bases long (L <= 16)
<pos> <bases(L chars)> <q0> <q1> ... <q(L-1)>   # one line per read
... (R such lines)
```

- **`REF`** — the reference substring this tile aligns to. A read base at cycle
  `c` of a read starting at `pos` is compared against `REF[pos + c]`.
- **`KNOWN`** — reference positions flagged as known variants (dbSNP/Mills in real
  BQSR). Bases at these columns are **skipped** when building the covariate table.
- **`READS`** — each read is `pos` (reference start), an `L`-character base string,
  then `L` integer PHRED quality scores.

### How the sample is engineered (so the result is interpretable)

Every base is **reported** at **Q30** (claimed 0.1% error), but the generator
injects a **~1.2% true error rate**, so the recovered empirical quality is
**~Q19** — the headline miscalibration BQSR corrects. Two reference columns (7 and
16) are **known variants** where every covering read carries a fixed alternate
allele; because BQSR masks known sites, those systematic "mismatches" do **not**
count as machine errors. See `demo/expected_output.txt`.

## Full dataset

Real BQSR runs on an aligned **BAM** plus known-variant **VCFs**:

- **dbSNP build 155** — known SNP positions for masking: <https://www.ncbi.nlm.nih.gov/snp/>
- **Mills & 1000G indels** (GATK bundle): <https://storage.googleapis.com/genomics-public-data/>
- **GIAB known-variant VCFs**: <https://www.nist.gov/programs-projects/genome-bottle>
- **1000 Genomes high-coverage WGS**: <https://www.internationalgenome.org/data>

`scripts/download_data.ps1` / `.sh` print these pointers (no credentials are
bypassed). To use real data, export a region's reference, known-variant positions,
and reads into the text format above (most BAM toolkits — samtools, pysam — can
emit per-read position/bases/qualities).

## Provenance & honesty

The sample is **synthetic** — pseudo-random reads with a *known* injected error
rate and hand-placed known-variant sites. It is **not** real sequencing data and
carries **no clinical meaning**. It exists only to make the recalibration result
verifiable (the recovered `Q_emp` matches the injected rate) and the GPU/CPU
comparison exact. Synthetic data is labeled synthetic everywhere it appears
(CLAUDE.md §8).
