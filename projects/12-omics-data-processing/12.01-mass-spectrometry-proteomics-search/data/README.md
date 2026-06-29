# Data — 12.01 Mass-Spectrometry Proteomics Search

## Committed sample (`sample/spectra_sample.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** spectra (`scripts/make_synthetic.py`, seed 4) |
| License | Public domain (CC0) — synthetic |
| Contents | 1 query + 1024 library spectra, 200 intensity bins each |

### File format

```
<N> <bins> <target>        # library size, bins/spectrum, known target index (-1 if none)
<query: bins floats>       # the observed spectrum
<library 0: bins floats>   # one theoretical spectrum per row
... (N rows)
```

Each spectrum is a sparse set of fragment peaks (~18 per spectrum) binned to a
fixed-length intensity vector. The **query** is library spectrum `target` with
per-peak intensity jitter plus a few stray peaks, so its best cosine match is the
target.

## Full dataset

Real searches match observed MS/MS spectra (from **mzML**) against a peptide
database (in-silico digested + fragmented):

- **ProteomeXchange / PRIDE** (<https://www.proteomexchange.org>) — raw/mzML repositories.
- **MSFragger** (<https://github.com/Nesvilab/MSFragger>) — fast database search.
- **GiCOPS** (<https://github.com/pcdslab/gicops>) — GPU database peptide search.
- **OpenMS** (<https://github.com/OpenMS/OpenMS>) — proteomics toolkit (mzML I/O, scoring).

Bin observed peaks + theoretical fragment ions to the same grid, then write the
format above. Bigger synthetic set: `python scripts/make_synthetic.py --N 8192`.

## Provenance & honesty

The spectra are **synthetic** random peak patterns, not real fragmentation data,
and carry no biological meaning. They exist to make the search result interpretable
(the target is recovered) and the GPU/CPU comparison verifiable.
