# Data — 3.30 Pangenome Graph Construction

## Committed sample (`sample/`)

| Field | Value |
|---|---|
| File | `sample/pangenome_sample.txt` |
| Origin | **Synthetic** (hand-built by `scripts/make_synthetic.py`) |
| License | Public domain (CC0) — it is synthetic, no real genomic data |
| Size | < 1 KB |
| Nodes / paths | 12 nodes, 4 genome paths |

This tiny file lets `demo/run_demo` run **offline, with zero downloads**, a hard
requirement for every project (CLAUDE.md §8). It is **synthetic** and labelled as
such everywhere; it implies nothing clinical.

### File format

Lines beginning with `#` (or text after `#`) are comments and ignored. Otherwise
the file is whitespace-separated tokens in this order:

```
N P                      # N = number of nodes, P = number of genome paths
len_0 len_1 ... len_{N-1}# N node lengths, in base pairs
L  id0 id1 ... id{L-1}   # path 0: its length L, then L node ids   (repeat P times)
...
```

### What the sample encodes (a textbook variation graph)

- **12 nodes** (ids `0`–`11`) with round bp lengths so the printed coordinates are
  easy to read.
- **Nodes 0–9** form a shared left-to-right backbone.
- **Node 10** is the *alternate allele* of a **SNP bubble** (substitutes for node 4).
- **Node 11** is an *inserted segment* (an **insertion bubble** between 6 and 7).
- **4 paths** = 4 haplotypes: `ref` (straight backbone), `hap_snp` (10 for 4),
  `hap_ins` (insert 11), `hap_del` (delete 5). These are the three canonical
  variant types: substitution, insertion, deletion.

The layout should pull node 10 next to node 4 and node 11 between nodes 6 and 7 —
which is exactly what `demo/expected_output.txt` shows.

To regenerate (or scale up) the sample:

```bash
python scripts/make_synthetic.py
```

## Full dataset

Real pangenome graphs are built from genome assemblies with the **PGGB** pipeline
and then laid out with **ODGI**. `scripts/download_data.ps1` / `.sh` print
instructions and links (they never bypass any registration). Sources from the
catalog:

- **HPRC year-1 assemblies** — 94 human haplotypes — https://humanpangenome.org/
- **Ensembl** non-human pangenome data — https://www.ensembl.org/
- **Vertebrate Genomes Project** assemblies — https://vertebrategenomesproject.org/
- **NCBI RefSeq** complete genomes (bacterial pangenomes) —
  https://ftp.ncbi.nlm.nih.gov/refseq/

**License:** assembly licenses vary by source and may forbid redistribution, so
**no real data is committed** here — the sample is synthetic. Respect each
source's terms when you download. A real graph is typically distributed as **GFA**
(`.gfa`): to feed it to this teaching program you would convert its `S` (segment)
and `P`/`W` (path/walk) lines into the simple `N P` / lengths / paths format
above (a small script left as an exercise).

## Provenance & field meanings

- `len_k` — length of node `k` in **base pairs**. Sets the target separations
  (a path step over node `k` contributes `len_k` bp to the target distance).
- A path's `id` list — the ordered nodes a haplotype visits; adjacency along a
  path is the co-linearity the layout preserves.

Never imply clinical validity. Synthetic data is labelled synthetic everywhere it
appears (CLAUDE.md §8).
