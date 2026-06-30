# Data — 3.9 Phylogenetic Likelihood / Tree Inference

## Committed sample (`sample/phylo_sample.txt`)

| Field | Value |
|---|---|
| File | `sample/phylo_sample.txt` |
| Origin | **Synthetic** (simulated down a known tree by `scripts/make_synthetic.py`, seed 12345) |
| License | Public domain (CC0) — it is synthetic, no real organism is involved |
| Size | ~5.6 KB |
| Contents | 8 taxa × 600 DNA sites + 3 candidate trees + the K2P `kappa` |

This tiny file lets `demo/run_demo` run **offline, with zero downloads**, which is
a hard requirement for every project (CLAUDE.md §8). It is **synthetic** — DNA
evolved by a Monte-Carlo simulation under the Kimura-2-parameter (K2P) model down
a **known** 8-taxon tree, so the demo result is interpretable: the program should
recover that true tree by maximum likelihood. **Not real sequence data; not for
any biological or clinical conclusion.**

### File format (the loader grammar, see `src/reference_cpu.cpp`)

Lines beginning with `#` and blank lines are ignored, so the file is human-editable.

```
n_taxa  n_sites  n_trees  kappa          # header (kappa = K2P ts/tv rate ratio)
<name_0>  <sequence_0>                   # n_taxa rows; sequence length == n_sites
...
<tree_label>                             # one block per tree, repeated n_trees times
<n_internal>                             #   == n_taxa - 1 (rooted binary tree)
left right t_left t_right                #   n_internal node lines, POST-ORDER, root LAST
...
```

- **Bases** `A,C,G,T` map to state indices `0,1,2,3`; `-`, `N`, `?` and unknown
  characters become a **gap** (uninformative; conditional likelihood 1 for every
  state). The `A,G = 0,2` (purine) / `C,T = 1,3` (pyrimidine) ordering is what
  lets the model classify transitions vs. transversions (`src/felsenstein.h`).
- **Node lines** name an internal node's two children by index. A child index
  `< n_taxa` is a **leaf** (taxon); an index `>= n_taxa` is an **earlier**
  internal node (post-order guarantees children are computed before parents). The
  **last** node is the **root**. Branch lengths `t_*` are expected substitutions
  per site (≥ 0).

### What the three sample trees are

1. `..._true` — the topology the data was generated under: `((t0,t1),(t2,t3))`
   joined to `((t4,t5),(t6,t7))`. Maximum likelihood should pick this one.
2. `wrong_NNI1` — a wrong resolution of the **deep** split (a nearest-neighbour
   interchange around the root), included so ML has a plausible alternative to reject.
3. `wrong_NNI2` — a wrong pairing **inside** a clade.

Regenerate (e.g. a longer alignment) with:

```bash
python scripts/make_synthetic.py --n-sites 2000 --seed 7
```

## Full / real datasets

This project is a **didactic, reduced-scope** likelihood evaluator. To study it on
real curated alignments, see `scripts/download_data.ps1` / `.sh`, which print
instructions and links (they never bypass any registration). Real sources from the
catalog:

- **TreeBASE** — curated phylogenetic alignments and trees: https://www.treebase.org/
- **SILVA rRNA database** — large rRNA alignment for phylogenetics: https://www.arb-silva.de/
- **NCBI CDD** — conserved domain alignments: https://www.ncbi.nlm.nih.gov/Structure/cdd/cdd.shtml
- **Open Tree of Life** — aggregated phylogenetic data: https://opentreeoflife.github.io/

Real alignments arrive as **FASTA/PHYLIP/NEXUS** with Newick trees; converting them
to this project's compact text format (encode bases, write a post-order node list)
is left as an exercise in the README. **Respect each source's license**; none of
that data is redistributed here.

## Provenance & honesty

- The committed sample is **100% synthetic** and labeled as such everywhere.
- Branch lengths and `kappa` are illustrative, not fitted to any real organism.
- Nothing here implies biological or clinical validity (CLAUDE.md §8).
