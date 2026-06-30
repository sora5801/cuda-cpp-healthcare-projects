# 3.30 — Pangenome Graph Construction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.30`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A **pangenome graph** represents many genomes as one sequence graph: shared
segments become **nodes**, and each genome is a **path** (a walk) through those
nodes. Where genomes agree they share nodes; where they differ — SNPs,
insertions, deletions — the graph forms "bubbles." This project builds such a
graph from a few synthetic haplotypes and then solves the step that dominates
real pangenome pipelines like **ODGI/PGGB**: laying the graph out in **1-D** so
that nodes which are close along a genome path also end up close on the axis. We
do this with **SMACOF stress majorization** (the deterministic cousin of ODGI's
path-guided SGD layout), evaluated on the GPU with one thread per layout
constraint and a deterministic integer-atomic reduction. The result is a node
ordering — exactly what `odgi sort` produces to linearise a graph — and it
correctly tucks each variant node next to its genomic neighbours.

## What this computes & why the GPU helps

Building a pangenome variation graph from dozens to thousands of genome
assemblies requires all-to-all pairwise alignment and progressive normalisation.
At the scale of the HPRC 94-haplotype human pangenome, alignment and **graph
layout** are the dominant costs. ODGI's GPU layout reports a **57.3× speed-up**
over multi-core CPU: force-directed node positioning is a particle-physics-style
simulation that is highly GPU-amenable.

**The parallel bottleneck:** the **1-D layout**. The layout minimises a weighted
*stress* over a large set of node-pair constraints ("term" `(i, j, d, w)`: nodes
`i, j` should sit `d` base pairs apart, with weight `w`). Real graphs have
millions of nodes and **billions** of terms; evaluating every term's contribution
to its two endpoints each sweep is the cost. Each term is **independent**, so we
give **one GPU thread per term** and scatter its contribution into per-node
accumulators — the classic *parallel-evaluate + atomic-reduce* pattern.

## The algorithm in brief

- **Graph model** — nodes (segments with a base-pair length) + paths (genome
  walks). Variant types appear as bubbles: a substituted node, an inserted node,
  a deleted node.
- **Term construction** — for each genome path, every pair of nodes within a few
  steps becomes a constraint with target distance = the base pairs between them
  and weight `1/d²` (short distances matter most). Duplicate pairs are merged to
  the tightest constraint deterministically.
- **SMACOF / Guttman transform** — repeatedly move each node to the weighted
  average of where its terms want it. This is **monotone** (stress never
  increases), so there is **no learning rate** to tune and **no divergence**.
- **GPU mapping** — per sweep: a *scatter* kernel (thread per term, atomic-adds a
  fixed-point numerator + weight onto both endpoints) and an *apply* kernel
  (thread per node, `x = numerator/denominator`).
- **Determinism** — accumulate in **fixed-point integers** so the atomic
  reduction commutes ⇒ GPU result is reproducible and equals the CPU bit-for-bit.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/pangenome-graph-construction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/pangenome-graph-construction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\pangenome-graph-construction.sln /p:Configuration=Release /p:Platform=x64
```

No extra CUDA libraries are linked — the kernels are hand-written, so only the
CUDA runtime (`cudart_static.lib`, added by the scaffold) is needed.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/pangenome_sample.txt`, prints the
layout, shows the GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/pangenome_sample.txt` — a tiny, **synthetic**
  12-node, 4-path pangenome with one SNP, one insertion, and one deletion bubble,
  so the demo runs offline and the recovered layout is interpretable.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print instructions for
  obtaining real assemblies and building a graph with PGGB (they never bypass any
  registration).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: HPRC year-1 assemblies — 94 haplotypes, human pangenome
(https://humanpangenome.org/); Ensembl non-human pangenome data
(https://www.ensembl.org/); Vertebrate Genomes Project assemblies
(https://vertebrategenomesproject.org/); NCBI RefSeq complete genomes for
bacterial pangenomes (https://ftp.ncbi.nlm.nih.gov/refseq/).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program computes the layout on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and asserts they agree — same node positions
(to `1e-9` bp) and the **same 1-D node order**. That agreement is the correctness
guarantee. The interesting result to read: the SNP-alternate node `10` lands next
to reference node `4`, and the inserted node `11` lands between nodes `6` and `7`
— the layout untangles the bubbles. The headline order is
`0 1 2 3 10 4 5 6 11 7 8 9`, and the stress falls from ~2.23M to ~18.9k.

## Code tour

Read in this order:

1. [`src/layout.h`](src/layout.h) — the shared `__host__ __device__` per-term
   physics (Guttman transform) + the fixed-point determinism trick.
2. [`src/main.cu`](src/main.cu) — loads the graph, builds the problem, runs CPU +
   GPU, verifies, reports.
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`reference_cpu.cpp`](src/reference_cpu.cpp)
   — the graph model, term construction, and the trusted serial SMACOF baseline.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the parallel pattern.
5. [`src/kernels.cu`](src/kernels.cu) — the scatter/apply kernels and host wrapper.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **PGGB** (https://github.com/pangenome/pggb) — the pangenome graph builder
  pipeline; learn the wfmash → seqwish → smoothxg → ODGI flow this project models
  the layout step of.
- **ODGI** (https://github.com/pangenome/odgi) — GPU-accelerated graph layout and
  operations; the source of the 57.3× layout speed-up and the `odgi sort -p Ygs`
  path-SGD algorithm we implement deterministically.
- **wfmash** (https://github.com/waveygang/wfmash) — WFA-based all-to-all aligner;
  read it to see the anti-diagonal wavefront alignment that seeds the graph.
- **vg** (https://github.com/vgteam/vg) — comprehensive graph alignment toolkit;
  the broader ecosystem (GFA, GBWT, indexing) around pangenome graphs.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Parallel **term evaluation + deterministic atomic reduction** (PATTERNS.md §1,
the k-means/`11.09` row, plus the §2 shared `__host__ __device__` core and the §3
fixed-point determinism rule). One thread per layout term computes the shared
Guttman contribution and atomic-adds a fixed-point numerator/denominator onto its
two endpoint nodes; a second kernel divides per node. The catalog's full vision
(custom WFA alignment kernels, multi-GPU all-to-all) is described under
"Where this sits in the real world" in THEORY.md; this project implements the
GPU-amenable **layout** core to the Definition of Done.

## Exercises

1. **Bigger bubbles.** Edit `scripts/make_synthetic.py` to add a second SNP and an
   inverted segment, regenerate the sample, and re-capture `expected_output.txt`.
   Does the node order still place each variant beside its neighbours?
2. **Sweep `hops`.** Change `HOPS` in `main.cu` from 3 to 1 and to 6. How does the
   term count and the final stress change? (More hops = stiffer, more long-range
   constraints.)
3. **Convergence curve.** Print the stress every 10 sweeps (to stderr) and watch
   SMACOF decrease monotonically — then prove to yourself it never increases.
4. **Quantiser resolution.** Halve `LO_SCALE` in `layout.h` and find the point
   where GPU and CPU stop matching exactly — a hands-on lesson in fixed-point
   precision vs. the weight range.
5. **Stochastic SGD.** Implement ODGI's *stochastic* path-SGD (one random term per
   step with a decaying learning rate) and compare its layout to this
   deterministic full-batch SMACOF.

## Limitations & honesty

- **Reduced scope.** The full catalog project is a whole pipeline (WFA all-to-all
  alignment → seqwish graph induction → smoothxg normalisation → ODGI layout). We
  implement the **1-D layout** step — the most GPU-amenable, self-contained, and
  verifiable piece — and *describe* the rest in THEORY.md. The input graph here is
  given, not aligned from sequences.
- **Synthetic data.** The 12-node sample is hand-built and labelled synthetic
  everywhere; it is engineered so the recovered layout is obviously correct. It is
  not real genomic data and implies nothing clinical.
- **SMACOF vs. SGD.** We use deterministic full-batch SMACOF so CPU and GPU match
  bit-for-bit; production ODGI uses *stochastic* path-SGD, which is faster per
  sweep but order-dependent (and therefore not bit-reproducible).
- **Teaching timing.** The graph is tiny, so the GPU is launch-bound and slower
  than the CPU here; the timing line is a teaching artifact, not a benchmark. The
  GPU's advantage appears only at ODGI's real scale (millions of nodes, billions
  of terms).
