# 3.13 — Pangenome Graph Alignment

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.13`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A **pangenome** replaces the single linear reference genome with a *graph* of many
genomes: nodes carry short DNA segments and directed edges spell out which segments
may follow which, so each path through the graph is one individual's haplotype. This
project aligns a sequencing **read** to such a graph with a **generalised local
Smith-Waterman** — the same dynamic-programming recurrence as ordinary pairwise
alignment, but generalised so a node's score column can inherit from *several*
predecessor nodes. The GPU fills one score matrix per node as an **anti-diagonal
wavefront**, processing nodes in topological order; the result recovers both the
best alignment score and the **path of alleles** the read took through the graph.
Everything runs on a tiny synthetic graph so you can read every number.

## What this computes & why the GPU helps

Pangenome graphs encode the genomic variation of an entire population as a sequence
graph (GFA format) rather than a single linear reference; aligning reads to this
graph involves generalised DP over a DAG of paths rather than a 1-D reference. The
vg toolkit's graph alignment applies a generalised Smith-Waterman on the graph DAG,
which is harder to parallelise than linear alignment due to irregular memory access.
A 2024 SC paper demonstrated GPU-accelerated pangenome *layout* achieving 57.3×
speedup over multi-core CPU for the ODGI layout algorithm by mapping node-force
computations to GPU threads; graph seeding via GBWT/r-index also benefits from
parallelised BWT operations.

**The parallel bottleneck.** The cost is the **DP matrix fill**: for a read of
length `m` against a graph whose nodes total `B` bases, we fill `O(m · B)` score
cells, and that dominates. Inside any one node the cells on a single anti-diagonal
`i+j = const` depend only on the two previous anti-diagonals, so they are mutually
independent — one GPU thread per cell. The "irregular" part (a node's first column
inheriting from predecessors) is localised to a tiny per-node host reduction, so the
heavy DP stays regular and coalesced. See [THEORY.md](THEORY.md) §GPU mapping.

## The algorithm in brief

- **Generalised DAG DP alignment** — local Smith-Waterman where a node's column 1
  draws its diagonal/left neighbours from the **last column of its predecessors**
  (max over predecessors); nodes processed in **topological order**.
- **Anti-diagonal wavefront** — each node's `(m+1)×(Lᵥ+1)` block is filled diagonal
  by diagonal, all cells of a diagonal in parallel (cf. flagship 3.01).
- **Shared `__host__ __device__` recurrence** — the per-cell formula lives in one
  function so the CPU reference and GPU kernel are bit-identical (PATTERNS.md §2).
- **Path traceback** — once the blocks are filled, a host traceback walks back
  through cells *and across node boundaries* to recover the allele path.

The catalog also lists production techniques this teaching version deliberately
does **not** implement (GBWT/r-index graph BWT seeding, force-directed GPU graph
*layout*, seqwish/wfmash graph construction) — see §Limitations and THEORY.md
"Where this sits in the real world".

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/pangenome-graph-alignment.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/pangenome-graph-alignment.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\pangenome-graph-alignment.sln /p:Configuration=Release /p:Platform=x64
```

This project links only the CUDA runtime (`cudart_static.lib`) — no extra CUDA
library is needed for the DP kernel.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/graph_sample.txt`, prints the best
score + recovered allele path + alignment, shows the GPU-vs-CPU agreement check,
and prints a timing line.

## Data

- **Sample (committed):** `data/sample/graph_sample.txt` — a tiny **synthetic**
  variation graph (13 nodes, 4 SNP bubbles) + one 54-base read, so the demo runs
  offline with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print the source links and
  the real `pggb`/`vg`/`odgi` pipeline (they never bypass any data-use agreement).
- **Provenance, format grammar & the embedded answer:** see
  [data/README.md](data/README.md).

Catalog dataset notes: Human Pangenome Reference Consortium (HPRC) — 94
haplotype-resolved assemblies (https://humanpangenome.org/); 1000 Genomes Project
GVCFs (https://www.internationalgenome.org/data); Ensembl Pangenome
(https://www.ensembl.org/); PGGB tutorial data (https://github.com/pangenome/pggb).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program fills the score blocks on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and asserts **every cell is identical** —
because the scoring is integer and both call the same `cell_score()`, the tolerance
is **exactly 0** (no floating-point slack; PATTERNS.md §4). The headline line

```
best path through graph = a0>s0ref>a1>s1alt>a2>s2ref>a3>s3alt>a4
```

recovers the exact allele path the synthetic read was built to follow (`ref` on
even bubbles, `alt` on odd), validating the science and not just CPU==GPU agreement.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the graph+read, runs CPU + GPU, verifies
   cell-for-cell, reports the path/alignment.
2. [`src/reference_cpu.h`](src/reference_cpu.h) — the data model, the shared
   `cell_score()` recurrence, and the graph/DP structs (read this for the "what").
3. [`src/kernels.cuh`](src/kernels.cuh) — the per-node wavefront idea + the kernel
   interface.
4. [`src/kernels.cu`](src/kernels.cu) — the anti-diagonal kernel and the host sweep
   that walks nodes in topological order.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial fill, the
   loader, and the cross-node traceback.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, I/O helpers.

## Prior art & further reading

- **vg** (<https://github.com/vgteam/vg>) — the comprehensive variation-graph
  toolkit; its `gssw`/`GSSWAligner` is the production version of the graph SW here.
  Study how it handles cyclic graphs and affine gaps.
- **PGGB** (<https://github.com/pangenome/pggb>) — the Pangenome Graph Builder
  pipeline (seqwish + smoothxg + odgi); learn how a graph is *constructed* from
  all-vs-all alignments before anything is aligned *to* it.
- **ODGI** (<https://github.com/pangenome/odgi>) — graph manipulation and the
  GPU-accelerated **layout** in the SC2024 paper. Learn the difference between graph
  *layout* (the paper's 57× win) and graph *alignment* (this project).
- **Rapid GPU-based pangenome layout** (SC2024,
  <https://www.csl.cornell.edu/~zhiruz/pdfs/pangenome-layout-sc2024.pdf>) — the
  force-directed-on-GPU reference; read it for how irregular graph work is mapped to
  threads.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Anti-diagonal wavefront DP, generalised across a DAG** (PATTERNS.md §1, the
`3.01` Smith-Waterman row). One CUDA thread fills one DP cell; one kernel launch
fills one anti-diagonal of one node's block; the host loop sweeps nodes in
topological order and reduces predecessor boundary columns between launches. The
catalog's headline pattern (force-directed *layout* with Barnes-Hut, graph BFS for
BWT, Thrust sort, cuSPARSE traversal) targets the *other* GPU-accelerated stage of
the pangenome pipeline and is described — not implemented — in THEORY.md.

## Exercises

1. **Affine gaps.** Replace the single linear `GAP` with the Gotoh affine model
   (separate gap-open and gap-extend, three matrices `H/E/F`). How does the
   recurrence and the per-node block change?
2. **Batch many reads.** Give each read its own CUDA block (one block per read,
   threads over the wavefront) so the GPU's launch overhead is amortised — the
   regime where it actually beats the CPU (THEORY.md §7, PATTERNS.md §7).
3. **Real GFA in.** Write a converter from a small PGGB/`vg` GFA to this project's
   `N`/`E` text format (one `S`→`N`, one `L`→`E`), topologically sorting first with
   `odgi sort`. Align a real read and inspect the path.
4. **Cyclic graphs.** Our loader requires a DAG. Sketch what breaks for a cyclic
   graph (the topological order no longer exists) and how `vg` handles it
   (dagify / unrolling).
5. **FP64 / scoring matrices.** Swap integer scoring for a BLOSUM/transition-aware
   substitution matrix in constant memory; confirm CPU==GPU still holds and discuss
   why determinism is now subtler.

## Limitations & honesty

- **Reduced-scope teaching version.** This implements the *alignment DP* on a
  **DAG** with **linear** gaps and **integer** scores. Production graph aligners
  (vg/gssw) handle **cyclic** graphs, **affine** gaps, seeding via GBWT/r-index,
  base-quality-aware scoring, and mapping-quality estimation — none of which are
  here. The catalog's force-directed GPU *layout* (the SC2024 57× result) is a
  **different** computation and is described in THEORY.md, not implemented.
- **Synthetic data.** The committed graph and read are generated locally with a
  fixed seed and labeled **synthetic** everywhere. No real or patient-derived
  genomic data is included; nothing here is validated for any clinical use.
- **Timing is a teaching artifact.** On this tiny graph the GPU issues many small
  per-diagonal launches and is *slower* than the trivially-fast CPU fill — exactly
  the launch-bound regime PATTERNS.md §7 warns about. The wavefront pays off on long
  reads, large bubbles, and batched reads; the printed milliseconds are not a
  benchmark claim.
- **One traceback.** We trace a single best path with deterministic tie-breaking;
  real tools may report multiple co-optimal paths or a richer GAF/GAM record.
