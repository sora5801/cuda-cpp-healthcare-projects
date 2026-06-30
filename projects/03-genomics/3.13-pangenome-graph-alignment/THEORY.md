# THEORY — 3.13 Pangenome Graph Alignment

> The deep didactic explanation (the "why"). Written for a sharp student who knows
> C++ but is new to CUDA and new to this domain. See [README.md](README.md) for the
> quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

For two decades, "the human genome" meant **one** linear reference string (GRCh38).
Every read from every patient was aligned to that single sequence. But no two humans
are identical, and the reference is mostly one person's DNA — so variants common in
under-represented populations look like "errors" against it (**reference bias**).

A **pangenome** fixes this by representing many genomes at once as a **sequence
graph**. Take a locus where the population carries either a `C` or a `G`:

```
            ┌──▶ [C] ──┐
  …ACGT ────┤          ├──▶ TTGA…           two parallel allele nodes = a "bubble"
            └──▶ [G] ──┘                    each path through the graph = a haplotype
```

Shared sequence is stored once (a node every haplotype traverses); where genomes
differ, the graph **branches** into a bubble and **rejoins**. A whole chromosome
becomes a long chain of shared "anchor" segments separated by bubbles (SNPs, indels,
larger structural variants). This is the data model of the **Human Pangenome
Reference Consortium** graphs and the GFA files produced by **PGGB**.

The computational question this project answers: **given a sequencing read, what is
its best alignment to *any path* through the graph, and which path is it?** Recovering
the path tells you which alleles the read supports — the input to variant calling and
genotyping without reference bias.

> **Layout vs. alignment.** The catalog highlights a 2024 SC paper that GPU-
> accelerates pangenome **layout** (assigning 2-D coordinates to nodes for
> visualisation, a force-directed *physics* simulation) by 57×. That is a different
> computation from **alignment** (placing a read onto the graph). This project teaches
> *alignment*; §7 explains how the layout problem maps to the GPU too.

## 2. The math

**Inputs.**
- A query read `q = q₁ q₂ … q_m`, each `qᵢ ∈ {A,C,G,T}` (encoded `0..3`), length `m`.
- A directed **acyclic** graph `G = (V, E)`. Node `v ∈ V` carries a segment
  `t^v = t^v₁ … t^v_{Lᵥ}` of length `Lᵥ`. An edge `(u → v) ∈ E` means `u`'s segment
  may be immediately followed by `v`'s.

**Scoring.** Linear-gap, integer:
- match `σ(a,b) = +2` if `a = b`, else `mismatch = −1`;
- gap penalty `γ = −2` per inserted/deleted base.

**The DP.** Define one score matrix per node. For node `v`, query row `i ∈ [0,m]`,
node column `j ∈ [0,Lᵥ]`:

```
H^v[i][j] = best local-alignment score that consumes q₁…qᵢ and ends exactly at
            column j of node v's segment.
```

Initialise `H^v[0][j] = 0` and `H^v[i][0] = 0` (the boundary row/column). The
**Smith-Waterman local recurrence** for `i ≥ 1, j ≥ 1` is

```
                ⎧ 0                                   (start a fresh local alignment)
H^v[i][j] = max ⎨ Diag(v,i,j) + σ(qᵢ, t^v_j)          (align qᵢ with the graph base)
                ⎨ Up(v,i,j)   + γ                      (gap on the graph path)
                ⎩ Left(v,i,j) + γ                      (gap in the query)
```

The **only** difference from linear SW is the three neighbours `Diag/Up/Left`. For an
interior column `j ≥ 2` they are inside node `v`:

```
Diag = H^v[i−1][j−1],   Up = H^v[i−1][j],   Left = H^v[i][j−1].
```

For the **first content column `j = 1`**, the left/diagonal neighbours live in the
**last column** of `v`'s predecessors, and we take the best over all of them:

```
Diag(v,i,1) = max over u with (u→v) of  H^u[i−1][Lᵤ]      (0 if v has no predecessor)
Left(v,i,1) = max over u with (u→v) of  H^u[i  ][Lᵤ]      (0 if v has no predecessor)
Up(v,i,1)   = H^v[i−1][1]                                 (still inside v)
```

A node with **no** predecessor is a graph **source**: its `Diag/Left` default to `0`,
so a local alignment may begin fresh at the start of any source. The reported answer
is `S* = max over (v,i,j) of H^v[i][j]`, and the alignment is recovered by tracing
back from the cell attaining `S*` until a `0` is reached (§6).

This is exactly the *partial-order alignment* / graph-Smith-Waterman that `vg`'s
`gssw` implements (minus affine gaps and cycles — §7).

## 3. The algorithm

```
load graph + read
layout per-node blocks H^v of size (m+1)×(Lᵥ+1), concatenated flat
for v in topological order:                 # predecessors already finished
    build Diag/Left boundary columns for v  # max over predecessor last columns
    for i in 1..m:                          # CPU reference: plain row-major
        for j in 1..Lᵥ:
            H^v[i][j] = cell_score(Diag, Up, Left, qᵢ == t^v_j)
S* = max cell over all blocks
traceback from S* across cells and node boundaries → score, path, alignment
```

**Why topological order works.** In a DAG, the `first_column` of node `v` only ever
reads *predecessor* blocks `u`, and every predecessor has a smaller topological rank.
Processing nodes in ascending rank guarantees all dependencies are final before we
touch `v`. Our loader assigns ranks in file-declaration order and **rejects** any
backward edge, so declaration order *is* a topological order (`load_problem`).

**Complexity.** Let `B = Σ Lᵥ` (total graph bases) and `m` the read length.
- **Work:** `Θ(m · B)` cells, each `O(1)` plus, at first columns, an `O(deg⁻(v))`
  max — total `Θ(m·B + m·|E|)`. For the sample: `m=54`, `B=78`, a few thousand cells.
- **Serial depth (critical path):** the cells along the longest anti-diagonal chain;
  within a node it is `O(m + Lᵥ)` diagonals, summed along the graph's longest path.
- **Memory:** `Θ(m · B)` ints if we keep all blocks (we do, to enable traceback). A
  production aligner keeps only two diagonals per node and stores traceback pointers.

The **arithmetic intensity** is low (a few integer ops per memory access), so the
fill is memory-bound — exactly why the regular, coalesced layout in §4 matters.

## 4. The GPU mapping

The DP has the classic SW dependency: `H[i][j]` needs its top, left, and top-left
neighbours, which *looks* serial. The **wavefront** trick (flagship 3.01): every cell
on one **anti-diagonal** `d = i + j` depends only on diagonals `d−1` and `d−2`, so the
cells of diagonal `d` are mutually independent → fill them all in parallel.

We apply the wavefront **per node**, walking nodes in topological order:

```
host loop over nodes v (topological order):
  (1) reduce predecessor last-columns -> diag_in[i], left_in[i]   (tiny host pass)
  (2) upload diag_in/left_in
  (3) for d = 2 .. m+Lᵥ:                       # one launch per anti-diagonal
        graph_sw_diagonal_kernel<<<blocks, 128>>>(...)   # one thread per cell on d
  (4) copy v's finished block back to the host mirror   # so successors can read it
```

**Thread-to-data mapping.** On diagonal `d`, the valid rows are
`i ∈ [max(1, d−Lᵥ) .. min(m, d−1)]`. Thread `k` owns `i = i_lo + k`, `j = d − i`. It
reads three neighbours (all on diagonals `d−1`, `d−2`, already finalised by earlier
launches) and writes `H^v[i][j]`. No two threads in a launch write the same cell and
none reads a cell another writes **this** launch → **no atomics, no `__syncthreads`**.

```
   node v block          anti-diagonals d = i+j are the parallel frontiers
 j: 0  1  2  3 (=Lᵥ)        d=2 d=3 d=4 …       a thread on d reads only
i0 [ boundary col ]                              cells on d−1 and d−2 (final),
i1 |  ·  ·  ·  · |       diag_in[i]/left_in[i]   plus the precomputed boundary
i2 |  ·  ·  ·  · |       feed column j=1         vectors for column j=1.
i3 |  ·  ·  ·  · |
```

**Why the boundary reduction is on the host.** The cross-node max (the "irregular
memory access" the catalog warns about) is the only part that chases graph pointers.
We isolate it into a small `Θ(m·deg⁻(v))` host pass between launches (`build_boundaries`)
and hand the kernel two dense vectors `diag_in/left_in`. The kernel then does only
**regular, coalesced** array indexing — the heavy DP stays GPU-friendly. (A fully on-
device version would gather predecessor columns with a second kernel; we keep it on
the host because it makes the dependency structure unmistakable, the teaching goal.)

**Launch configuration.** `THREADS_PER_BLOCK = 128` — a multiple of the 32-lane warp,
ample for the short diagonals of a teaching-sized block; `blocks = ⌈count/128⌉`. No
shared memory or constant memory is needed here (the working set per cell is three
ints); the natural next optimisation is intra-node shared-memory tiling (Exercise).

**No CUDA library is used.** The DP is a custom kernel; we link only `cudart_static`.
The catalog's `Thrust`/`cuSPARSE`/Barnes-Hut belong to the *layout* pipeline (§7), not
this alignment kernel — calling them here would be a black box for no benefit.

## 5. Numerical considerations

- **Precision: pure integers.** Scores are `int`. Integer addition and `max` are
  associative and exact, so there is **no rounding and no floating-point reordering
  risk** — unlike the float-atomic hazard in PATTERNS.md §3. This is why we can verify
  to tolerance **exactly 0**.
- **Determinism.** stdout is byte-identical every run: the same nodes in the same
  topological order, the same cells, and `cell_score()` shared between CPU and GPU. The
  per-node block copy-back (`cudaMemcpy` of a contiguous slice) does not reorder
  anything. Run-varying numbers (kernel ms) go to **stderr** only.
- **No races.** Within a diagonal launch, each thread writes a distinct cell and reads
  only already-final cells → no read-after-write hazard, hence no atomics. The
  `CUDA_CHECK_LAST` after the sweep synchronises and surfaces any launch/exec error.
- **Tie-breaking is fixed.** The max scan picks the first cell in `(v, i, j)` order;
  traceback prefers `diagonal > up > left` and the first predecessor in CSR order — so
  the recovered path is unique and reproducible even when several paths co-score.

## 6. How we verify correctness

Two independent checks, both in `main.cu`:

1. **CPU == GPU, cell for cell, tolerance 0.** `graph_sw_cpu` fills every block with a
   plain double loop; `graph_sw_gpu` fills the same blocks with the wavefront. Both call
   the shared `cell_score()` (PATTERNS.md §2), so identical integer math must produce
   **identical** blocks. We compare all `Σ (m+1)(Lᵥ+1)` cells and require
   `mismatches == 0`. An independent serial implementation agreeing bit-for-bit with the
   parallel one is strong evidence the parallelisation introduced no bug (a transposed
   index or a missed boundary would change some cell).
2. **The science: path recovery.** The synthetic read is built to follow a *known*
   allele path (`ref` on even bubbles, `alt` on odd; `data/README.md`). The traceback
   recovers exactly `a0>s0ref>a1>s1alt>a2>s2ref>a3>s3alt>a4` at 92.6 % identity — so we
   validate not just "CPU==GPU" but "the aligner found the right haplotype" (PATTERNS.md
   §4's stronger analytic check).

**Edge cases exercised by the sample:** source nodes (no predecessor → `0` boundary),
fan-in at anchors (two alleles → one anchor, so the `max` over predecessors actually
fires), node-boundary hops during traceback, and the ragged last block in each
diagonal launch.

## 7. Where this sits in the real world

Production tooling does much more than this teaching kernel:

- **vg / gssw** align to **cyclic** graphs (loops from inversions/duplications) via
  *dagify*/unrolling, use **affine** gaps (Gotoh: separate open/extend with `H/E/F`
  matrices), incorporate **base qualities**, and emit mapping quality + GAF/GAM records.
  Our DAG-only, linear-gap, integer version is the readable core inside all of that.
- **Seeding.** Real aligners never run full DP genome-wide. They **seed** with exact
  matches found through a graph **BWT** (GBWT / r-index / GBWTgraph), then run graph SW
  only in small windows around seeds — the `vg giraffe` strategy. The catalog's "graph
  BFS for BWT construction" and "parallelised BWT operations" refer to building/querying
  those indices on the GPU.
- **Graph construction.** Before you align *to* a graph you must *build* it: **seqwish**
  induces a graph from all-vs-all alignments, **wfmash** does the wavefront all-to-all
  mapping, and **PGGB** orchestrates the pipeline. Different problem, upstream of us.
- **The SC2024 57× result is *layout*, not alignment.** **ODGI**'s `path-guided
  stochastic gradient descent` layout treats nodes as particles under spring forces and
  is parallelised on the GPU (one thread per node-force, **Barnes-Hut** to approximate
  far-field forces, **Thrust** to sort node positions, **cuSPARSE** for sparse adjacency).
  That is the "force-directed GPU particles" pattern in the catalog — it produces the 2-D
  picture of the graph, and is a great follow-on project (the `5.01`/`9.02`-style
  per-thread-particle pattern), but it is orthogonal to placing a read on the graph.

In short: this project is the **didactic heart** (graph DP) of the alignment step;
the catalog's headline GPU pattern is the **visualisation/layout** step. THEORY.md and
the exercises point at both so the learner sees the whole pangenome pipeline.

---

## References

- Garrison et al., *Variation graph toolkit (vg)* — Nat. Biotechnol. 2018, and the
  `gssw` library: the production graph Smith-Waterman this project distils.
  <https://github.com/vgteam/vg>
- Garrison & Guarracino et al., **PGGB** — building pangenome graphs (seqwish + smoothxg
  + odgi). <https://github.com/pangenome/pggb>
- **ODGI** and the SC2024 *Rapid GPU-based pangenome graph layout* paper — the 57×
  force-directed layout; read for irregular-graph→GPU mapping.
  <https://www.csl.cornell.edu/~zhiruz/pdfs/pangenome-layout-sc2024.pdf>
- Lee, Grasso & Sharlow (2002), *Multiple sequence alignment using partial order graphs*
  — the original POA recurrence that graph SW generalises.
- Gotoh (1982), *An improved algorithm for matching biological sequences* — the affine-
  gap model named in Exercise 1.
- Flagship **3.01** (this repo) — linear Smith-Waterman; the wavefront pattern reused
  here, generalised across a DAG.
