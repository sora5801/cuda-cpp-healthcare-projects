# THEORY — 3.30 Pangenome Graph Construction

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

### What a pangenome graph is

For decades, genomics used a single **linear reference genome** (e.g. GRCh38).
Every new sample was compared against that one sequence. The problem: any single
genome is biased — it cannot represent the variation present across a population,
and reads from sequences absent in the reference simply fail to map ("reference
bias"). A **pangenome** fixes this by representing *many* genomes at once.

The dominant data structure is the **variation graph** (a.k.a. sequence graph):

- **Nodes** are stretches of DNA sequence (segments). In our model each node has a
  length in base pairs (bp).
- **Edges** connect nodes that are adjacent in at least one genome.
- **Paths** are the genomes themselves: a path is an ordered walk
  `n₀ → n₁ → n₂ → …` listing the nodes a particular haplotype visits.

Where genomes agree, their paths share nodes. Where they differ, the graph
"bubbles":

```
            ┌── 4  (reference allele) ──┐
... 2 ─ 3 ──┤                           ├── 5 ─ 6 ...     SNP bubble
            └── 10 (alternate allele) ──┘

... 6 ──────────── 7 ...                         (reference)
... 6 ── 11 ─────── 7 ...                         insertion bubble
... 4 ──────────── 6 ...                         deletion bubble (skips 5)
```

The three textbook variant types — **substitution/SNP**, **insertion**,
**deletion** — are exactly the three bubbles in our synthetic sample
(`scripts/make_synthetic.py`).

### Why "construction" needs a layout

Building the graph from raw assemblies is a pipeline (PGGB): all-to-all
**alignment** (wfmash), graph **induction** from the alignments (seqwish), and
**normalisation** (smoothxg). But a freshly-built graph has no natural order on
its nodes — node ids are arbitrary. To *use* the graph (visualise it, index it,
compute coordinates, export a sorted GFA) you must **order the nodes in 1-D** so
that genomically co-linear nodes are adjacent. That is `odgi sort`/`odgi layout`,
and it is the step this project implements and accelerates. ODGI reports a
**57.3× GPU speed-up** for exactly this layout.

## 2. The math

### The stress objective

The 1-D layout assigns each node `i` a coordinate `xᵢ ∈ ℝ` (base pairs along the
axis). We want the **1-D separation** of two nodes to match their **genomic
distance** along the genomes. Encode each desired separation as a **term**:

> term `(i, j, dᵢⱼ, wᵢⱼ)` — nodes `i` and `j` should be `dᵢⱼ` bp apart, enforced
> with weight `wᵢⱼ ≥ 0`.

The layout minimises the **weighted stress** (the classic metric-MDS objective):

```
E(x) = Σ_terms  wᵢⱼ · ( |xᵢ − xⱼ| − dᵢⱼ )²
```

- `|xᵢ − xⱼ|` is the realised 1-D separation; `dᵢⱼ` the target. The squared
  residual penalises being too close or too far.
- **Targets `dᵢⱼ`** come from the graph: for two nodes a few steps apart on a
  genome path, `dᵢⱼ` is the number of base pairs you travel between them (the sum
  of the intervening node lengths).
- **Weights `wᵢⱼ = 1/dᵢⱼ²`** make *short* distances dominate — adjacent nodes must
  be placed precisely, while far-apart nodes may stretch. This is ODGI's
  weighting (and standard in graph drawing, Gansner et al. 2004).

`E(x)` is invariant to a global shift (`x → x + c`) and reflection, so the
solution is defined up to those gauges; we anchor the printout so the leftmost
node sits at 0.

### Why minimising this orders the graph

If every adjacency along a genome is satisfied (`|xᵢ − xⱼ| ≈ dᵢⱼ`), then walking a
genome path moves you monotonically along the axis, and sorting nodes by `xᵢ`
recovers the genome order — with variant nodes slotted next to their reference
neighbours. The **node order** (the permutation that sorts `x`) is the deliverable.

## 3. The algorithm

### Term construction

For each genome path and each start index `s`, look ahead up to `hops` steps. The
pair `(path[s], path[s+h])` gets a term with target `d` = bp travelled from the
start of `path[s]` to the start of `path[s+h]` (sum of intervening node lengths)
and weight `1/d²`. The same unordered pair can arise from several paths; we keep
the **smallest** target (tightest constraint) in an ordered map, so the emitted
term array is deterministic regardless of path order. Complexity: `O(P · L · hops)`
to build, where `P` = paths, `L` = path length.

### SMACOF (stress majorization) — and why not plain gradient descent

A naïve "step each node down the stress gradient" needs a hand-tuned step size:
in a **full-batch** update (a node feels the sum of *all* its terms at once), too
large a step overshoots and **diverges** — we saw exactly this during
development. **Stress majorization / SMACOF** sidesteps the problem. It replaces
`E(x)` with a quadratic function that *upper-bounds* it and *touches* it at the
current `x`, then jumps to that bound's exact minimiser. Because the bound is
above `E` and they touch, the jump can only *decrease* `E` — SMACOF is
**monotone** and needs **no learning rate**.

For weighted 1-D MDS the minimiser is a closed form, the **Guttman transform**:

```
                Σⱼ wᵢⱼ · ( xⱼ + dᵢⱼ · sign(xᵢ − xⱼ) )
   xᵢ_new  =   ───────────────────────────────────────
                              Σⱼ wᵢⱼ
```

Each node moves to the **weighted average** of where each of its terms wants it: a
neighbour `xⱼ` offset by the target distance `dᵢⱼ`, placed on `i`'s current side
(`sign(xᵢ − xⱼ)`). Each sweep is a **Jacobi** update: compute all new positions
from the sweep's starting positions, then apply them together.

**Complexity.** Serial: `O(T)` per sweep over `T` terms, `O(T · iters)` total. The
work is two passes — *scatter* each term onto its endpoints, then *divide* per
node — so it is `O(T)` work and `O(1)` depth per sweep in the parallel model.

## 4. The GPU mapping

The pattern is **parallel term evaluation + deterministic atomic reduction**
(PATTERNS.md §1, the same shape as `11.09` k-means; §2 shared core; §3 fixed-point
determinism). Per sweep we launch two kernels:

```
            terms[0..T)                      one thread per TERM
        ┌──────────────────┐        scatter_kernel
 thread │ (i, j, d, w)     │  ──►  ni = w·(xj + d·sign(xi−xj))   atomicAdd → num[i]
   t →  │ read x[i], x[j]  │       nj = w·(xi + d·sign(xj−xi))   atomicAdd → num[j]
        └──────────────────┘                                     atomicAdd → den[i], den[j]
                                            (collisions on shared nodes → atomics)
            ─────────────── launch boundary = global barrier ───────────────
            nodes[0..N)                      one thread per NODE
        ┌──────────────────┐        apply_kernel
 thread │ x[k] = num[k]    │
   k →  │        / den[k]  │       (independent; no atomics)
        └──────────────────┘
```

- **Thread-to-data map.** `scatter_kernel`: thread `t = blockIdx.x·blockDim.x +
  threadIdx.x` owns term `t`. `apply_kernel`: thread `k` owns node `k`. Both guard
  the ragged last block with `if (idx >= count) return;`.
- **Launch config.** 256 threads/block (a warp multiple; 8 warps to hide latency;
  good occupancy on sm_75–sm_89). Grids are `ceil(T/256)` and `ceil(N/256)`.
- **Memory hierarchy.** Positions `x`, the term array, and the two accumulators
  live in **global memory**. The term array is read once per thread (streamed,
  coalesced within a warp). The accumulators are the scatter target: many terms
  touch the same node, so writes **collide** → `atomicAdd`. We use no shared
  memory — the collisions are spread across all nodes and integer atomics are
  cheap; a production kernel might privatise per-block partial sums in shared
  memory for very high-degree nodes.
- **Why two kernels.** Every node's `num`/`den` must be *fully* reduced before any
  node moves. A kernel launch is a global synchronisation point, so splitting
  scatter and apply gives us that barrier for free — and makes the in-place `x`
  update a correct Jacobi step.
- **No CUDA library.** The kernels are hand-written. We deliberately do **not**
  reach for Thrust here: the lesson is the atomic scatter-reduction and the
  fixed-point determinism trick, which a library call would hide.

## 5. Numerical considerations

- **Precision.** Positions and the per-term math are **double** (FP64). Base-pair
  coordinates reach thousands and the division `num/den` wants headroom; FP64 is
  the safe teaching choice.
- **The determinism problem.** Many threads `atomicAdd` into the same node
  accumulator. **Floating-point addition is not associative**, so a *float*
  atomic sum depends on the (nondeterministic) order threads finish → the GPU
  result would vary run-to-run and would not match the serial CPU.
- **The fix: fixed-point integers** (PATTERNS.md §3, same as `5.01`/`11.09`). We
  quantise each contribution to an integer count of "quanta" (`LO_SCALE = 2²⁰`)
  and `atomicAdd` those. **Integer addition commutes**, so the reduction is
  order-independent ⇒ reproducible *and* bit-identical to the CPU.
- **Signed atomics.** Contributions are signed, but CUDA's `atomicAdd` only
  supports `unsigned long long`. We store the **two's-complement bit pattern** of
  a signed `long long` in an unsigned accumulator: unsigned (modular 2⁶⁴) addition
  is bit-identical to signed addition, so the round-trip is exact (see
  `ll_to_ull`/`ull_to_ll` in `kernels.cu`).
- **Weight normalisation.** Raw weights `1/d²` span a wide range (~`1e-6` for the
  longest distances). At `2²⁰` quanta, `1e-6` rounds to ~1 quantum — far too
  coarse for the `num/den` division. Since the Guttman ratio is invariant to a
  global weight scale, we **normalise so the largest weight is 1.0** in
  `build_problem`; the smallest weight then resolves to thousands of quanta. This
  changes nothing mathematically but everything numerically.

## 6. How we verify correctness

- **Independent CPU reference.** `reference_cpu.cpp` runs the *identical* algorithm
  serially: same term order, same shared `LO_term_numerator` math (from
  `layout.h`), same fixed-point accumulation. Because the GPU reuses the same
  inline functions and the integer reduction commutes, the two agree **exactly**.
- **Tolerance.** We allow `1e-9` bp on positions — effectively exact (the
  fixed-point reduction is identical on both sides; the slack only guards against
  compiler reordering of the final integer→double conversions). The 1-D **node
  order** must match exactly. This is the "exact / `==0`" tolerance class of
  PATTERNS.md §4 (integer/fixed-point computations), not the loose physical
  tolerance of long iterative float solvers.
- **A stronger, scientific check.** Beyond CPU==GPU, the *result itself* is
  validated against a known answer: the SNP-alternate node `10` must land beside
  reference node `4`, and the inserted node `11` between nodes `6` and `7`. The
  recovered order `0 1 2 3 10 4 5 6 11 7 8 9` and the 100× stress drop confirm the
  layout actually solved the biology, not just that two implementations agree.
- **Edge cases.** Coincident nodes (zero separation) break the `sign` tie
  deterministically toward `+1`; nodes with no terms (zero denominator) keep their
  position; the ragged last block is guarded in both kernels.

## 7. Where this sits in the real world

This is a **reduced-scope teaching version** (CLAUDE.md §13). The full catalog
project is the whole PGGB pipeline; here is what production tools do that we omit:

- **wfmash** does **all-to-all alignment** of the input assemblies using
  **wavefront alignment (WFA)**. WFA's DP front expands along anti-diagonals — the
  same *wavefront* parallelism flagship `3.01` (Smith-Waterman) teaches — and a
  custom CUDA kernel accelerates it. We **assume the graph is already built** and
  start from paths.
- **seqwish** induces the graph from the alignment overlaps; **smoothxg** runs
  partial-order alignment (POA) to normalise messy regions into clean blocks.
- **ODGI** then lays the graph out. Production ODGI uses **stochastic path-SGD**:
  each step samples *one* term and takes a decaying-learning-rate gradient step
  (Zheng et al. 2019). That is faster per unit work but **order-dependent** (not
  bit-reproducible) and needs a tuned schedule. We use **deterministic full-batch
  SMACOF** so CPU and GPU match exactly — the right trade for a verifiable teaching
  artifact. At scale ODGI also goes 2-D (then projects to 1-D) and runs multi-GPU.
- **Indexing.** Real pipelines then build a GBWT / r-index over the graph for
  haplotype-aware queries — out of scope here.

The transferable lessons — stress majorization, the Guttman transform, the
thread-per-term atomic scatter-reduction, and fixed-point determinism — are
exactly what the production layout uses.

---

## References

- **Gansner, Koren & North (2004)**, *Graph Drawing by Stress Majorization* — the
  SMACOF/Guttman update and `1/d²` weighting this project implements.
- **Zheng, Vaddadi, et al. (2019/2022)**, ODGI path-guided SGD layout — the
  stochastic algorithm we deterministically reformulate; source of the 57.3×.
- **PGGB** (https://github.com/pangenome/pggb) — the end-to-end pipeline; read the
  README to see where layout sits.
- **ODGI** (https://github.com/pangenome/odgi) — `odgi sort`/`odgi layout`; the
  GPU layout we model.
- **wfmash** (https://github.com/waveygang/wfmash) — WFA all-to-all alignment that
  seeds the graph (the anti-diagonal wavefront of flagship `3.01`).
- **vg** (https://github.com/vgteam/vg) — the broader pangenome graph toolkit
  (GFA, GBWT, graph alignment).
- **Garrison et al. (2018)**, *Variation graph toolkit* (Nature Biotechnology) —
  foundational reference for variation graphs and reference bias.
