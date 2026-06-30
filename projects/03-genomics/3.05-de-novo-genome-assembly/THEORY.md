# THEORY — 3.5 De Novo Genome Assembly

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

This project implements a **reduced-scope teaching version** of de-novo assembly:
the **all-vs-all read-overlap detection** stage, the first and most GPU-amenable
bottleneck of the pipeline. The later stages (graph construction, repeat
resolution, consensus polishing) are described in §7.

---

## 1. The science

A DNA molecule is a long string over the 4-letter alphabet `{A, C, G, T}`. A
**sequencing machine** cannot read a whole chromosome at once; it produces
millions of short **reads** — substrings of the genome, sampled at random
positions, on either of the two complementary strands, and corrupted by errors.

**De novo assembly** is the inverse problem: reconstruct the original genome from
the reads alone, with **no reference** to map against (the "de novo" — "from
scratch"). It is how the first genome of any new species is built, and how the
**telomere-to-telomere (T2T) CHM13** human genome finally filled the gaps left by
reference-guided methods.

The key observation that makes assembly possible: if two reads come from
**overlapping** regions of the genome, they share a long common substring. So the
first job is **overlap detection** — find, for every pair of reads, whether they
overlap (and by how much). Reads that overlap are neighbours; chaining
overlapping reads reconstructs each contiguous stretch of genome (a **contig**).

Two wrinkles make naive substring comparison too slow and too brittle:

1. **Scale.** Comparing every read against every other is `O(n²)` pairs, and a
   full alignment of each pair is itself expensive. A human dataset has tens of
   millions of reads.
2. **Strands and errors.** A read may be from either strand, so we must compare
   it against the **reverse complement** too; and sequencing errors mean exact
   substring matches are rare.

The **minimizer** trick (Roberts et al. 2004; minimap2, Li 2018) solves both: it
reduces each read to a small, strand-independent *sketch* of representative
k-mers, so overlapping reads share many sketch elements while the comparison cost
drops by a large constant factor. This project computes those sketches and the
all-vs-all shared-minimizer counts on the GPU.

---

## 2. The math

**Inputs.** A set of `n` reads `R = {r_0, …, r_{n−1}}`, each a string over
`Σ = {A,C,G,T}`.

**k-mers.** A *k-mer* of a string `s` at position `p` is the length-`k` substring
`s[p .. p+k−1]`. We encode each base in 2 bits (`A=0,C=1,G=2,T=3`), so a k-mer is
a `2k`-bit integer (here `k = 15` → 30 bits, fits in a `uint32`).

**Canonical k-mer.** Let `rc(x)` be the reverse complement of k-mer `x` (reverse
the bases and swap `A↔T`, `C↔G`). The **canonical** k-mer is
`ĉ(x) = min(x, rc(x))`. Using `ĉ` makes the sketch **strand-independent**: a read
and its reverse-complemented overlap partner produce the same canonical k-mers.

**Minimizer.** Fix a hash `h : k-mer → uint32` (a bijective bit-mixer, §5). For a
window of `w` consecutive k-mers `K = (κ_1, …, κ_w)`, the **minimizer** is the
k-mer with the smallest hash:
```
minimizer(K) = argmin_{κ ∈ K} h(ĉ(κ))
```
Sliding the window across a read of length `L` yields `L − k − w + 2` window
minimizers; we then **sort and deduplicate** them to get the read's **sketch**
`M(r) ⊆ uint32` — a small sorted set. Because adjacent windows overlap, the same
minimizer is usually selected by several windows, so `|M(r)| ≈ (L − k + 1)/w` in
expectation: a `~w`-fold reduction.

**Overlap score.** For two reads we score overlap by the **size of the sketch
intersection**:
```
S(r_i, r_j) = | M(r_i) ∩ M(r_j) |
```
This is an integer ≥ 0. Two reads that overlap by many bases share many
minimizers → large `S`; non-overlapping reads share only chance collisions →
small `S`.

**Overlap graph.** Fix a threshold `τ` (`MIN_SHARED = 3` here). The overlap graph
`G = (V, E)` has a vertex per read and an edge `(i,j)` whenever `S(r_i,r_j) ≥ τ`.
The **connected components** of `G` are the draft contigs: reads in one component
tile one region of the genome.

**Outputs.** The thresholded edge set `E` (with weights `S`), and the
component structure (#components, largest component size).

---

## 3. The algorithm

```
SKETCH (host, per read r):
  1. roll a 2-bit k-mer across r, maintaining forward and reverse registers;
     emit ĉ = min(fwd, rev) at each position           --> k-mer stream κ
  2. for each window of w consecutive k-mers, emit argmin h(κ)   --> minimizers
  3. sort + unique                                      --> sketch M(r)

OVERLAP (GPU, per pair):
  4. for each unordered pair (i<j):
       S = | M(r_i) ∩ M(r_j) |   via merge-intersection of two sorted sets

LAYOUT (host):
  5. edges = { (i,j) : S(i,j) >= τ }
  6. connected components of (V, edges)                 --> contigs
```

**Complexity.**

- Sketch: `O(L)` per read for the k-mer roll; the window minimum is `O(L·w)` with
  the simple loop here, or `O(L)` with a monotonic deque (Exercise 1). The
  sort+unique is `O(m log m)` on `m = |minimizers before dedup|`. Sketching is
  inherently serial per read and cheap; we do it on the host.
- Overlap: `P = n(n−1)/2` pairs, each an `O(|M_i| + |M_j|)` merge. Total serial
  work `Θ(n² · m̄)` where `m̄` is the mean sketch size. **This `O(n²)` term is the
  bottleneck** — it grows quadratically in read count — and it is what we move to
  the GPU.
- Layout: union-find over `|E| ≤ P` edges, near-linear `O(P · α)`.

**Data-access pattern.** Each pair reads two short, contiguous sketch slices and
writes one integer. Arithmetic intensity is low (a few comparisons per loaded
word), so the kernel is **memory-/launch-bound** on small inputs; its advantage
is the sheer number of independent pairs at scale (§7, §timing note in README).

**Why minimizers and not full alignment?** A Smith-Waterman alignment of every
pair (project 3.01) is `O(L²)` per pair — `Θ(n² L²)` total, hopeless at scale.
Minimizers replace the `L²` per-pair cost with a cheap set intersection on a
`~L/w`-element sketch, turning the inner cost from quadratic-in-`L` to
linear-in-sketch. Real overlappers go further and *index* minimizers so they
never enumerate all `P` pairs (§7).

---

## 4. The GPU mapping

**The parallelism.** The `P = n(n−1)/2` pair scores are **mutually independent**:
`S(r_i,r_j)` depends only on the two sketches, never on another pair. So we assign
**one logical thread per pair**.

**Thread-to-data mapping.** We index pairs by a single flat id `p ∈ [0, P)` and
decode the upper-triangle coordinate `(i, j)` with `pair_to_ij(p)` (in
`assembly.h`, shared with the CPU). Walking the triangle row by row:
- row `i` holds `n−1−i` pairs; we subtract row sizes until `p` lands in a row.
A fixed grid covers any `P` via a **grid-stride loop**: thread
`p₀ = blockIdx.x·blockDim.x + threadIdx.x` processes `p₀, p₀+stride, …` where
`stride = blockDim.x · gridDim.x`.

**Launch configuration.**
- **block =** 256 threads — a multiple of the 32-lane warp, eight warps to hide
  global-memory latency, many blocks resident for occupancy on sm_75…sm_89.
- **grid =** `min(⌈P / 256⌉, 65535)` blocks; the grid-stride loop handles any
  remainder, so the grid stays modest regardless of `P`.

**Memory hierarchy.**
- **Global memory** holds the flattened sketches. We use a **CSR (compressed
  ragged) layout**: one flat buffer `mins` with all sketches concatenated, plus an
  `offset[n+1]` prefix-sum so read `r`'s minimizers are `mins[offset[r] ..
  offset[r+1])`. A thread finds any read's slice in `O(1)` — no 2-D jagged array
  (which GPUs handle poorly). The buffers are read-only and tagged `__restrict__`
  so the compiler can cache aggressively.
- **Registers** hold the two cursors and the running count of the merge — the
  whole inner loop is register-resident.
- **No shared memory, no atomics, no reduction.** Each thread writes exactly one
  independent output `out_score[p]`. That is what makes the result **fully
  deterministic** (PATTERNS.md §3) and keeps the kernel trivially correct.

```
   reads:   r0      r1      r2   ...            (n reads)
            |       |       |
   sketch:  M(r0)   M(r1)   M(r2) ...           (sorted-unique minimizer sets)
            \_______|_______/
                    v   flatten (CSR)
   mins  = [ M0 .... | M1 ... | M2 ... | ... ]  (one global buffer)
   offset= [ 0,        a,       b,        ... , total ]

   pairs (upper triangle), one thread each:
       p ──pair_to_ij──▶ (i,j) ──▶ count_shared_sorted(Mi, Mj) ──▶ out_score[p]

   grid:  [ block 0 ][ block 1 ] ... (256 threads each, grid-stride over p)
```

**Which "library" does what.** This project deliberately **hand-rolls** the
overlap kernel rather than calling a library — there is no black box to explain.
The two library-ish steps it *would* use at scale, and what hand-rolling them
takes, are: (a) **Thrust/CUB radix sort** to sort minimizers (we sort tiny
sketches on the host with `std::sort` instead); and (b) a GPU **hash table**
(minimizer → reads) to index pairs and skip the `O(n²)` enumeration — that is the
real win in GenomeWorks `cudamapper` and is left as Exercise 3.

---

## 5. Numerical considerations

**Everything on the scored path is integer.** k-mers are 2-bit-packed integers;
the hash is integer bit-mixing; the score is a popcount-free **count** of shared
keys. There is **no floating point anywhere** in the sketch or the overlap score.

**Determinism.** Because the score is an integer produced by an order-independent
merge, and because each thread writes its own output (no atomics, no FP reduction
whose summation order would vary), the GPU result is **bit-identical run to run**
and **bit-identical to the CPU**. This is the ideal case in PATTERNS.md §4:
tolerance `0`.

**The hash.** We use Thomas Wang's 32-bit integer hash (`hash32` in
`assembly.h`) — a fixed sequence of XOR/shift/multiply steps. It is a pure
bijection of bit operations, identical on host and device, so both sides choose
the **same** minimizer in every window. We pick minimizers by smallest *hash*
rather than smallest *k-mer value* so the selected set is a pseudo-random,
well-spread sample (a raw lexicographic minimum would over-select poly-A runs and
make sketches collide spuriously).

**Precision / overflow.** `k = 15` packs into 30 bits, comfortably inside a
`uint32`. Scores and counts are small `int`s (≤ sketch size). `P = n(n−1)/2` is
held as `long long` so the pair index does not overflow for large `n`.

**Edge cases.** Reads shorter than `k` contribute no k-mers (skipped); reads
shorter than one window take their single global-minimum k-mer; ambiguous bases
(`N`) reset the rolling k-mer run. Empty sketches simply yield `S = 0`.

---

## 6. How we verify correctness

**Independent reference.** `reference_cpu.cpp` recomputes every pair's score with
a plain serial double loop. Crucially, it calls the **same** `__host__ __device__`
routine `count_shared_sorted()` (in `assembly.h`) that the kernel calls — the
host/device parity idiom (PATTERNS.md §2). So "CPU agrees with GPU" tests the
*plumbing* (CSR layout, pair decoding, launch config, copies), while the *math* is
shared and thus exact.

**Tolerance = 0.** `main.cu` compares the full `P`-length score arrays element by
element and demands **zero** mismatches (`max |diff| = 0`). Integer counts have no
rounding, so anything other than exact agreement is a real bug, not floating-point
drift.

**A stronger, scientific check.** Beyond CPU == GPU, the synthetic sample has a
**known ground truth**: six reads tiled from one pseudo-genome at offsets
0,12,…,60 must form a **single chain / one connected component** (one contig).
The demo recovers exactly that (11 edges, 1 component, largest = 6), and the
shared-minimizer counts **decay with read distance** (neighbours share ~10,
distance-3 reads share ~3, distance-5 reads fall below threshold) — exactly what
the tiling predicts. That validates the *science*, not just CPU/GPU agreement.

**Edge cases exercised by the loader/sketcher:** CRLF line endings, multi-line
FASTA records, lowercase bases, `N`/ambiguous bases, and reads shorter than a
window.

---

## 7. Where this sits in the real world

This project implements **only** the overlap-detection stage, and even that in a
simplified form (count shared minimizers; no positional chaining). A production
de-novo assembler does much more:

- **Indexed overlap, not all-vs-all.** minimap2 / GenomeWorks `cudamapper` build a
  hash table from minimizer → (read, position), then for each read look up only
  the reads that share a minimizer — avoiding the full `O(n²)`. They also **chain**
  shared minimizer *anchors* by position to estimate the overlap length, strand,
  and coordinates, and to reject random collisions. (Our score is the count only.)
- **String / De Bruijn graph.** Overlaps are assembled into a graph whose nodes
  are reads (string graph) or k-mers (De Bruijn graph); transitive edges are
  removed and unambiguous paths are threaded into **unitigs/contigs**.
- **Repeat resolution & scaffolding.** Repeats create tangles; long reads, read
  depth, and **Hi-C** scaffolding resolve them and order contigs into chromosomes.
- **Consensus polishing.** Draft contigs are error-corrected by **partial-order
  alignment (POA)** over the reads — this is the stage NVIDIA's **racon-GPU**
  accelerates ~70× with custom shared-memory DP kernels, and the **Darwin**
  accelerator showed ~109× for the overlap step on PacBio data.
- **hifiasm** assembles low-error PacBio **HiFi** reads to near-T2T quality; its
  string-graph phase is CPU-centric, with GPU overlap kernels an active research
  insertion point — precisely the stage this project miniaturizes.

So the mental model: **this kernel is the very first box of the overlap-layout-
consensus pipeline**, taught in isolation so the GPU mapping is legible.

---

## References

- **Roberts et al. (2004), "Reducing storage requirements for biological sequence
  comparison."** Bioinformatics — the original minimizer idea.
- **Li, H. (2018), "Minimap2: pairwise alignment for nucleotide sequences."**
  Bioinformatics — the canonical minimizer sketch + chaining we miniaturize;
  read `mm_sketch` for the real `O(m)` rolling implementation.
- **GenomeWorks / racon-GPU** (<https://github.com/NVIDIA-Genomics-Research/GenomeWorks>)
  — GPU `cudamapper` (overlap) and POA polishing; the production analogue.
- **hifiasm** (<https://github.com/chhylp123/hifiasm>) — state-of-the-art HiFi
  assembler; how overlaps become a string graph and then contigs.
- **Turakhia et al. (2018), "Darwin: a genomics co-processor."** ASPLOS — the
  ~109× hardware-accelerated read-overlap result cited in the catalog.
- **Nurk et al. (2022), "The complete sequence of a human genome" (T2T-CHM13).**
  Science — why reference-free assembly matters.
