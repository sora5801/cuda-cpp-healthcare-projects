# THEORY — 3.20 Long-Read HiFi Assembly Overlap & Polishing

> Deep dive for the curious. The code is in `src/`; this file is the "why".
> Read order: this file → `src/overlap_core.h` → `src/reference_cpu.{h,cpp}` →
> `src/kernels.{cuh,cu}` → `src/main.cu`.

---

## The science

A genome is one long string over the alphabet `{A, C, G, T}`. A sequencer does
**not** read it whole; it reads many short fragments. **PacBio HiFi** reads are
10–25 kb long and >99.5% accurate — accurate enough that, if you can figure out
which reads come from overlapping pieces of the genome, you can stitch them back
into a near-perfect *de novo* assembly (no reference needed). The crux is that
first step: **all-vs-all read overlap** — for every pair of reads, do they share
a stretch of genome, and where?

Naively, testing whether read *i* overlaps read *j* means aligning them
(Smith-Waterman, project 3.01): `O(L²)` per pair. With *N* reads there are
`N(N-1)/2` pairs, so the total is `O(N² L²)`. For a human genome that is
`N ≈ 20` million reads — utterly impossible at face value. Real overlappers
(minimap2, the Darwin GPU overlapper, hifiasm) make it tractable with a two-stage
**seed-and-chain** strategy:

1. **Sketch** each read down to a sparse set of **minimisers** — a tiny,
   strand-symmetric subset of its k-mers — once per read.
2. For a pair, the **shared minimisers** are *seed anchors*. A real overlap shows
   up as a diagonal band of anchors. **Chain** the anchors (find the best
   collinear run) and the chain's score is the overlap strength.

This project implements stages 1–2 as a teaching version: it consumes
pre-sketched reads and computes the all-vs-all overlap-chain score on the GPU,
one read pair per thread. (Stage-3 *polishing* — racon/medaka POA and RNN
consensus — is described under "real world" below; it is out of scope for this
reduced teaching build, by design.)

---

## The math

### Minimisers (the sketch)

A **k-mer** is a length-`k` substring. Pack each base into 2 bits
(`A=0,C=1,G=2,T=3`) so a k-mer is a `2k`-bit integer. Because double-stranded DNA
reads the same sequence from either end, we take the **canonical** k-mer — the
numerically smaller of the forward packing and its reverse complement — then run
it through an integer avalanche hash `h(·)` (a splitmix32 finalizer) so "smallest
hash" is not biased toward poly-A.

Given a window of `w` consecutive k-mers, the **(w, k)-minimiser** is the k-mer of
minimum hash in that window. Sliding the window over a read and keeping each
window's minimiser yields the read's sketch. Two key properties:

- **Sparsity:** on average a fraction `≈ 2/(w+1)` of k-mers are minimisers, so the
  sketch is small (here `w=5` → ~1/3 of k-mers).
- **Robustness:** a single substitution destroys only the `k` k-mers spanning it;
  windows whose minimiser lies elsewhere are untouched. So overlapping reads —
  even with HiFi's rare errors — keep sharing most of their minimisers.

### Anchors and the chaining objective

For reads *q* and *t*, an **anchor** is a pair of positions `(x, y)` such that *q*
has a minimiser at `x` and *t* has the *same hash* at `y`. Collect all anchors and
sort by `x`. A genuine overlap is a **collinear chain**: a subsequence of anchors
along which both coordinates increase together (a near-`+1` diagonal for a
same-strand overlap; a `−1` anti-diagonal for a reverse-strand overlap).

We score a chain with the standard minimiser-chaining recurrence (Li 2018,
*minimap2*), simplified to integers. Let `f[a]` be the best chain score **ending**
at anchor `a`:

```
f[a] = max( MATCH_AWARD,
            max over b<a with link(b→a) valid of  f[b] + link(b → a) )
overlap_score = max_a f[a]
```

where a **link** `b → a` is valid iff both coordinates strictly increase and
neither gap exceeds `MAX_GAP`, and its reward is

```
link(b → a) = MATCH_AWARD − ( |dq − dt| >> GAP_PENALTY_SHIFT )
```

with `dq = x_a − x_b`, `dt = y_a − y_b`. A perfectly collinear step (`dq = dt`)
costs nothing; an indel of size `g` costs `≈ g/16`. Everything is an **integer**,
which (see "verify") is what makes the GPU and CPU results bit-identical.

**Both strands.** A reverse-complement overlap is collinear in `(x, −y)`. So we
run the DP twice — once on `(x, y)`, once on `(x, −y)` — and keep the larger
score (`ovl_chain_best_both_strands`).

---

## The algorithm

Per ordered pair `(i, j)`, `i < j`:

1. **Build anchors** by scanning read *i*'s minimisers (outer) against read *j*'s
   (inner), emitting `(x, y)` on a hash match. Query-position-major order, capped
   at `MAX_ANCHORS`. Cost `O(m_i · m_j)` with `m` = minimisers per read.
2. **Chain** with the `O(A²)` DP above, `A ≤ MAX_ANCHORS`, both strands.

Complexity:

| | serial | parallel (this project) |
|---|---|---|
| one pair | `O(m² + A²)` | same work, on one thread |
| all pairs | `O(N² · (m² + A²))` | `O(N²/P · (m² + A²))` on `P` threads |

The `N²` over pairs is the bottleneck the GPU attacks: every pair is independent,
so we map **one pair per thread** and run `N(N-1)/2` of them concurrently. (The
production speed-up also comes from a **minimiser hash table** that avoids the
inner `O(m_i·m_j)` scan — see "real world".)

---

## The GPU mapping

- **Thread ↔ pair.** Thread `t` (after a grid-stride) owns flat pair slot `t`. It
  decodes `t → (i, j)` with the closed-form inverse of the upper-triangular index
  (`decode_pair` in `kernels.cu`, the device twin of `pair_index` in
  `reference_cpu.h`) so CPU slot `k` and GPU slot `k` describe the same pair — the
  precondition for an element-wise comparison.
- **No shared memory, no atomics.** Every pair's output is independent, so there
  is nothing to reduce or synchronize across threads. Each thread does its anchor
  build + `O(A²)` DP entirely in **on-thread local memory** — fixed-size scratch
  arrays (`anchor_q`, `anchor_t`, `f`, `neg`) of `MAX_ANCHORS` ints. Fixed size is
  what lets a kernel allocate nothing.
- **Memory layout.** Minimisers are uploaded **struct-of-arrays** (a `pos` array
  and a `hash` array) rather than array-of-structs, so a thread streaming a read's
  hashes touches one contiguous run. Per-read `[offset, count)` arrays index into
  the flat minimiser arrays — the canonical "ragged data on the GPU" pattern (the
  device cannot chase host pointers, so we flatten + index).
- **Block size.** 128 threads/block: a warp multiple, kept modest because each
  thread holds `4·MAX_ANCHORS` ints of scratch and runs a non-trivial DP, so we
  trade some occupancy for register/local-memory headroom on `sm_75..sm_89`.
- **Grid-stride loop.** A capped grid (≤ 4096 blocks) covers an arbitrary pair
  count, so the launch config does not grow without bound.

This is the **"score one item against many, each independent"** pattern from
`docs/PATTERNS.md §1` — the same shape as flagship 1.12 (Tanimoto) and 12.01
(spectral search), specialized to "score one read pair" with an inner DP.

```
flat pair slot t  --decode_pair-->  (i, j)
  read i minimisers:  [ (x0,h0) (x1,h1) ... ]   (global memory, pos-sorted)
  read j minimisers:  [ (y0,g0) (y1,g1) ... ]
        |  match hashes  ->  anchors (x,y)
        v
  O(A^2) chaining DP in local memory (both strands)  ->  best score  ->  d_score[t]
```

---

## Numerical considerations

- **Integer everything.** k-mer packing, the canonical min, the hash mix, anchor
  positions, the chaining DP, and the final score are all integer/bitwise. There
  is no floating-point in the scored path, so there is **no rounding, no FMA
  reassociation, no nondeterminism** — the result is exact and reproducible.
- **The one float** is inside `decode_pair`: a `sqrt` to invert the triangular
  number. We immediately snap the result to the correct integer row with a tiny
  `while` correction, so floating rounding cannot misroute a thread to the wrong
  pair. The mapping is exact for every slot.
- **Determinism of the report.** Ties in chain score are broken by `(read_i,
  read_j)` ascending, so the printed top-K is identical every run. Timings (which
  vary) go to **stderr**; the diffed result goes to **stdout** (PATTERNS.md §3).
- **The anchor cap.** `MAX_ANCHORS` bounds per-thread work and scratch. It is
  applied **identically** on CPU and GPU (same scan order, same cap), so both keep
  exactly the same anchors — the cap changes the *score* a little but never the
  *agreement*. The committed sample is tuned so neighbour pairs stay under the cap
  (scores vary meaningfully); see Exercises for what happens when it saturates.

---

## How we verify correctness

Two independent checks:

1. **CPU == GPU, exactly.** `main.cu` runs `overlap_cpu` and `overlap_gpu` and
   compares the full `OverlapResult` arrays element by element — `read_i`,
   `read_j`, `score`, and `n_anchors` must all match for all `N(N-1)/2` pairs.
   Because both sides call the *same* `ovl_chain_best_both_strands` from
   `overlap_core.h` over integer data, the tolerance is **exactly zero** (the
   strongest check; PATTERNS.md §4, like 1.12/3.01/5.01/11.09/12.01). The demo
   prints `66/66 pairs identical`.
2. **The science check.** The synthetic sample has a **known answer**: reads are
   tiled along the genome, so the true overlaps are exactly the consecutive pairs
   `(i, i+1)`. The program recovers precisely 11 candidate overlaps (the 11
   neighbour pairs) and the top-K are all consecutive read indices — validating
   that the seeding + chaining actually finds real overlaps, not just that two
   implementations agree.

---

## Where this sits in the real world

This is a deliberately **reduced-scope teaching version** (CLAUDE.md §13). The
shape is faithful to production overlappers; the omissions are about scale and the
polishing stage:

- **Hash-table seeding.** We find shared anchors with an `O(m_i·m_j)` scan per
  pair. Real tools (minimap2, the **Darwin** GPU overlapper) build a **global
  minimiser hash table** once, then look up each query minimiser in `O(1)` — which
  is where Darwin's reported **109× GPU speed-up** comes from (table resident in
  GPU global memory, seed chains resolved across CUDA blocks). Building a GPU hash
  map is the natural next project.
- **Real chaining.** minimap2's chaining adds a `log`-gap cost, a band, and a
  back-pointer traceback to emit the actual aligned interval; we report only the
  best score. The recurrence is the same idea.
- **String graph + unitigs.** Overlaps feed a string graph that is simplified
  (transitive-edge reduction, tip/bubble removal) into unitigs — the assembly.
  **hifiasm** is the state of the art here, and also does **haplotype phasing** by
  threading heterozygous markers.
- **Polishing.** After assembly, **racon** (GPU partial-order alignment) and
  **medaka** (an RNN consensus, cuDNN-accelerated) correct residual errors. These
  are different GPU patterns (POA DP in shared memory; RNN inference) and would be
  their own projects.

Further reading is in the project `README.md` ("Prior art & further reading").

---

### Appendix — the upper-triangular pair index

We enumerate ordered pairs row by row of the upper triangle:

```
(0,1)(0,2)…(0,N-1) (1,2)…(1,N-1) … (N-2,N-1)
```

Row `i` starts at triangular offset `T(i) = i·N − i(i+1)/2` and holds `N−1−i`
pairs, so `pair_index(i,j) = T(i) + (j−i−1)`. Inverting it (slot → row) is a
quadratic: `i = ⌊(2N−1 − √((2N−1)² − 8t)) / 2⌋`, snapped to the exact row. CPU
and GPU share this mapping so their output arrays line up slot-for-slot.
