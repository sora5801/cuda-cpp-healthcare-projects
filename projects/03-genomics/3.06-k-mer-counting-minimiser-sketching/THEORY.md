# THEORY — 3.6 k-mer Counting & Minimiser Sketching

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

A DNA **read** is a short string over the four-letter alphabet `{A, C, G, T}`
produced by a sequencer (Illumina reads are typically 100–150 bases). A
sequencing run produces millions to billions of reads that together cover the
genome many times over ("30× coverage" means each base is, on average, read 30
times).

A **k-mer** is any length-`k` substring of a read. The genome of an organism is a
long string; the multiset of all k-mers across all reads is a *lossy fingerprint*
of that genome that is far easier to manipulate than alignments. Counting k-mers
underlies a surprising amount of genomics:

- **Genome-size estimation & profiling.** Plot the histogram "how many distinct
  k-mers occur exactly `c` times?". A genome of size `G` sequenced at coverage `λ`
  produces a peak at `c ≈ λ`; the area under the peak recovers `G` without ever
  assembling the genome (this is what GenomeScope does).
- **Error detection.** A sequencing error creates k-mers that appear **once**
  (the huge spike at `c = 1` in real data); true genomic k-mers recur. Trimming
  low-count k-mers cleans reads before assembly.
- **Assembly.** De Bruijn graph assemblers (SPAdes, Velvet) are built *entirely*
  on k-mers: nodes are (k−1)-mers, edges are k-mers.
- **Metagenomics / species typing.** Two organisms that share DNA share k-mers.
  Comparing k-mer *sets* gives a fast, alignment-free distance — the basis of Mash
  and of pathogen surveillance (GenomeTrakr).

**Why both strands matter.** DNA is double-stranded and antiparallel: the read
`5'-ACG-3'` on one strand is `5'-CGT-3'` on the complementary strand (complement
`A↔T, C↔G`, then reverse). A fragment can be sequenced from either strand, so we
must count a k-mer and its **reverse complement** as the *same thing*. We pick a
**canonical** representative: the lexicographically smaller of the two.

**Minimisers** are a compression trick. Instead of keeping every k-mer, slide a
window of `w` consecutive k-mers and keep only the one with the smallest hash —
its **minimiser**. Overlapping reads from the same locus pick the *same*
minimiser, so minimisers (a) shrink the data ~`w`-fold and (b) preserve the
ability to detect shared sequence. Keep the `s` smallest distinct minimiser
hashes and you have a **MinHash sketch** — a constant-size summary whose overlap
with another sketch estimates set similarity.

---

## 2. The math

### 2.1 Encoding

Map bases to 2 bits: `A=0, C=1, G=2, T=3`. A k-mer `b₀b₁…b_{k−1}` packs into a
64-bit integer (most-significant base first):

```
code = Σ_{i=0}^{k−1} b_i · 4^{k−1−i}      (requires 2k ≤ 64, i.e. k ≤ 31)
```

Because bases are packed MSB-first and `A<C<G<T` as codes, **numeric order of
`code` equals lexicographic order of the string** — handy for canonicalisation
and sorted output.

### 2.2 Reverse complement & canonical form

The complement of base `b` is `3 − b`. The reverse complement of a packed k-mer is

```
revcomp(code) = Σ_{i} (3 − b_i) · 4^{i}    (note the reversed exponent)
```

computed with a `k`-step bit loop (see `kmer_revcomp` in `kmer.h`). The canonical
k-mer is

```
canon(code) = min(code, revcomp(code)).
```

### 2.3 Counting

Let the read set produce a multiset `M` of canonical k-mers. The **count
histogram** is the map `key ↦ multiplicity`,

```
count(x) = |{ p : canon(kmer at position p) = x }|,   for each distinct x ∈ M.
```

### 2.4 Hashing & minimisers

We hash a 64-bit key with the SplitMix64/Murmur3 finalizer `h(x)` (an
*avalanche* mix: flipping one input bit flips ~half the output bits). For a read
with k-mer hashes `h₀, h₁, …`, the minimiser of window `i` is

```
m_i = argmin_{i ≤ j < i+w} h_j ,   emit h_{m_i}.
```

### 2.5 MinHash & Jaccard

For two sets `A, B`, the **Jaccard similarity** is

```
J(A,B) = |A ∩ B| / |A ∪ B|  ∈ [0,1].
```

Bottom-`s` MinHash estimates it without storing the sets: let `S(X)` be the `s`
smallest distinct hash values in `X`. Merge `S(A) ∪ S(B)`, take its `s'` smallest
distinct values `U`, and

```
Ĵ = |{ u ∈ U : u ∈ S(A) and u ∈ S(B) }| / |U|.
```

`Ĵ` is an unbiased estimator of `J` with standard error ≈ `√(J(1−J)/s)`, so larger
sketches are more accurate. (Mash converts `Ĵ` into a mutation distance
`D = −(1/k)·ln(2Ĵ/(1+Ĵ))`.)

---

## 3. The algorithm

### 3.1 Counting (per read set)

```
for each read r:
    for each window position p in [0, len(r)−k]:
        if window has no invalid base:
            x = canon(encode(r, p, k))
            histogram[x] += 1
emit histogram sorted by key
```

- **Serial complexity:** `O(total_bases · k)` with the simple re-encode (an O(1)
  rolling encode drops the `k`). Sorting/compaction is `O(D log D)` in the number
  of distinct k-mers `D`.
- **Parallel:** one independent unit of work per k-mer position; the only sharing
  is the `+= 1` into the table.

### 3.2 Minimiser sketch (per read set)

```
for each read r:
    compute h_j = hash(canon(kmer_j)) for all j         # or "invalid"
    for each window i in [0, (#kmers − w)]:
        emit min_{i ≤ j < i+w} h_j
collect all emitted minimisers -> sort -> dedup -> keep s smallest  (bottom-s)
```

- **Serial complexity:** `O(#kmers · w)` for the naive window min (an O(1)
  monotonic-deque version exists). Bottom-s is `O(W log W)` in the number of
  windows `W`.

### 3.3 Jaccard

Merge two sorted bottom-`s` lists in `O(s)`, count shared via membership tests —
`O(s log s)` here with binary search (see `jaccard_estimate`).

---

## 4. The GPU mapping

### 4.1 Counting = parallel insert + atomic reduce (PATTERNS.md §1 row "clustering / atomic reduce")

We flatten all reads into one `bases` buffer and precompute, on the host, the
absolute start index of every candidate k-mer (`d_pos`). Then:

```
grid  = ceil(P / 256) blocks            # P = number of k-mer positions
block = 256 threads
thread t  ->  k-mer starting at bases[d_pos[t]]
```

Each thread encodes + canonicalises its k-mer and inserts it into a **device
open-addressing hash table** (two arrays `d_keys`, `d_counts` of capacity `C`, a
power of two):

```
slot = hash & (C−1)
loop:
    if d_keys[slot] == key:                 atomicAdd(&d_counts[slot], 1); done
    if d_keys[slot] == EMPTY:
        prev = atomicCAS(&d_keys[slot], EMPTY, key)
        if prev == EMPTY or prev == key:    atomicAdd(&d_counts[slot], 1); done
    slot = (slot + 1) & (C−1)               # linear probe
```

Memory hierarchy: the table lives in **global memory** (it is far larger than a
block's shared memory); the atomics are **global** atomics. We size `C ≥ 2P` so
the **load factor < 0.5**, keeping probe chains short and guaranteeing the loop
terminates. This is a hand-rolled cousin of Jellyfish's lock-free table.

> **The library alternative (no black box).** The catalog also lists
> `thrust::sort_by_key` for a **sort-then-reduce** counter: materialise all `P`
> canonical k-mers, `thrust::sort` them (a GPU radix sort, `O(P)` for fixed-width
> keys), then `thrust::reduce_by_key` to run-length-encode equal runs into counts.
> Hand-rolling that radix sort means writing multi-pass counting sort over 2-bit
> digits with prefix-sum offsets — which is exactly what Thrust does for you.

### 4.2 Minimiser = independent windows + reduction

One thread per minimiser window scans its `w` k-mer hashes and writes the
minimum:

```
grid  = ceil(W / 256); block = 256
thread t -> window starting at bases[d_win[t]]; output d_out[t] = min over w hashes
```

The host then sorts/dedups/truncates the `W` window minima to the bottom-`s`
sketch. A production kernel would do the window-min with **warp shuffles**: load
32 consecutive hashes into a warp's lanes and reduce with `__shfl_down_sync`, so
one warp produces (up to) 32 window minima cooperatively, using register-to-
register moves instead of global traffic. We keep the explicit per-window loop for
readability; the result is identical.

### 4.3 Occupancy & bandwidth

256 threads/block is a good default across sm_75…sm_89. Counting is
**bandwidth-bound** (stream the reads once, scatter into the table); the limiting
resource on real inputs is global-memory throughput and atomic contention on hot
k-mers, *not* arithmetic. That is precisely why GPUs (and tools like Gerbil) win
on large read sets.

---

## 5. Numerical considerations

- **Everything is integer.** k-mers are integers, hashes are integers, counts are
  integers. There is **no floating-point** in counting or sketching, so there is no
  rounding and no precision question — only the Jaccard *ratio* is a `double`, and
  it is a ratio of integers computed identically on both sides.
- **Atomics & determinism (PATTERNS.md §3).** Two threads inserting the same key
  race, but `atomicCAS` lets exactly one claim the slot; both then `atomicAdd 1`
  to the **same** counter. Integer addition is associative and commutative, so the
  final count is **independent of thread order** → reproducible *and* equal to the
  CPU. (A float `atomicAdd` would *not* be order-independent; that is the lesson of
  projects 5.01 and 11.09, applied here for free because counts are integers.)
- **The `set` of table entries is order-independent**, but the *physical slot* a
  key lands in depends on the probe race. We therefore **sort the compacted table
  by key** before printing, so stdout is byte-stable.
- **Sentinel choice.** `EMPTY = 0xFFFF…F` is the all-`T` k-mer at k=32, which we
  forbid (k ≤ 31), so no real key can collide with the "free slot" marker.
- **Invalid bases.** A window containing `N` (or any non-ACGT char) is skipped
  identically by CPU and GPU, so they stay in lockstep.

---

## 6. How we verify correctness

The CPU reference (`reference_cpu.cpp`) and the GPU kernels call the **same**
inline functions in `kmer.h`, so they compute byte-identical per-k-mer values.
`main.cu` then checks, with **tolerance 0** (PATTERNS.md §4, "exact" tier):

1. **Histograms** — same number of distinct k-mers, and for each key the same
   count (parallel walk of two ascending-by-key vectors).
2. **Sketches** — `S_cpu == S_gpu` as exact vectors of `uint64_t` (same bottom-`s`
   hashes in the same order).
3. **Jaccard** — `Ĵ_cpu == Ĵ_gpu` exactly (a ratio of equal integers).

A **second, scientific** check (the analytic cross-check the cookbook recommends):
the planted motif in the synthetic sample is the **unique top-count k-mer**
(`ACGTACGTACG`, count 7), so the demo demonstrates the pipeline recovers a *known*
signal, not just that two implementations agree.

Failure modes this would catch: a wrong canonicalisation (counts split between a
k-mer and its revcomp), a hash-table race bug (lost increments), an off-by-one in
the window loop (different sketch), or a non-deterministic ordering (stdout diff).

---

## 7. Where this sits in the real world

- **Gerbil / KMC3 / Jellyfish** count k-mers at genome scale. They differ from our
  teaching version in ways worth knowing: **cuckoo / Robin Hood probing** (bounded
  worst-case probe length instead of our linear probing), **minimiser-based
  bucketing** so the table is partitioned and can spill to disk, **2-bit packed
  super-k-mers** to cut memory traffic, and careful **out-of-core** I/O for inputs
  far larger than RAM/VRAM. We size a single in-VRAM table with generous headroom;
  production code resizes and partitions.
- **Approximate counting.** When even a packed exact table is too big,
  **count-min sketch** (a few hashed counter rows; over-counts, never under-counts)
  and **HyperLogLog** (cardinality estimation from leading-zero statistics) trade a
  little accuracy for a fixed, tiny footprint. We implement exact counting and
  describe these here rather than coding them (a good exercise).
- **Mash / sourmash** turn minimiser MinHash sketches into species-level distances
  over thousands of genomes — the production face of §2.5. `cuRAND` (catalog) seeds
  the multiple independent hash functions a classic `s`-permutation MinHash uses;
  our **bottom-`s`** variant needs only one hash, so we skip it.
- **Honesty.** This is a reduced-scope teaching build (CLAUDE.md §13): correct and
  exact on the committed sample, but linear-probing, single-table, and in-core. The
  algorithms and data structures named above are the path from here to a real tool.
