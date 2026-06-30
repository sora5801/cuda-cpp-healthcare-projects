# THEORY — 3.27 Suffix Array / BWT / FM-Index Construction

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

A genome is a string. The human reference (GRCh38) is ~3.1 billion characters
over the four-letter DNA alphabet `{A, C, G, T}`. The central computational task
of read alignment — "where in this genome did this 100-base sequencing read come
from?" — is *substring search at massive scale*: hundreds of millions of short
patterns matched against one enormous text.

Scanning the whole genome for every read is hopeless. Instead, aligners
**preprocess the genome once** into an index that makes search depend on the
*pattern* length, not the genome length. The index of choice is the
**FM-index**, built from the **Burrows-Wheeler transform (BWT)** of the genome,
which is in turn derived from the genome's **suffix array (SA)**. These three
structures power BWA, Bowtie2, and the string-graph assemblers; the BWT also
underlies `bzip2`-style compression because it groups similar contexts together.

Building the index is the expensive, one-time step — and for a 3 Gb genome it is
the bottleneck. This project teaches how to build the SA, BWT, and a backward-
search FM-index, and how the SA construction — a giant sort — maps onto the GPU.

## 2. The math

Let `T = t_0 t_1 … t_{n-2}` be the input text. We append a unique **sentinel**
`$` that compares strictly smaller than every real symbol:

```
T$ = t_0 t_1 … t_{n-2} $      (length n, including the sentinel)
```

The sentinel guarantees all `n` suffixes are distinct (no suffix is a prefix of
another), so the suffix array is **unique**.

- **Suffix `i`** is the substring `T$[i .. n-1]`.
- **Suffix array** `SA[0 .. n-1]` is the permutation of `{0, …, n-1}` such that

  ```
  suffix(SA[0]) < suffix(SA[1]) < … < suffix(SA[n-1])      (lexicographic order)
  ```

  i.e. `SA[r]` is the starting position of the `r`-th smallest suffix.

- **Burrows-Wheeler transform** `L[0 .. n-1]` (the "last column"):

  ```
  L[r] = T$[ (SA[r] - 1 + n) mod n ]
  ```

  — the character cyclically *preceding* the `r`-th sorted suffix.

- **FM-index** consists of:
  - `C[c]` = number of characters in `T$` strictly smaller than `c`
    (the first row of the sorted matrix that begins with `c`), and
  - `Occ(c, r)` = number of occurrences of `c` in `L[0 .. r-1]` (a *rank* query).

- **Backward search.** To count occurrences of a pattern `P`, keep a half-open
  range `[lo, hi)` of SA rows whose suffixes start with the current suffix of
  `P`, and process `P` right-to-left with the **LF-mapping** step:

  ```
  lo  ←  C[c] + Occ(c, lo)
  hi  ←  C[c] + Occ(c, hi)        for c = P[m-1], P[m-2], …, P[0]
  ```

  The final `hi - lo` is the number of occurrences of `P` in `T`.

The hard part is computing `SA`. Everything else (`L`, `C`, `Occ`, backward
search) is a linear post-pass once `SA` exists.

## 3. The algorithm

### Prefix doubling

We never compare two suffixes character-by-character. Instead we maintain a
**rank** for every suffix that captures the order of its first `K` characters,
and we double `K` each round.

```
rank[i] ← code(T$[i])                      # round 0: order by 1 character
for k = 1, 2, 4, … while not all ranks unique:
    key[i] ← (rank[i], rank[i+k])          # order by 2k characters
    sort all suffixes by key
    renumber ranks from the sorted order   # equal keys share a rank
```

**Why it works.** If we already know the order of all length-`k` prefixes
(that's `rank[]`), then the order of all length-`2k` prefixes is decided by the
pair *(rank of the first half, rank of the second half)* = `(rank[i], rank[i+k])`.
Each comparison is now `O(1)` instead of `O(k)`. After `⌈log2 n⌉` rounds the
known prefix length exceeds `n`, so every rank is unique and the sorted order is
the suffix array.

### Packing the key (the shared core)

We pack the pair into one 64-bit integer so a single scalar sort suffices
(`src/sa_core.h`):

```
key = ((rank[i] + 1) << 32) | (rank[i+k] + 1)     # +1 so sentinel -1 → 0
```

This *identical* function runs on the CPU and GPU, so both sort by bit-identical
keys — the reason their suffix arrays match exactly.

### Complexity

| | Work | Depth (parallel) |
|---|---|---|
| Per round | `O(n)` keys + one sort `O(n)` (radix) or `O(n log n)` (comparison) | `O(log n)` for the sort |
| Rounds | `O(log n)` | — |
| **Total** | `O(n log n)` (radix) / `O(n log^2 n)` (comparison) | `O(log^2 n)` |

The serial CPU reference uses `std::stable_sort` per round → `O(n log^2 n)`.
The GPU uses a radix sort per round → `O(n log n)` total work, and the sort is
where all the parallelism lives.

### BWT and FM backward search

Both are linear passes over `SA` / `L` and are computed on the host from
whichever suffix array we hold (`src/reference_cpu.cpp`: `bwt_from_sa`,
`fm_count`). `Occ()` here is a simple `O(n)` scan per step (clear over fast); the
production rank-dictionary is discussed in §7.

## 4. The GPU mapping

One doubling round is a **chain of data-parallel kernels** (`src/kernels.cu`):

```
                 d_rank (ranks from previous round)        d_valA (current order)
                        │                                         │
                        ▼                                         ▼
   build_keys_kernel: key[p] = pack_key(val[p], k, rank)   (1 thread / slot p)
                        │
                        ▼
   ┌──────────── LSD radix sort over (key,val), 8 passes of 8 bits ───────────┐
   │  histogram_kernel  : atomicAdd into 256 integer buckets  (1 thread/elem) │
   │  host exclusive scan of 256 buckets → per-bucket start offsets           │
   │  scatter_kernel    : stable scatter into sorted slots    (single thread) │
   └──────────────────────────────────────────────────────────────────────────┘
                        │  (sorted keys + suffix indices)
                        ▼
   flag_kernel        : flag[p] = (key[p] != key[p-1])       (1 thread / slot p)
   scan_inclusive_kernel: prefix[p] = Σ flag[0..p]  (= new rank) (single thread)
   write_ranks_kernel : rank[val[p]] = prefix[p]             (1 thread / slot p)
                        │
                        ▼   repeat with k ← 2k until #distinct ranks == n
```

- **Thread-to-data mapping.** In the element-wise kernels, thread
  `p = blockIdx.x * blockDim.x + threadIdx.x` owns sorted slot `p` (or element
  `p`). The ragged last block is guarded by `if (p >= n) return;`.
- **Launch configuration.** `THREADS_PER_BLOCK = 256` (8 warps — a multiple of
  the 32-lane warp, good occupancy on sm_75…sm_89); `grid = ceil(n / 256)`.
- **Memory hierarchy.** All large arrays (keys, values, ranks, flags, prefixes)
  live in **global memory**; accesses are contiguous (slot `p` reads index `p`),
  so they coalesce. The 256-bucket histogram is small and lives in global memory
  too; we exclusive-scan it on the host (256 ints — negligible) for clarity.
  Integer **atomics** (`atomicAdd`) build the histogram — see §5 on why integer
  atomics keep the result deterministic. No shared memory or constant memory is
  needed for this teaching version (a tiled, per-block radix would use shared
  memory; see Exercise 2).
- **Library calls / no black boxes.** The catalog specifies
  `thrust::sort_by_key` for the sort and CUB prefix sums for the rank update. We
  **hand-roll both**: the LSD radix sort is exactly what `thrust::stable_sort_by_key`
  does internally (count digits → scan → scatter), and the inclusive scan is what
  `thrust::inclusive_scan` / `cub::DeviceScan` does. Writing them out means the
  learner sees the primitive instead of trusting it. To use the real library
  instead, see Exercise 1 (note CUDA 13's CCCL needs `-Xcompiler /Zc:preprocessor`
  on MSVC).

## 5. Numerical considerations

- **Everything is integer.** Ranks, keys, histogram counts, prefix sums, and the
  suffix array are all integers. There is **no floating point anywhere**, so
  there is no rounding, no FMA divergence, and no FP non-associativity to worry
  about. This is why the verification tolerance is exactly **zero**.
- **Determinism.** The histogram uses `atomicAdd` on `unsigned int`; integer
  addition is associative and commutative, so the histogram is identical
  regardless of the (nondeterministic) thread completion order. The radix
  scatter is run **single-threaded** so it is provably *stable* (equal-digit
  records keep their input order) and deterministic. The inclusive scan is also
  single-threaded. Result: stdout is byte-identical on every run (verified across
  repeated runs).
- **Stability matters for correctness, not just reproducibility.** An LSD radix
  sort is only correct if each pass is stable. Our sequential scatter guarantees
  that. A parallel scatter (Exercise 2) must reproduce the same stable order, or
  the intermediate ranks — and hence the final SA — can differ.
- **The renumber must be an *inclusive* scan.** `rank[p]` = number of key
  boundaries at-or-before slot `p` = `Σ flag[0..p]`. Using an *exclusive* scan
  (a classic off-by-one) drops the boundary *at* `p` and gives every tied group
  the wrong rank — this was a real bug caught by the exact-match verification
  during development, which is precisely what the CPU reference is for.
- **Range / overflow.** Ranks are `< n ≤ 2^31`, so `(rank+1)` fits in 32 bits and
  the two halves of the 64-bit key never collide.

## 6. How we verify correctness

The CPU reference (`src/reference_cpu.cpp`) builds the suffix array with the same
prefix-doubling recurrence but a plain `std::stable_sort` and an obvious serial
renumber — code simple enough to read and trust. `src/main.cu` then checks three
things, **all of which must hold**:

1. **`SA mismatches == 0`** — the GPU suffix array equals the CPU one at every
   position. Because both are integer permutations of `{0,…,n-1}`, the tolerance
   is `0` (PATTERNS.md §4: exact when the computation is integer on both sides).
2. **`BWT match`** — the BWT strings derived from each SA are identical.
3. **`FM match`** — FM-index backward search returns the same occurrence count.

Why this is convincing: the CPU and GPU are *independent implementations* (a
serial comparison sort vs. a parallel radix sort) that share only the tiny
`pack_key`/`char_to_code` math. Agreement on the full permutation is strong
evidence both are right. As an additional *semantic* check, the synthetic input
plants the motif `ACGT` repeatedly so the substring `"ACG"` occurs a known **6**
times — the FM-index recovers exactly that, validating the science, not just
CPU==GPU agreement. Edge cases: the sentinel suffix always lands at `SA[0]`
(`SA[0] = 60`), and a length-1 text (`"$"` alone) is handled by the wrapper.

## 7. Where this sits in the real world

This is a **reduced-scope teaching version** (CLAUDE.md §13). Production index
builders differ in scale and machinery:

- **Skew / DC3 and SA-IS.** Real linear-time SA construction often uses the
  **DC3 / skew** algorithm (recursive, `O(n)`) or **SA-IS** (induced sorting,
  `O(n)`), not `O(n log n)` prefix doubling. The catalog names the GPU skew
  algorithm; prefix doubling is chosen here because its per-round structure (key
  → sort → renumber) is the clearest way to *see* the GPU sort.
- **`thrust::sort_by_key` / CUB.** A production GPU build would call
  `thrust::stable_sort_by_key` (a tuned multi-pass radix sort using shared memory
  and warp-level primitives) and `cub::DeviceScan` rather than our single-block
  kernels. Our hand-rolled versions exist to demystify them.
- **External memory & terabase scale.** **Big-BWT** uses *prefix-free parsing* to
  build the BWT of strings far larger than RAM; **ropebwt2** grows a BWT
  incrementally over millions of reads using a rope/B+-tree. Both target the
  metagenomics/pangenome scale this in-core demo cannot reach.
- **2-bit packing & FM-index dictionaries.** Real aligners pack DNA at 2 bits/base
  and replace our `O(n)`-per-step `Occ()` with a **wavelet tree** or sampled
  occurrence table, making backward search truly `O(|pattern|)` (Exercise 5).

---

## References

- **GPU suffix array via prefix doubling** —
  https://www.researchgate.net/publication/303594470 — the fast parallel SA
  construction this project simplifies; read for the GPU-resident radix sort and
  rank-update details.
- **ropebwt2** — https://github.com/lh3/ropebwt2 — incremental BWT over read
  collections; the rope data structure for growing a BWT.
- **CUDPP BWT / NVIDIA parallel-algorithms blog** —
  https://devblogs.nvidia.com/cutting-edge-parallel-algorithms-research-cuda/ —
  the primitive-based BWT our hand-rolled kernels stand in for.
- **Big-BWT** — https://github.com/alshai/Big-BWT — external-memory BWT via
  prefix-free parsing for terabase strings.
- Ferragina & Manzini, *Opportunistic Data Structures with Applications* (FOCS
  2000) — the original FM-index and backward search.
- Burrows & Wheeler, *A Block-sorting Lossless Data Compression Algorithm* (1994)
  — the transform itself.
