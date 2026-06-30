# THEORY — 3.26 GPU BAM Sorting & Deduplication

> The deep "why" behind the code. Read this alongside `src/bam.h` (the shared
> key/compare math), `src/kernels.cu` (the GPU pipeline), and `reference_cpu.cpp`
> (the trusted baseline). This is a **reduced-scope teaching model** of two real
> sequencing-pipeline steps; the simplifications are spelled out under
> "Where this sits in the real world".
>
> _Educational only — not for clinical use._

---

## The science

A DNA sequencer does not read a genome end to end. It shreds many copies of the
sample into short fragments, reads ~100–250 bases from each fragment ("reads"),
and an **aligner** (BWA-MEM, minimap2, …) maps every read back to a reference
genome, recording **where** it landed: which chromosome, at which position, on
which strand. The output is a **BAM file** — a compressed, binary table with one
row per aligned read.

Before anything biological can be computed (calling variants, measuring coverage),
two housekeeping steps must happen, and both are classic bottlenecks:

1. **Coordinate sorting.** Reads come out of the aligner in roughly the order they
   were sequenced — i.e. *not* in genome order. Almost every downstream tool
   (pileup engines, variant callers, index builders) requires the reads sorted by
   genomic coordinate: chromosome, then position, then strand. `samtools sort`
   does this on the CPU.

2. **Duplicate marking.** Library prep amplifies fragments by PCR. If one original
   fragment is copied many times, several reads map to the *exact same place* and
   look like independent evidence — but they are not; they are echoes of a single
   molecule. Counting them inflates apparent depth and biases variant calls.
   **Duplicate marking** groups reads that share a *fragment signature* and keeps
   one representative (the highest-quality copy), flagging the rest. This is what
   Picard `MarkDuplicates` and `samtools markdup` do.

Both steps are pure **data wrangling** — no floating-point physics — which is
exactly why they map so cleanly onto two of the GPU's best-understood parallel
primitives: a **radix sort** and a **segmented (group-by) reduction**.

---

## The math

We model each aligned read as a small integer tuple (see `ReadRecord` in
`src/bam.h`):

$$ r = (\text{ref}, \text{pos}, \text{strand}, \text{mate}, q, \text{id}) $$

where `ref` is the chromosome index, `pos` the leftmost coordinate, `strand`
∈ {0,1}, `mate` the fragment's other-end coordinate, `q` the base-quality sum
(the duplicate *score*), and `id` the original input index.

**Coordinate sort.** Define a total order `<` on reads:

$$ r_a < r_b \iff (\text{ref}_a, \text{pos}_a, \text{strand}_a, \text{id}_a)
   <_{\text{lex}} (\text{ref}_b, \text{pos}_b, \text{strand}_b, \text{id}_b). $$

The first three fields are the genomic order; appending `id` makes the order
**total** (no ties), which is what guarantees a *unique* sorted output. We encode
the first three fields into a single unsigned 64-bit **key** so a radix sort can
order them in one pass:

$$ \text{key}(r) = (\text{ref} \ll 40)\;|\;(\text{pos} \ll 16)\;|\;\text{strand}. $$

Because the most-significant bits hold `ref` and the least hold `strand`, the
*natural unsigned integer order* of the keys equals the desired lexicographic
order. (See `coord_key` in `bam.h`.)

**Duplicate marking.** Two reads are duplicates iff they share the **signature**

$$ \text{sig}(r) = (\text{ref}, \text{pos}, \text{strand}, \text{mate}), $$

packed into another 64-bit key by `dup_key`. Within each signature group `G`, the
read to **keep** (the inferred original) is the argmax of the score, ties broken
by lowest id:

$$ \text{keep}(G) = \arg\max_{r \in G}\;(q_r,\; -\text{id}_r). $$

Every other read in `G` is marked a duplicate. The duplicate count is
$\sum_G (|G| - 1)$.

---

## The algorithm

### Coordinate sort
1. Map each read → `(coord_key, id)` (one independent computation per read).
2. **Radix-sort** the pairs by `coord_key`, with `id` as a stable tie-breaker, to
   get the total order above.
3. Gather the full records into the sorted order.

Complexity: a comparison sort is $O(n \log n)$; **radix sort is $O(w \cdot n)$**
for a fixed key width `w` (here ~8 byte-passes), i.e. effectively **linear** in
`n` — the algorithmic reason GPUs (and `samtools` with radix on integer keys)
beat naive comparison sorts on huge read counts.

### Duplicate marking
1. Map each read → `(dup_key, id)`.
2. **Sort by `dup_key`** so equal-signature reads become contiguous runs.
3. **Segmented reduction** (`reduce_by_key`): collapse each run to the id of its
   best read, using the combine op `argmax(score, −id)`. This is "GROUP BY
   signature, keep argmax(quality)".
4. Map each read → flag it a duplicate unless it is its group's kept id.

Complexity: dominated by the sort, again $O(w \cdot n)$ with a radix sort; the two
map passes and the reduction are each $O(n)$.

The CPU reference (`reference_cpu.cpp`) does the identical computation with
`std::sort` and a single hash-map grouping pass — obviously correct, the baseline
we trust.

```
input reads (shuffled)        sort by coord_key            sorted by genome order
  [chr3 p=4.7M +]      ──►      radix sort        ──►        [chr0 p=14968 +]
  [chr2 p=13.8M -]             (thrust)                      [chr0 p=46030 - ] ┐
  [chr1 p=15.0M +]                                           [chr0 p=46030 - ] │ a duplicate
   ...                                                       [chr0 p=46030 - ] │ cluster (same
                                                             [chr0 p=46030 - ] ┘ signature)
```

---

## The GPU mapping

This project is the worked example for **PATTERNS.md §1 "clustering / group
aggregate"** (sort + segmented reduce) and **§5 "use a CUDA library without a
black box"**. The heavy primitives are **Thrust** (which ships with CUDA, header
only — no extra `.lib`):

| Step | Primitive | What runs on the GPU |
|---|---|---|
| key computation | custom `__global__` **map** kernel | one thread per read; `i = blockIdx.x*blockDim.x + threadIdx.x`; writes `keys[i]`, `ids[i]` |
| coordinate sort | `thrust::sort_by_key` + `stable_sort_by_key` | parallel **LSD radix sort** on the 64-bit keys |
| grouping | `thrust::sort_by_key` | brings equal signatures together |
| best-per-group | `thrust::reduce_by_key` | **segmented reduction** with the `BestDup` combine op |
| group ids | `thrust::transform` + `inclusive_scan` | boundary flags → prefix-sum → group index per slot |
| flag writeback | custom `__global__` **map** kernel | one thread per sorted slot; writes `is_dup[id]` (distinct addresses → **no atomics**) |

**Why the total-order trick (the two-pass sort).** A radix sort orders by a single
scalar key; ties in `coord_key` (reads at the same ref/pos/strand) would land in
an arbitrary order, and the GPU's order could then differ from the CPU's. We fix
this by composing **stable** passes least-significant-key-first: sort by `id`,
then `stable_sort` by `coord_key`. Equal keys retain the `id` order, so the result
is the exact total order `coord_less` — and the GPU matches the CPU byte-for-byte.
(See the comments around `sort_gpu` in `kernels.cu`.)

**Memory hierarchy.** The keys and ids live in **global memory**; Thrust's radix
sort internally tiles into **shared memory** for per-block histogram/scatter, but
we let the library own that. The map kernels are pure streaming reads/writes —
**bandwidth-bound**, which is precisely the regime where the GPU's high memory
bandwidth wins. No constant/texture memory is needed.

**Why no atomics in the writeback.** Each read's `id` appears in exactly one
sorted slot, so `mark_kernel` writes each `is_dup[id]` from exactly one thread —
the writes hit **distinct addresses**, so there is no contention and no atomic.
(Contrast project 11.09, where many threads accumulate into the *same* centroid
and must use atomics.)

---

## Numerical considerations

There is **no floating point anywhere** in this project's comparison path. Every
sortable/comparable quantity — ids, positions, strands, quality sums, packed keys
— is an integer. That has two consequences:

- **Determinism.** Integer comparisons and integer "argmax" are exact and
  order-independent. The `BestDup` combine op for `reduce_by_key` is associative
  **and** commutative (it is an argmax over a total order), which is exactly what
  Thrust requires for a correct parallel reduction. So the GPU's per-group winner
  does not depend on thread scheduling — it is reproducible run to run, and equal
  to the CPU's choice.
- **No tolerance needed.** Unlike the iterative-solver flagships (which verify to a
  small physical epsilon because FMA reorders float sums), this project verifies
  with **exact equality** (PATTERNS.md §4, the "integer / fixed-point" row).

**Key-width budget.** The packed keys use 24 bits for `ref`, 24 for `pos`, 1 for
`strand`, and 15 for `mate`. The loader (`load_readset`) range-checks every field
so a stray high bit cannot corrupt a neighbouring field in the packed key. Real
genomes need 32-bit positions; widening the key is discussed below.

---

## How we verify correctness

`main.cu` runs the CPU reference and the GPU pipeline on the same input and
asserts, **exactly**:

1. **Sort order identical** — the two sorted sequences agree field-by-field at
   every position (`sort mismatches == 0`).
2. **Duplicate flags identical** — `is_dup[id]` agrees for every read
   (`dup-flag mismatches == 0`), and the total duplicate counts agree.

A second, **independent** check validates the *science*, not just CPU==GPU
agreement: the synthetic generator plants a **known** number of duplicates (2–5
copies per fragment cluster, all but the best are duplicates) and prints it
(**358** for the committed sample). The demo's reported duplicate count must equal
that planted number — so we know the dedup logic is right, not merely
self-consistent. The deterministic stdout (including an FNV-1a digest of the
sorted order) is diffed against `demo/expected_output.txt`.

Edge cases handled: the ragged last thread block (guarded by `if (i >= n) return`),
empty/duplicate-free groups (group size 1 → nothing flagged), and ties in score
(broken by lowest id, identically on both sides).

---

## Where this sits in the real world

This is a **reduced-scope teaching model**. A production GPU BAM tool differs in
ways worth knowing:

- **Real BAM I/O.** Production tools parse compressed **BGZF/BAM** records (CIGAR
  strings, soft-clipping, flags, optional tags) and write a coordinate-sorted BAM
  plus a **BAI/CSI index** (built with a parallel prefix scan over the sorted
  positions). We operate on a flat in-memory integer record and skip the index;
  the catalog notes index construction as a parallel-prefix extension.
- **Wider keys / real coordinates.** Real positions are 32-bit and chromosomes
  number in the tens; a production key is 128-bit (or the four fields are sorted
  directly with a tuple comparator). Our 24/24/1/15-bit packing is a didactic
  simplification, range-checked by the loader.
- **Correct duplicate definition.** Real `MarkDuplicates` accounts for clipping
  (using the *unclipped* 5′ coordinate), read pairs, optical vs. PCR duplicates,
  and **UMI-aware** collapsing (the catalog lists UMI collapsing as the next step).
  Our signature `(ref, pos, strand, mate)` captures the essence — group by
  fragment ends, keep the best — without those refinements.
- **The research-grade tool.** **NVIDIA Parabricks `fq2bam`** fuses GPU sort +
  markdup *into* the alignment step, overlapping the GPU sort with alignment I/O
  so the whole post-alignment cleanup is nearly free (the catalog's "~6-minute"
  figure). **FastDup** (arXiv:2505.06127) explores a speculation-and-test GPU
  duplicate-marking scheme. `biobambam2` and `samtools` are the CPU references.
- **Multi-GPU.** Terabyte BAMs exceed one GPU's memory; the production pattern is
  **shard-and-merge** — sort shards on each GPU, then a parallel merge — which our
  single-GPU, in-memory version omits.

The point of this project is the *pattern*: post-alignment cleanup is two parallel
primitives (radix sort, segmented reduce-by-key), made deterministic by keeping
every key an integer and every order total.
