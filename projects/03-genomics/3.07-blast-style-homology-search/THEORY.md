# THEORY — 3.7 BLAST-Style Homology Search

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and to sequence bioinformatics. Read
> [README.md](README.md) first for the overview; read `src/blast_core.h`
> alongside the "GPU mapping" section.

---

## The science

Proteins are chains of **amino acids** (residues), written as strings over a
20-letter alphabet (`A C D E F G H I K L M N P Q R S T V W Y`, plus a few
ambiguity codes). Two proteins are **homologous** if they descend from a common
ancestor; homologs usually share **sequence similarity** because evolution
conserves functionally important residues while drifting elsewhere.

**Homology search** is the workhorse question of bioinformatics: *given a query
protein, which sequences in a giant database are its homologs?* Answering it lets
you transfer functional annotation ("this unknown protein looks like a known
kinase"), build the **multiple-sequence alignments (MSAs)** that protein-structure
predictors like **AlphaFold2** consume, and trace evolutionary relationships.

A full alignment of the query against every database sequence (Smith-Waterman,
flagship 3.01) is too slow at database scale — UniRef50 has tens of millions of
sequences. **BLAST** (Basic Local Alignment Search Tool, Altschul et al. 1990)
made the problem tractable with a heuristic: most database sequences share *no*
short exact words with the query and can be rejected instantly; only the few that
do are worth scoring. This is **seed-filter-extend**, and it is what we build.

> **Not a clinical tool.** This is educational. Our data is synthetic; real
> homology search informs research, not diagnosis.

---

## The math

**Substitution scores.** Aligning residue `a` to residue `b` earns a score
`s(a,b)` from a **substitution matrix**. We use **BLOSUM62** (Henikoff & Henikoff
1992), where

```
s(a,b) = round( (1/λ) · log2( p(a,b) / (f(a)·f(b)) ) )
```

is the **log-odds** that `a` and `b` are aligned in *homologous* proteins
(observed joint frequency `p(a,b)`) versus *by chance* (product of background
frequencies `f(a)·f(b)`), scaled and rounded to an **integer**. Identities and
conservative swaps (e.g. `I↔L`, `K↔R`) score positive; dissimilar pairs score
negative. The integer-ness matters: it makes every downstream score exact.

**Ungapped local alignment.** A seed fixes a **diagonal**: query position `qpos`
is aligned to database position `dpos`, and we only consider alignments where
`query[qpos+t]` pairs with `db[dpos+t]` for integer offsets `t` (no insertions or
deletions — "gapless"). The score of the segment spanning offsets `[t0, t1]` is

```
S(t0,t1) = Σ_{t=t0}^{t1}  s( query[qpos+t], db[dpos+t] ).
```

The **HSP score** is the maximum of `S(t0,t1)` over all segments containing the
seed — a one-dimensional **maximum-subarray** problem along the diagonal.

**X-drop.** Computing the exact maximum subarray is `O(L)`, but we can prune: walk
outward from the seed accumulating score `run`; track the best `run` seen; if
`run` ever falls more than `X` below that best, **stop** — no later extension in
this direction can recover, because the matrix's negative scores would have to be
overcome. `X` (here 12) trades sensitivity for speed. This is BLAST's actual
ungapped-extension rule.

**E-values (not computed here).** A real tool converts a raw HSP score `S` into an
**E-value** `E = K·m·n·e^{−λS}` (Karlin–Altschul), the expected number of
chance hits of that score in a database of `m·n` residues. We report raw scores
only; see "Where this sits in the real world" and Exercise 4.

---

## The algorithm

```
encode query and DB residues to indices 0..23
build query k-mer index:  map  kmer_code -> list of query positions
for each DB sequence d (INDEPENDENT):
    best = 0
    for dpos in 0 .. len(d)-k:
        code = pack_kmer(d, dpos, k)          # base-24 fold of k residues
        for each qpos in query_index[code]:   # every seed on this k-mer
            hsp = gapless_xdrop(query, d, qpos, dpos, k, BLOSUM62, X)
            best = max(best, hsp)
    score[d] = best
rank sequences by score
```

**Why k-mers filter so well.** With a 20-letter alphabet there are `20^k` possible
words; for `k=4` that is 160,000. A random length-`L` sequence shares a *specific*
4-mer with the query with probability `≈ L/160000` per position, so unrelated
sequences almost never seed, while true homologs (which contain conserved runs)
seed reliably. The filter discards the haystack and keeps the needles.

**Packing k-mers.** Each residue is `< 24`, so a length-`k` word folds **perfectly
and collision-free** into one integer by base-24 digits:
`code = ((r0·24 + r1)·24 + r2)·24 + r3`. No hashing, no collisions — see
`pack_kmer()` in `src/reference_cpu.h` (shared by host and device).

**Complexity.**
- Building the query index: `O(Q)` for query length `Q`.
- Serial search (CPU): `O(Σ_d L_d · (seeds per window) · (extension length))`. In
  the worst case extensions are `O(L_d)`, giving `O(Σ_d L_d²)`; in practice X-drop
  keeps extensions short, so it is closer to `O(total residues × seeds)`.
- The parallel version does the **same total work**, but spreads the outer
  `for each DB sequence` loop across GPU threads, so wall-clock time falls by up
  to the number of resident threads (bounded by memory bandwidth).

---

## The GPU mapping

**Thread-to-data mapping.** One **thread per database sequence** (the pattern in
PATTERNS.md §1, "score one query vs N items, each independent" — same family as
flagship 1.12 Tanimoto). Thread `i = blockIdx.x·blockDim.x + threadIdx.x` handles
DB sequence `i`; a **grid-stride loop** lets a fixed-size grid cover an
arbitrarily large database. Each thread runs the *entire body* of the CPU's
inner loop for its sequence — slide window, look up seeds, extend, keep the max —
and writes one integer to `out[i]`. No two threads touch the same output, so
there are **no atomics and no shared memory** to coordinate.

**Memory hierarchy and why.**

| Data | Where | Why |
|---|---|---|
| **BLOSUM62** (576 B) | **constant memory** (`__constant__ c_blosum`) | read by every thread for every scored pair, never written, identical for the whole launch → the constant cache **broadcasts** one address warp-wide in a single transaction. Mirrors how 1.12 keeps its query in constant memory. |
| Query residues | global | read by all threads; small; cached. |
| DB residues | global, **concatenated** into one buffer + per-seq `(offset,length)` | one coalesced array instead of N little ones; thread `i` reads a contiguous slice. |
| Flattened query index | global, **sorted by k-mer code** | a `SeedPair{code,qpos}` array the thread **binary-searches** (`device_lower_bound`) — the GPU stand-in for BLAST's hash lookup, written out explicitly (no black box). |
| Per-thread scalars (`run`, `best`, indices) | registers | the X-drop loop is a handful of registers → high occupancy. |

**Block size.** 128 threads/block — a multiple of the 32-lane warp with healthy
occupancy on sm_75…sm_89 while keeping register pressure low. DB sequences vary in
length, so threads in a warp **diverge** in their inner-loop trip counts; a
moderate block size keeps the SM busy with other warps while stragglers finish.

**Divergence & load imbalance (the honest caveat).** Because sequences differ in
length and seed count, neighbouring threads do different amounts of work — a warp
runs as long as its slowest lane. This is the main inefficiency of
thread-per-sequence and why production tools (MMseqs2-GPU) use a **warp- or
block-per-sequence** mapping with a shared work queue for long sequences
(Exercise 5). For a teaching kernel over short sequences, thread-per-sequence is
the clearest correct mapping.

---

## Numerical considerations

- **All-integer arithmetic.** BLOSUM62 entries are `int8`; HSP scores are sums of
  them in `int`. There is **no floating point anywhere** in the scoring path, so
  there is no rounding, no FMA reordering, and no associativity worry. A 120-aa
  query scores at most a few thousand — nowhere near `INT_MAX`, so no overflow.
- **No atomics → deterministic.** Each thread owns one output; nothing is summed
  across threads. So the GPU result is **bit-identical run to run** (PATTERNS.md
  §3), and identical to the CPU.
- **Deterministic ties.** When two sequences tie on score, `main.cu` breaks the
  tie by **lower index**, so the reported top-K is a total order and stdout is
  byte-stable (e.g. `decoy_1` before `decoy_2`, both 0).
- **Encoding totality.** `encode_residue` maps any unknown byte to `X`, and
  `pack_kmer` refuses to seed on a window containing `X`, so malformed input
  never produces an invalid index or an out-of-range matrix access.

---

## How we verify correctness

Three layers, strongest first:

1. **CPU == GPU, exactly.** `blast_cpu()` and `blast_kernel` call the **same**
   `gapless_xdrop()` and `pack_kmer()` from `src/blast_core.h` /
   `src/reference_cpu.h`, marked `__host__ __device__` so one source compiles for
   both. With integer scores and identical code, the only difference is the
   *order* sequences are processed — which cannot change a per-sequence maximum.
   So we verify with an **exact** integer comparison: `max |cpu − gpu| == 0`
   (PATTERNS.md §4, the "exact" tolerance class). No floating-point epsilon is
   needed or used.
2. **Known-answer recovery.** The synthetic database embeds designed homologies
   (a near-identical copy, a 25%-diverged copy, a 40-residue shared domain, and
   random decoys). A correct search must rank them
   `hit_close > hit_medium > hit_domain > decoys`, which `demo/expected_output.txt`
   pins. The `hit_domain` case specifically validates that the search is **local**
   (it finds a buried shared region inside otherwise-random flanks).
3. **Determinism gate.** `demo/run_demo` diffs stdout byte-for-byte against the
   captured expected output; any nondeterminism or regression fails the demo.

Edge cases covered: empty seed list (a DB sequence shorter than `k`, or all-decoy
windows) → best score 0; the `cudaMalloc` of a zero-length seed array is guarded.

---

## Where this sits in the real world

This is a **reduced-scope teaching version** (CLAUDE.md §13). Production homology
search adds, in roughly this order:

- **Gapped extension.** BLAST re-scores promising HSPs with **gapped**
  Smith-Waterman (allowing indels) for a final alignment. That DP recurrence is
  flagship **3.01**; Exercise 3 wires it on top of this seed stage.
- **Smarter seeds.** DIAMOND uses **spaced seeds** and a **reduced amino-acid
  alphabet** to catch diverged homologs a contiguous 4-mer misses; BLAST uses a
  **two-hit** rule (require two nearby seeds before extending) to cut wasted work.
  Exercises 1–2.
- **Statistics.** Karlin–Altschul **E-values** turn a raw score into a
  significance estimate that accounts for database size — the number users
  actually filter on. Exercise 4.
- **Profiles.** PSI-BLAST and HHsearch iterate: build a **profile / HMM** from the
  first round's hits and search again, dramatically increasing remote-homology
  sensitivity. (Profile-HMM Viterbi is project 3.28.)
- **GPU at scale.** **MMseqs2-GPU** (2025, Nature Methods) is the production
  realization of this project's idea: a GPU-parallel gapless prefilter over the
  entire database, ~20× faster and ~71× cheaper than a 128-core CPU, with a
  warp/block-per-sequence mapping and CUDA streams overlapping I/O with compute.
  **NVIDIA NIM** exposes it as a cloud API for protein-design and AlphaFold-style
  pipelines, where MSA construction is the dominant cost.

The takeaway: the embarrassingly-parallel **seed scan + ungapped extension** you
see here *is* the part that GPUs accelerated to transform structure prediction —
the rest of BLAST is sensitivity and statistics layered on top of this core.
