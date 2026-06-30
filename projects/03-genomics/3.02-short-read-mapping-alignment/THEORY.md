# THEORY — 3.2 Short-Read Mapping / Alignment

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

A DNA-sequencing machine cannot read a chromosome end to end. Instead it shreds
the sample into millions of short fragments and reads each one, producing
**reads**: strings of `A/C/G/T` typically 50–300 bases long (Illumina). To learn
anything — where a mutation sits, how strongly a gene is expressed, which microbe
is present — you must first put each read **back where it came from** on a known
**reference genome**. That placement step is **short-read mapping** (a.k.a.
alignment), and it is the first stage of essentially every genomics pipeline.

The difficulty is scale and noise:

- **Scale.** A human genome is ~3.2 billion bases. At 30× coverage you sequence
  ~900 million reads. Each read could in principle match anywhere, so naively you
  would compare every read against every genome position — astronomically slow.
- **Noise.** Reads contain sequencing errors, and the sampled individual differs
  from the reference by millions of SNPs and indels. So mapping is not exact
  string search; it is *best approximate* placement.

The standard answer is **seed-and-extend**: cheaply find a few exact "anchor"
matches (seeds), then do expensive alignment only near those anchors. This
project implements that idea on the GPU, with the honest simplifications spelled
out in §7.

## 2. The math

**Inputs.**

- A reference $R = r_0 r_1 \dots r_{L_\text{ref}-1}$, each $r_i \in \{A,C,G,T\}$.
- A batch of $N$ reads, each $Q^{(t)} = q^{(t)}_0 \dots q^{(t)}_{L-1}$ of length
  $L$ (uniform here; see §7).

**Per-read objective (ungapped).** For a read $Q$ placed at reference offset
$p$ (read base 0 over $r_p$), define the **ungapped alignment score**

$$
S(p) \;=\; \sum_{b=0}^{L-1} s\!\left(q_b,\, r_{p+b}\right),
\qquad
s(x,y) = \begin{cases} +1 & x = y \;(\text{match}) \\ -1 & x \neq y \;(\text{mismatch}) \end{cases}
$$

valid only for $0 \le p \le L_\text{ref} - L$ (the read must fit inside $R$). The
read's mapping is

$$
p^\star = \arg\max_{p} S(p),
$$

with ties broken by the smallest $p$ (a deterministic rule we share between CPU
and GPU). The number of mismatches at $p^\star$ is the **edit count**; here
score and edits are linked, $S = L - 2\cdot(\text{mismatches})$, because every
base is either +1 or −1.

**Seeding restricts the search.** Evaluating $S(p)$ at all $L_\text{ref}-L+1$
offsets is wasteful. A length-$k$ **k-mer** is $k$ consecutive bases. We use the
read's leading k-mer $Q[0{:}k]$ and only consider offsets $p$ where the reference
contains that exact k-mer, i.e.

$$
\text{candidates}(Q) = \{\, p : R[p{:}p{+}k] = Q[0{:}k] \,\}.
$$

Each base packs into 2 bits ($A{=}00,C{=}01,G{=}10,T{=}11$), so a k-mer is a
single integer **code** $\in [0,\,4^{k})$. With $k=12$ that is a 24-bit code and
$4^{12}\approx 1.7\times10^7$ possible k-mers — enough that a 12-mer is nearly
unique in a small reference, so the candidate set is tiny.

## 3. The algorithm

**Phase A — build the index (once).** For each reference window
$w = 0 \dots L_\text{ref}-k$, compute its k-mer code $c_w$. Sort the pairs
$(c_w, w)$ by code. Now all offsets sharing a given code form a **contiguous run**,
found by two binary searches (lower/upper bound). Cost: $O(L_\text{ref}\log L_\text{ref})$
once, reused for every read.

**Phase B — map each read.**

1. Compute the read's seed code $c_Q = \text{code}(Q[0{:}k])$. — $O(k)$
2. Binary-search the sorted index for the run $[lo, hi)$ of offsets with code
   $c_Q$. — $O(\log L_\text{ref})$
3. For each candidate offset $p$ in that run, compute $S(p)$ and keep the best
   (highest score, lowest $p$). — $O(C \cdot L)$ where $C = hi-lo$.

**Complexity.**

| | Serial (CPU) | Parallel (GPU) |
|---|---|---|
| Index build | $O(L_\text{ref}\log L_\text{ref})$ | (done on host here) |
| Map all reads | $O\!\big(N\,(\log L_\text{ref} + C L)\big)$ | **work** the same; **depth** $O(\log L_\text{ref} + C L)$ per thread, all $N$ reads concurrent |

The naive all-positions alternative is $O(N\,L_\text{ref}\,L)$. Seeding replaces
the $L_\text{ref}$ factor with $C$ (a handful of candidates) — the whole point.

**Arithmetic intensity / access pattern.** Extension is memory-light and compute-
light: per candidate it reads $L$ reference bases and $L$ read bases and does $L$
compares. The read row is reused across all its candidates (keep it hot). The
index binary search is a logarithmic random walk through global memory — the one
latency-bound step, hidden by running many warps (occupancy).

## 4. The GPU mapping

**Thread-to-data mapping.** *One thread maps one read.* Thread
$r = \text{blockIdx.x}\cdot\text{blockDim.x} + \text{threadIdx.x}$ owns read $r$,
and a **grid-stride loop** ($r \mathrel{+}= \text{blockDim.x}\cdot\text{gridDim.x}$)
lets a fixed grid cover any $N$. Each thread runs the *entire* Phase-B pipeline
for its read and writes one $(pos, score, mism)$ triple to its own output slot.

```
reads (row-major, N x L):            sorted k-mer index (length L_ref-k+1):
  read 0  ───► thread 0                codes:   [ c0 c1 c2 ... ]  (ascending)
  read 1  ───► thread 1                offsets: [ o0 o1 o2 ... ]
  read 2  ───► thread 2
   ...                                 thread r: binary-search codes for seed(r),
  read N-1 ─► thread N-1                          score read r at each offset in
                                                  [lo,hi), keep best -> out[r]
  grid-stride: a fixed grid of B*G threads sweeps all N reads.
```

**Launch configuration.** `block = 256` threads (8 warps — a warp multiple that
gives the scheduler enough warps to hide the binary-search latency, with good
occupancy on sm_75–sm_89). `grid = min(1024, ceil(N/256))` blocks; the grid-stride
loop handles the remainder. (See `kernels.cu`.)

**Memory hierarchy.**

- **Global memory** holds the reference, the reads, the sorted index, and the
  outputs. Consecutive threads (consecutive reads) touch *different* read rows, so
  perfect coalescing is not automatic — a known trade-off; real mappers transpose
  reads or use a warp-per-read layout (see §7, exercise 5).
- **Registers** hold each thread's running best $(pos, score, mism)$ — the
  reduction is private, so **no shared memory and no atomics** are needed.
- A natural optimization (left as study material) is to stage the query read or a
  reference tile in **shared memory** when many candidates reuse it.

**Which CUDA library does what (no black boxes).** We *sort* the small reference
index on the **host** with `std::stable_sort`; at genome scale you would sort the
k-mers on the device with **Thrust** (`thrust::sort_by_key`) or **CUB**
(`DeviceRadixSort`) — a radix sort over 2-bit-packed integer keys, exactly the
sorted-array layout our binary search consumes. The catalog also lists **cuSPARSE**
(framing index look-ups as sparse gathers) and **NCCL** (sharding reads across
multiple GPUs). Hand-rolling the device sort would mean writing a multi-pass LSD
radix sort with per-block histograms and a global prefix scan — correct but easy
to get subtly wrong, which is why production code calls the library.

## 5. Numerical considerations

This project is **all integers** — base codes, k-mer codes, and scores are
exact. There is **no floating point**, hence no rounding, no FMA divergence, no
precision question. Consequences:

- **Determinism.** Each thread's max-reduction touches only its own registers and
  its own output slot; there are no cross-thread atomics and no reordered float
  sums. The output is therefore **bit-identical run to run** and independent of
  block/grid size (PATTERNS.md §3). That is what lets `demo/run_demo` diff stdout.
- **Tie-breaking parity.** Both CPU and GPU scan candidates in the index's
  ascending-offset order and replace the best only on a *strict* score increase,
  so the *same* lowest offset wins a tie on both sides.
- **Overflow.** A 12-mer code needs 24 bits; we store it in `uint64_t` with vast
  headroom. Scores are bounded by $\pm L$. No overflow is possible at these sizes.
- **Off-end placements** return a large negative sentinel so they can never win
  the max — guarding the ragged ends without a special case in the reduction.

## 6. How we verify correctness

The CPU reference (`map_reads_cpu` in `reference_cpu.cpp`) is written to be
obviously correct: plain serial loops, the same `build_index`, and the **same**
`__host__ __device__` `score_window`/`kmer_equal_range`/`kmer_code` the kernel
calls. `main.cu` maps every read on both engines and demands

$$
\text{pos}_\text{CPU}^{(r)} = \text{pos}_\text{GPU}^{(r)}, \quad
\text{score}_\text{CPU}^{(r)} = \text{score}_\text{GPU}^{(r)}, \quad
\text{mism}_\text{CPU}^{(r)} = \text{mism}_\text{GPU}^{(r)}
$$

for **every** read — exact equality, **tolerance = 0** (the right choice for an
all-integer computation; PATTERNS.md §4). Because the two implementations are
independent in *control flow* (serial loop vs. thousands of threads) but identical
in *arithmetic*, their agreement is strong evidence the parallel version is
correct.

**A second, stronger check — the embedded known answer.** The synthetic sample
(`make_synthetic.py`) builds reads by copying the reference at **known** positions
with a **known** number of mutations (and one pure-noise read). So we can check
the science, not just CPU==GPU: read $i$ must map to its construction position
with score $40 - 2\cdot(i \bmod 4)$, and the noise read must be `UNMAPPED`. The
demo output shows exactly this (`pos 0,22,44,…`, scores `40,38,36,34,…`).

**Edge cases exercised.** A read whose seed is absent (→ `UNMAPPED`, the empty
binary-search run); reads with the maximum mutation count; placements at the very
start of the reference (`pos 0`).

## 7. Where this sits in the real world

Production mappers (BWA-MEM, Bowtie2, minimap2, **NVIDIA Parabricks**) keep the
seed-and-extend skeleton but add, at every step, what this teaching version omits:

- **Index: FM-index / BWT, not a sorted array.** The Burrows-Wheeler transform
  plus rank/select structures (an **FM-index**) supports *backward search* of
  variable-length exact matches in space close to the genome itself — essential
  for 3.2 GB. Parabricks runs this backward search as a parallel BFS across thread
  groups. Our sorted-k-mer array is the same *idea* (find exact seed occurrences)
  in a form a student can read in one sitting.
- **Seeding: minimizers and multiple seeds.** Real tools seed with **minimizers**
  (a sampled, canonical subset of k-mers) to shrink the index, take **many** seeds
  per read, and **chain** co-linear seeds (a sparse dynamic program) to reject
  spurious single hits before extending.
- **Extension: banded *gapped* Smith-Waterman.** To handle insertions/deletions,
  the extend step is a banded SW alignment producing a full **CIGAR** string —
  precisely the anti-diagonal wavefront kernel of
  [project 3.01](../3.01-smith-waterman-needleman-wunsch-alignment/). On the GPU
  each read's extension gets a thread block with a shared-memory score matrix.
- **Both strands, paired ends, mapping quality (MAPQ), and duplicate marking**
  (`markduplicates` hashing) round out a real pipeline.
- **Scale-out.** Sort seeds with **Thrust/CUB** on-device, shard reads across
  GPUs with **NCCL**. This is how Parabricks v4.7 maps a 30× human genome in
  <10 minutes on an H100 vs. >30 CPU-hours for BWA-MEM.

This project is the faithful Beginner-level core: **exact seeding via a sorted
k-mer index + ungapped extension, one thread per read**, with the gapped/FM-index/
chaining upgrades described above and demonstrated (for the SW part) in 3.01.

---

## References

- H. Li, *Aligning sequence reads, clone sequences and assembly contigs with
  BWA-MEM*, arXiv:1303.3997 (2013) — the seed-chain-extend algorithm this project
  miniaturizes.
- H. Li, *Minimap2: pairwise alignment for nucleotide sequences*, Bioinformatics
  (2018) — minimizer seeding and chaining, the modern long/short-read approach.
- P. Ferragina & G. Manzini, *Opportunistic data structures with applications*,
  FOCS (2000) — the FM-index/BWT backward search behind real genome indexes.
- **NVIDIA Parabricks** docs (<https://docs.nvidia.com/clara/parabricks/latest/>)
  — how BWA-MEM + GATK are reimplemented in CUDA for whole-genome scale.
- **CUSHAW3** (<https://github.com/asbschmidt/CUSHAW3>) — GPU banded-SW seed
  extension; the natural next step beyond our ungapped extend.
- **GenomeWorks** (<https://github.com/NVIDIA-Genomics-Research/GenomeWorks>) and
  **Scrooge** (<https://github.com/CMU-SAFARI/Scrooge>) — GPU mapping/overlap
  kernels and GPU/CPU co-design.
