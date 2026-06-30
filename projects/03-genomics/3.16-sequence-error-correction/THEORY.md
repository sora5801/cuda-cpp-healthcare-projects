# THEORY — 3.16 Sequence Error Correction

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

A **DNA sequencer** does not read a genome end-to-end. It shears the genome into
fragments and reads many short, overlapping pieces — **reads** — each typically
100–250 bases for short-read (Illumina) technology. Because the same genomic
position is covered by many reads (the **coverage**, e.g. 30×), the true sequence
is heavily redundant in the data.

Every base call is a physical measurement and so carries an **error probability**
(≈0.1–1% for Illumina substitutions; much higher, 5–15%, and indel-dominated for
ONT/PacBio). Downstream steps — genome **assembly**, **variant calling** — are
extremely sensitive to these errors: an erroneous base can fork an assembly graph
or masquerade as a mutation. **Error correction** is the pre-processing step that
uses the redundancy across reads to fix wrong bases *before* assembly.

The key biological insight that makes correction possible: **a real genomic
k-mer (length-*k* substring) appears in many overlapping reads, but a k-mer that
contains a sequencing error appears in essentially only the one read where the
error occurred.** Frequency separates signal from noise.

```
genome:    ... A C G T A C G T A C ...
read 1:        A C G T A C            (true)
read 2:          C G T A C G          (true)
read 3:        A C G A A C            (error at pos 4: T->A)
            the 4-mer "GTAC" recurs (trusted);  "GAAC" appears once (suspect)
```

## 2. The math

**Inputs.** A set of reads $R = \{r_1, \dots, r_n\}$ over the alphabet
$\Sigma = \{A,C,G,T\}$ (with $N$ for "no call"). A k-mer length $k$ and a trust
threshold $T \in \mathbb{Z}^{+}$.

**The k-mer spectrum.** For a read $r$ of length $L$, its k-mers are the $L-k+1$
length-$k$ windows $r[p..p{+}k)$ for $p = 0 \dots L-k$. Encode each k-mer as a
base-4 integer

$$\text{code}(w) = \sum_{i=0}^{k-1} 4^{\,k-1-i}\, \beta(w_i), \qquad
  \beta(A)=0,\ \beta(C)=1,\ \beta(G)=2,\ \beta(T)=3,$$

so $\text{code}(w) \in [0, 4^k)$. The **spectrum** is the histogram

$$\text{count}[c] = \big|\{(r,p) : \text{code}(r[p..p{+}k)) = c\}\big|.$$

A k-mer is **trusted** iff $\text{count}[c] \ge T$.

**The correction objective.** For each read, find a small set of base
substitutions that maximizes the number of trusted k-mers covering the read
(ideally: make every k-mer trusted), under the prior that errors are rare
(few substitutions). Solving this exactly over a read is a shortest-path /
dynamic-program problem; the teaching version uses a fast, deterministic greedy
heuristic (§3).

**Why $T$ works.** Under coverage $\bar c$ and per-base error rate $\varepsilon$,
a true k-mer's expected count is $\approx \bar c\,(1-\varepsilon)^k$ (large), while
an error k-mer's count is $O(1)$. Any $T$ between these — here $T=3$ at
$\bar c \approx 18$ — cleanly separates them.

## 3. The algorithm

**Phase 1 — build the spectrum.** Slide the window over every read and increment
`count[code]`. Cost: $O(B)$ where $B = \sum_i (L_i)$ is the total number of bases
(each k-mer code is built incrementally or, as in our teaching code, recomputed in
$O(k)$ — $O(Bk)$, still linear in the data for fixed $k$).

**Phase 2 — correct each read** (the greedy single-pass heuristic in
`correct_one_read`, shared by CPU and GPU):

```
for p = 0 .. L-k:                      # each k-mer start, left to right
    c = code(read[p..p+k))
    if c is valid and trusted: continue   # window already good
    # window untrusted -> the left-most unlocked base (read[p]) is the suspect
    for each alternative base x at position p (A,C,G,T order, x != current):
        if code(read[p:=x .. p+k)) is trusted:
            remember x with the highest spectrum count
    commit the best trusted substitution (or leave the base unchanged)
```

- **At most one substitution per window**, decided left-to-right so a fixed base
  helps the next overlapping window. Cost per read: $O(L \cdot k \cdot |\Sigma|)$.
- **Determinism:** the A,C,G,T scan order and the "highest count wins" tie-break
  make the output a pure function of the input + spectrum — identical on CPU/GPU.

**Complexity, serial vs parallel.**

| Phase | Serial work | Parallel depth (GPU) |
|---|---|---|
| 1 Count | $O(Bk)$ | $O(\max_i L_i \cdot k)$ (one thread per read; atomics serialize collisions in hardware) |
| 2 Correct | $O(B k |\Sigma|)$ | $O(\max_i L_i \cdot k |\Sigma|)$ (one thread per read, no inter-read dependency) |

Both phases are $O(n)$ in the number of reads with full data parallelism — the
reason the GPU wins at scale.

## 4. The GPU mapping

Two kernels, both launched as **one thread per read**
(`grid = ceil(n / 256)`, `block = 256`):

```
            reads (n)                         spectrum table (4^k = 262144 slots)
   r0  r1  r2  r3 ... r(n-1)                  [ c0 | c1 | c2 | ... ]
    |   |   |   |       |                              ^   ^
  t0  t1  t2  t3 ...  t(n-1)   --phase1-->  atomicAdd(&count[code], 1)   (collisions
    each thread slides its read's window      from many threads -> hardware-serialized)
                                            ----------------------------------------
                                  --phase2-->  each thread READS count[] (frozen),
                                               runs correct_one_read(), writes its
                                               own output slice (no races, no atomics)
```

**Memory hierarchy.**
- **Global memory** holds the concatenated reads (CSR layout: one flat `bases`
  array + `offset`/`length`), the spectrum table, and the corrected output. The
  CSR layout means the whole dataset is a few contiguous device arrays — ideal
  for `cudaMemcpy` and for coalesced-ish access.
- **The spectrum table (4^9 = 262 144 × 4 B ≈ 1 MB)** lives in global memory. In
  phase 2 it is read-only and heavily reused, so the L2 cache absorbs most of the
  traffic. (A real corrector's hash table is far larger and is the bandwidth
  bottleneck — see §7.)
- **Atomics, phase 1.** Many reads share true k-mers, so many threads target the
  same `count[]` slot. `atomicAdd` serializes those collisions in hardware. The
  classic optimization is a **per-block shared-memory sub-histogram** merged once
  at the end (Exercise 2); we keep global atomics because they are the clearest
  to learn from.
- **No CUDA library is used.** This is a hand-written histogram + a hand-written
  per-read scan, deliberately, so nothing is a black box. The catalog mentions
  cuBLAS/GEMM for *MSA scoring* — that belongs to the long-read POA regime (§7),
  not to short-read k-mer correction.

**Launch config reasoning.** 256 threads/block is a multiple of the 32-lane warp,
gives the scheduler 8 warps to hide global-memory latency, and keeps many blocks
resident for occupancy on sm_75…sm_89. The grid is tiny here ($n=120$ →
1 block); at real scale ($n=10^8$) it is hundreds of thousands of blocks — the
same code, unchanged.

## 5. Numerical considerations

- **Everything is integer / byte work.** k-mer codes are `uint32`, counts are
  `uint32`, corrected bases are `char`. There is **no floating point** anywhere in
  the computation, so there is nothing to round and no FMA divergence between host
  and device.
- **`uint32` count width.** A single 9-mer's count stays far below $2^{32}$ even
  at extreme coverage, so the histogram never overflows. (If you raise coverage to
  the billions, widen to `uint64`.)
- **Determinism of the atomic histogram.** Integer addition is associative *and*
  commutative, so the final `count[]` table is independent of the order in which
  the atomics land — the GPU spectrum equals the serial CPU spectrum **bit for
  bit**, every run (PATTERNS.md §3: integer atomics are deterministic; float
  atomics would not be). This is the whole reason we count in integers.
- **No phase-2 races.** Each thread writes only its own read's output slice and
  only *reads* the frozen spectrum, so phase 2 needs no atomics and has no data
  races. A `cudaDeviceSynchronize` (inside `CUDA_CHECK_LAST`) between the phases
  guarantees the spectrum is complete before any thread reads it.

## 6. How we verify correctness

Two independent checks, both in `main.cu`:

1. **GPU == CPU, exactly.** `build_spectrum_cpu` / `correct_reads_cpu` are a
   plain serial baseline; the kernels are the parallel twin. We compare the two
   spectra slot-by-slot (`spectrum_mismatch`) and the two corrected-read buffers
   byte-by-byte (`corrected_mismatch`) and require **both to be 0** — an *exact*
   equality, not a tolerance. This is possible because the per-element logic
   (`kmer_code_at`, `is_trusted`, `correct_one_read`) is the **same
   `__host__ __device__` code** compiled for both sides (PATTERNS.md §2). If the
   GPU code had a race or an indexing bug, the byte counts would diverge.
2. **The science actually works.** Because the sample is synthetic we know each
   read's error-free truth, so we report errors **before** correction (raw vs
   truth) and **after** (corrected vs truth). In the demo: 132 → 39, i.e. ~70% of
   substitution errors removed. This validates the *method*, not just CPU/GPU
   agreement.

**Edge cases handled:** windows containing `N` (skipped, code = `0xFFFFFFFF`);
reads shorter than $k$ (no k-mers, left unchanged); the ragged last thread block
(guarded by `if (g >= n) return`); degenerate all-untrusted positions (left
unchanged rather than guessed).

Tolerance choice (PATTERNS.md §4): **exact `== 0`**, because the entire pipeline
is integer/byte arithmetic running identically on both sides — anything other than
bit-identity would signal a bug.

## 7. Where this sits in the real world

This is a faithful but **reduced-scope** teaching version. Production correctors
differ in several load-bearing ways:

- **k-mer size & data structure.** Real tools use $k = 15$–$31$. You cannot
  direct-index $4^{31}$ slots, so **CARE**/**BFC** use **GPU hash tables** with
  atomic compare-and-swap (CAS) for insertion — the "GPU hash table with atomic
  CAS" the catalog names. Our $k=9$ exact table trades scale for a collision-free,
  perfectly legible spectrum.
- **Smarter correction.** Real correctors iterate to convergence, correct from
  both directions, weigh **base-quality scores**, and resolve multi-error windows
  via a DP over candidate paths — not our single greedy pass. CARE additionally
  gathers the set of reads overlapping each read (via minhashing) and corrects
  from a multiple alignment, which is far more powerful than spectrum lookup alone.
- **Long reads are a different regime.** For ONT/PacBio (high error, indels), the
  k-mer spectrum breaks down. **racon-GPU** and **CONSENT** instead build a
  **partial-order alignment (POA)** / local de Bruijn graph of overlapping reads
  and take a consensus; **Medaka** replaces the hand-tuned model with an **RNN**
  run on the GPU. The catalog's "cuBLAS/GEMM for MSA scoring" and "one CUDA block
  per read during POA" refer to *this* regime.
- **Scale.** CARE processes **millions of reads per second** by keeping the hash
  table resident in GPU memory and streaming reads through — the throughput story
  our two kernels gesture at but do not chase.

The teaching takeaways transfer directly: the spectrum/trust idea, the
integer-atomic histogram, the one-thread-per-read parallelization, and the
shared-core exact-verification discipline.

---

## References

- Kelley, Schatz & Salzberg, **"Quake: quality-aware detection and correction of
  sequencing errors"**, *Genome Biology* 2010 — the canonical k-mer spectrum
  corrector; read for the coverage/threshold statistics in §2.
- Li, **"BFC: correcting Illumina sequencing errors"**, *Bioinformatics* 2015 —
  a fast, BWT/k-mer hybrid; read for engineering a real spectrum at scale.
- Kallenborn et al., **CARE** (https://github.com/fkallen/CARE) — the CUDA
  short-read corrector this project abstracts; read for GPU hash tables + minhash
  read gathering.
- Vaser et al., **racon** / NVIDIA **racon-GPU**
  (https://github.com/NVIDIA-Genomics-Research/racon-gpu) — read for GPU POA and
  the long-read correction regime.
- **CONSENT** (https://github.com/morispi/CONSENT) and **Medaka**
  (https://github.com/nanoporetech/medaka) — read for local-de-Bruijn and
  RNN-consensus long-read correction respectively.
