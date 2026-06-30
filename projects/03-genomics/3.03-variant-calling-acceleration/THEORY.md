# THEORY — 3.3 Variant Calling Acceleration

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

**Variant calling** answers a clinically central question: at each position of a
person's genome, how does their DNA differ from the reference — a SNP
(single-base change), a small insertion, a deletion? Those differences (variants)
are what link a genome to disease risk, drug response, and diagnosis.

The raw input is **sequencing reads**: short DNA fragments (~100–250 bases for
Illumina) sampled from many copies of the genome, each carrying **sequencing
errors**. Modern germline callers — GATK's **HaplotypeCaller**, Google's
**DeepVariant**, and their GPU re-implementations in **NVIDIA Parabricks** — do
*not* call variants base-by-base. Instead, around each candidate site they:

1. **Find active regions** — windows that look different from the reference.
2. **Locally assemble candidate haplotypes** — short hypothesised versions of the
   local genome (e.g. "reference", "reference with this SNP", "reference with that
   1-bp deletion"). This is a small De Bruijn-graph assembly over the reads.
3. **Score every read against every candidate haplotype** — compute
   `P(read | haplotype)`, the probability the read was sequenced from that
   haplotype given the error model. **This is the PairHMM forward algorithm and
   it is the dominant runtime cost** of the whole pipeline.
4. **Compute genotype likelihoods** from those per-read likelihoods and emit a
   genotype (homozygous reference, heterozygous, homozygous alternate).

This project implements **step 3** — the PairHMM forward likelihood — which is
exactly the part that GPUs accelerate (Parabricks turns ~9 hours of CPU germline
calling on a 30× whole genome into under 10 minutes on a datacentre GPU, using
*GATK-identical math*). Steps 1, 2, and 4 are described in §7.

Our headline output is a small, checkable proxy for "calling": for each read we
report the **most-likely haplotype**. In the synthetic sample all reads come from
one truth haplotype, so a correct PairHMM assigns them all back to it.

## 2. The math

### 2.1 The pair Hidden Markov Model

Aligning a read `r = r_1 … r_R` to a haplotype `h = h_1 … h_H` is modeled by a
Hidden Markov Model with **three hidden states** per step:

- **M** (Match/mismatch): a read base is aligned to a haplotype base.
- **I** (Insertion): a read base that is *not* in the haplotype (read got ahead).
- **D** (Deletion): a haplotype base *missing* from the read (haplotype got ahead).

The **forward algorithm** computes the total probability of the read summed over
**all** alignment paths through this HMM — not the single best path (that would be
Viterbi), but the marginal likelihood `P(r | h)`.

### 2.2 Emission probabilities

A base read with Phred quality `Q` has error probability

```
e = 10^(-Q/10).
```

When read base `r_i` is in the **M** state against haplotype base `h_j`:

```
emission(i, j) = 1 - e        if r_i == h_j   (read correctly)
               = e / 3        if r_i != h_j   (an error to one of 3 other bases)
```

(`src/pairhmm_core.h::base_emission_prob`.) Insertions and deletions emit with
probability 1 in this simplified model (GATK folds insertion-emission into the
gap penalties; see §7).

### 2.3 Transition probabilities

With gap-open `δ` and gap-extend `ε`:

```
P(M→M) = 1 - 2δ      P(M→I) = δ      P(M→D) = δ
P(I→M) = 1 - ε       P(I→I) = ε
P(D→M) = 1 - ε       P(D→D) = ε
```

(`PairHmmParams` in `pairhmm_core.h`.) The real GATK model derives `δ` per base
from insertion/deletion quality scores; we use constants (a documented
simplification, §7).

### 2.4 The recurrence

Let `M[i][j], I[i][j], D[i][j]` be the forward probabilities of being in each
state having consumed read prefix `r_1..i` and haplotype prefix `h_1..j`:

```
M[i][j] = emission(i,j) · ( P(M→M)·M[i-1][j-1] + P(I→M)·I[i-1][j-1] + P(D→M)·D[i-1][j-1] )
I[i][j] =                  P(M→I)·M[i-1][j]   + P(I→I)·I[i-1][j]
D[i][j] =                  P(M→D)·M[i][j-1]   + P(D→D)·D[i][j-1]
```

This is exactly `pairhmm_core.h::pairhmm_step`, called identically by the CPU
reference and the GPU kernel.

**Initialisation** (GATK convention): the read may start anywhere along the
haplotype, so row 0's deletion mass is spread uniformly,
`D[0][j] = 1/H` for `j ≥ 1`, with `M[0][j] = I[0][j] = 0` and the `j=0` boundary
column all zero. **Termination**: the likelihood is the sum over the last read
row, `Σ_j ( M[R][j] + I[R][j] )` — every way the read could finish aligned to the
haplotype. We report `log10` of this sum because the raw probability underflows
fast.

## 3. The algorithm

Filling one `(R+1)×(H+1)` table is **O(R·H)** time. Each cell reads three already-
computed neighbours (up-left, up, left), so the dependency is a classic 2-D DP.
We keep only **two rolling rows** (previous + current), reducing memory from
`O(R·H)` to **O(H)** per pair — the same trick in `reference_cpu.cpp` and the GPU
kernel.

For `n_reads` reads and `n_haps` haplotypes there are `P = n_reads · n_haps`
**independent** tables. Total serial work is `O(P · R · H)`. Crucially, the `P`
tables share no data dependencies — that independence is the entire source of GPU
parallelism.

| Quantity | Cost |
|---|---|
| One DP table (forward) | `O(R·H)` time, `O(H)` memory (two rows) |
| Full problem (serial) | `O(n_reads·n_haps·R·H)` |
| Parallel **depth** (per pair) | `O(R+H)` if you wavefront a single table; `O(R·H)` if one thread fills it serially |
| Parallel **width** | up to `P` pairs at once |

## 4. The GPU mapping

This project uses the **independent-jobs** pattern (PATTERNS.md §1, exemplified by
`1.12` and `12.01`): **one CUDA thread per (read, haplotype) pair**, each thread
filling its whole DP table serially with the two-row scheme.

- **Thread-to-data map.** Pair index `p = blockIdx.x·blockDim.x + threadIdx.x`,
  decoded to `read = p / n_haps`, `hap = p % n_haps`. A guard `if (p >= n_pairs)
  return;` covers the ragged last block.
- **Launch config.** `block = 128` threads (a multiple of the 32-lane warp);
  `grid = ceil(P / 128)`. 128 (not 256) keeps register/local-memory pressure
  manageable because each thread holds two rows of `PairHmmCell` (3 doubles each).
- **Memory hierarchy.** Reads, qualities, and haplotype bytes live in **global
  memory** (read-only, `__restrict__`). The DP rows live in **per-thread local
  memory** (`PairHmmCell prev[]`, `cur[]`) — effectively each thread's private
  scratch. No shared memory, no atomics, no `__syncthreads`: the pairs are
  independent, so there is nothing to coordinate.
- **No CUDA library.** PairHMM has no off-the-shelf primitive; the kernel is
  hand-written. The only library is the CUDA runtime (`cudart`) for memory and
  launch. (Contrast `8.03`/`2.06`, which lean on cuFFT/cuSOLVER.)

```
   pairs P = n_reads x n_haps           one thread per pair
   ┌───────────────────────────┐        ┌──────────────────────────────┐
   │ p0=(r0,h0) p1=(r0,h1) ...  │  --->  │ thread p fills its own DP     │
   │ pk=(r1,h0) ...             │        │ table:  (R+1) x (H+1) cells   │
   │ ...                        │        │ kept as 2 rolling rows in     │
   └───────────────────────────┘        │ local memory; O(H) storage    │
        grid of 128-thread blocks        └──────────────────────────────┘
```

### Why this mapping (and the production refinement)

One-thread-per-pair is the clearest teaching mapping and is already correct and
embarrassingly parallel. The **production** refinement (Parabricks, the GATK
`PairHMM` AVX/GPU kernels) instead gives **one thread BLOCK per pair** and fills
the single DP table along **anti-diagonals** (the wavefront of `3.01`), staging
the active diagonals in **shared memory** so the `O(R·H)` work of one table is
itself parallel across the block's threads. That cuts each pair's latency from
`O(R·H)` to `O(R+H)` and raises arithmetic intensity. We keep the simpler mapping
because the goal is to make the *independence of pairs* and the *recurrence*
unmistakable; the wavefront idea is left as an exercise and demonstrated in
project 3.01.

## 5. Numerical considerations

- **Precision: FP64 throughout.** Forward probabilities are products of many
  small numbers; FP32 loses too many bits. All cells, transitions, and emissions
  are `double`. This is also what makes verification tight.
- **Underflow.** Even in FP64 the raw `Σ(M+I)` shrinks with read length. We take
  `log10` of the final sum for reporting. (GATK's production kernel additionally
  rescales each row to avoid intermediate underflow for 250-bp reads; our
  20–100-bp teaching reads do not need it — noted as an exercise.)
- **Determinism.** Each thread does a *serial* fill with no atomics and no
  cross-thread reduction, so there is **no floating-point reordering** — the GPU
  result is bit-reproducible run to run. Because the CPU and GPU call the *same*
  `pairhmm_step`, they execute identical IEEE-754 operations in the same order, so
  they agree to a few ULP (not merely "close").
- **No race conditions.** Disjoint output cells (`loglik[p]`), disjoint scratch
  (per-thread local rows). The only shared reads are the immutable input arrays.

## 6. How we verify correctness

Two independent checks:

1. **GPU vs. CPU reference.** `reference_cpu.cpp` fills every table with a plain
   serial loop and the shared `pairhmm_step`. `main.cu` compares the two full
   `n_reads×n_haps` `log10`-likelihood matrices with `max_abs_err` (matching
   `-inf` impossible-pairs treated as equal). The tolerance is **`1.0e-9`**, and
   the *measured* error is ~`1.8e-15` — essentially machine epsilon — exactly as
   PATTERNS.md §4 predicts for "the same exact operations on both sides". The
   tolerance is a generous safety net, not a fudge factor.
2. **The science check.** The synthetic reads are all drawn from haplotype 0, so
   the PairHMM should assign every read's argmax to haplotype 0. The demo prints
   "reads assigned to truth haplotype 0: 8 of 8" — a sanity check that the math
   *means* the right thing, not just that two implementations agree.

Edge cases handled: the ragged last thread block (`p >= n_pairs`), the `sum == 0`
impossible pair (both sides emit `-inf`), and a compile-time cap `hap_len ≤ 127`
(the kernel refuses longer haplotypes rather than overrun its fixed local rows).

## 7. Where this sits in the real world

This is a **reduced-scope teaching version** of one stage. Production germline
calling (GATK HaplotypeCaller, NVIDIA Parabricks, DeepVariant, Clair3) adds:

- **Base Quality Score Recalibration (BQSR)** upstream, so the `Q` values fed to
  the emission model are empirically corrected (project 3.25 covers this).
- **Local de novo assembly** of active regions (a De Bruijn graph over reads) to
  *generate* the candidate haplotypes — here we are handed them.
- **Per-base gap penalties.** Real PairHMM derives `δ`/`ε` from per-base
  insertion/deletion quality scores and a context-dependent gap-continuation
  penalty, not the constants we use.
- **Row rescaling** to keep 250-bp reads from underflowing, and a fused
  log-sum-exp termination.
- **Genotype likelihoods (GL/PL)** and joint genotyping downstream of the
  likelihood matrix, then VCF emission.
- **DeepVariant's** entirely different approach: render the pileup as an image and
  score it with a CNN (cuDNN inference) — a candidate for batched GPU inference
  rather than a hand-written DP kernel.
- **Engineering:** one block per pair, shared-memory anti-diagonal tables, CUDA
  streams overlapping I/O with compute, and multi-GPU pipeline parallelism
  (BQSR → alignment → calling).

The math we implement is the *same forward recurrence* those tools spend most of
their time on; the rest is scale, accuracy, and plumbing.

---

## References

- **GATK / HaplotypeCaller** — https://github.com/broadinstitute/gatk — the CPU
  reference whose PairHMM math production GPU callers reproduce bit-for-bit. Read
  the `pairhmm` package for the exact recurrence and rescaling.
- **NVIDIA Parabricks** — https://docs.nvidia.com/clara/parabricks/latest/ —
  GATK-identical GPU HaplotypeCaller; the source of the "9 h → <10 min" figure and
  the one-block-per-pair PairHMM design.
- **DeepVariant** — https://github.com/google/deepvariant — the CNN-pileup
  alternative; shows where cuDNN inference replaces a DP kernel.
- **Clair3 / Clairvoyante** — https://github.com/HKU-BAL/Clair3 — deep-learning
  caller with GPU inference, strong on long reads.
- **Durbin, Eddy, Krogh, Mitchison, _Biological Sequence Analysis_ (1998)** — the
  canonical derivation of pair-HMMs and the forward algorithm (Ch. 4).
- **GiaB truth sets** — https://www.nist.gov/programs-projects/genome-bottle —
  how callers are benchmarked for accuracy.
