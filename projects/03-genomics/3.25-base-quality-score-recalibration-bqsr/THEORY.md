# THEORY — 3.25 Base Quality Score Recalibration (BQSR)

> For a reader who knows C++ but is new to CUDA and to sequencing bioinformatics.
> See [README.md](README.md) for the tour and build. _Educational only._

## 1. The science

A short-read sequencer (Illumina) reads DNA in cycles: at cycle `c` it adds one
labeled nucleotide to each of millions of clusters and images the flash of color.
From the brightness and crosstalk it **calls** a base (A/C/G/T) and emits a
**quality score** `Q` — its confidence that the call is correct. `Q` is a PHRED
score: the machine asserts an error probability

```
P_err = 10^(-Q/10)        Q20 -> 1%   Q30 -> 0.1%   Q40 -> 0.01%
```

Downstream **variant callers** weight evidence by these probabilities, so if `Q`
is wrong the calls are wrong. And `Q` *is* systematically wrong: the basecaller's
internal model does not perfectly capture how the real error rate depends on

- the **cycle** (chemistry degrades along the read → late bases err more),
- the **sequence context** (homopolymers and specific di-nucleotides err more),
- the **reported `Q`** itself (the model is mis-scaled in places).

**BQSR** (GATK's `BaseRecalibrator`) measures the *actual* error rate as a function
of these **covariates** from the data at hand, and rewrites every `Q` to match.
The one subtlety that makes it work: to measure machine error you must not count
**real genetic variation** as error. So BQSR **masks known-variant sites** (dbSNP,
Mills indels): a base that disagrees with the reference at a known SNP is the
sample's true allele, not a miscall, and is skipped.

## 2. The math

Index every base by a covariate tuple `θ = (q, cycle, context)` where `q` is the
reported quality, `cycle ∈ [0, L)` is the position in the read, and `context` is
the di-nucleotide of the reference (previous + current reference base, 16 values,
plus one "no context" slot for the first cycle). For each covariate bin `θ` we
count

```
obs(θ)  = number of (non-masked) bases with covariates θ
err(θ)  = number of those that MISMATCH the reference base
```

A base is an "error" iff its called base ≠ the reference base (and the site is not
a known variant). The **empirical** error probability of a bin is the mismatch
fraction, with a **+1 Yates/Laplace correction** so an all-correct bin is finite
and conservative (GATK does the same):

```
P_emp(θ) = (err(θ) + 1) / (obs(θ) + 1)
Q_emp(θ) = round( -10 · log10 P_emp(θ) )
```

The recalibrated quality of a base is the `Q_emp` of its bin. A bin never observed
(`obs = 0`) has no evidence → the base keeps its original reported `Q`.

> Symbols: `q ∈ [0, NUM_Q)`, `cycle ∈ [0, MAX_CYCLE)`, `context ∈ [0, NUM_CONTEXT)`;
> `obs, err` are non-negative integers; `Q_emp` is an integer PHRED in `[0, 93]`.

## 3. The algorithm

```
init obs[·] = err[·] = 0
for each base b (read r, cycle c):                # the COVARIATE SCAN
    p = ref_pos(r, c)
    if known[p]: continue                          # mask known variant
    if base(b) or ref(p) is 'N': continue          # undefined context
    θ = (q(b), c, dinuc(ref[p-1], ref[p]))
    obs[θ] += 1
    if called(b) != ref[p]: err[θ] += 1            # a sequencing error
for each bin θ: Q_emp[θ] = phred((err+1)/(obs+1))  # EMPIRICAL QUALITY
for each base b: new_q(b) = Q_emp[θ(b)] or q(b)    # RECALIBRATE
```

**Complexity.** The scan is `O(R·L)` in the number of bases (one table increment
each); the empirical-quality step is `O(#bins)`; recalibration is again `O(R·L)`.
For a 30× human genome `R·L ≈ 10^11`, which is why the scan dominates and why it is
worth parallelising. The table itself is tiny (a few thousand bins).

## 4. The GPU mapping

This is the **parallel-assign + atomic-reduce** pattern (PATTERNS.md, flagship
11.09). The scan over bases is embarrassingly parallel *up to the tally*, so:

- **Thread-to-data map.** One thread per base. The global index
  `g = blockIdx.x*blockDim.x + threadIdx.x` decodes to `read = g/L`,
  `cycle = g%L` (the regular `R×L` layout — same trick as 11.09's `event×marker`).
- **Launch config.** `block = 256` threads (a multiple of the 32-lane warp, 8 warps
  to hide latency, good occupancy on sm_75..sm_89); `grid = ceil(R·L / 256)`. The
  ragged last block is guarded by `if (g >= total) return;`.
- **Memory.** Read arrays, the reference, and the known-site mask live in **global
  memory** (read-only, accessed once per base — `__restrict__` lets the compiler
  keep loads in registers). The covariate decision is per-thread register work.
- **The reduction.** Many bases hit the **same** bin, so the increments collide.
  We `atomicAdd` into two global integer arrays `obs[·]`, `err[·]`. A second kernel
  (one thread per base, no atomics) reads the finished table and writes each base's
  recalibrated quality.

```
   bases (R*L)                         covariate table (small)
  ┌───────────┐   classify_base()     ┌──────────────────────┐
  │ b0 b1 ... │ ───────────────────▶  │ obs[θ]   err[θ]      │  atomicAdd
  │ one GPU   │   skip if known/N      │  (UNSIGNED INTEGERS) │ ◀── collisions
  │ thread    │                        └──────────┬───────────┘
  │ per base  │                                   │ empirical_q()
  └───────────┘                                   ▼
                                        new_q[b] = Q_emp[θ(b)]
```

**No CUDA library is needed at this scale** — the "regression solve" the catalog
mentions (cuBLAS/cuSOLVER) is GATK's hierarchical log-linear model (THEORY §7); our
reduced-scope version uses the closed-form per-bin empirical quality, so the only
primitive is `atomicAdd`. Exercise 5 swaps in the regression view.

## 5. Numerical considerations

- **Determinism via integer atomics.** The tallies are **counts**, so we
  `atomicAdd` into `unsigned int`. Integer addition is associative and commutative,
  so the final table does **not** depend on the (nondeterministic) order in which
  warps retire — the GPU table is bit-identical run-to-run *and* equal to the CPU
  table. A *float* atomicAdd would reorder and lose the last bits → irreproducible
  (the lesson shared with flagships 5.01 and 11.09; Exercise 3 reproduces it).
- **The PHRED math is the only floating point**, and CPU and GPU run the *same*
  `double`-precision `log10`/`pow` from the shared `bqsr.h`, so `Q_emp` (an integer
  after rounding) is identical on both. No race conditions in the recalibrate
  kernel — each thread owns one output element.
- **Overflow.** Counts fit comfortably in `unsigned int` (4.3 billion) for any
  per-bin tally a teaching tile produces; a real WGS run would use 64-bit counts.

## 6. How we verify correctness

`main.cu` builds the table and recalibrates on **both** CPU (`reference_cpu.cpp`)
and GPU (`kernels.cu`), then asserts:

- **Tables identical:** every `obs[θ]` and `err[θ]` match — **tolerance 0**, exact,
  because integer atomics commute (PATTERNS.md §4, the "exact" case).
- **Recalibrated qualities identical:** every base's new `Q` matches — also exact.

Because the CPU reference is an independent, obviously-correct serial loop, exact
agreement is strong evidence the GPU kernels are right. Beyond CPU/GPU parity the
result is **scientifically interpretable**: the synthetic reads were *reported* at
Q30 but injected with a ~1.2% error rate, and the table recovers `Q_emp ≈ 19`
(≈1.3% → the +1 correction and rounding), exactly the miscalibration BQSR exists to
fix. The two known-variant columns are masked, so they do not inflate `err`.

## 7. Where this sits in the real world

Production BQSR (GATK4 `BaseRecalibrator`, **NVIDIA Parabricks**) differs in scope,
not in spirit:

- **More covariates:** read group (per-lane/sample), longer sequence context, and a
  separate **indel** quality model.
- **A hierarchical log-linear regression**, not independent per-bin estimates: a
  global error offset, plus additive per-covariate deltas, fit by maximum
  likelihood — which is the cuBLAS/cuSOLVER "regression solve" the catalog cites.
  This shares strength across sparse bins (a bin seen 5 times borrows from its
  read-group and quality marginals).
- **Scale & I/O:** Parabricks streams a whole BAM through the GPU with pinned-memory
  pipelining and per-block read buffers, matching GATK's output bit-for-bit while
  running ~30× faster. **DeepVariant** sidesteps BQSR entirely by learning context
  inside a CNN.

What this teaching version keeps faithful: the **covariate scan**, **known-variant
masking**, the **+1-corrected empirical quality**, and — the GPU lesson — the
**atomic integer reduction** that makes the parallel table deterministic.

---

## References

- DePristo et al. (2011), *A framework for variation discovery… (GATK)* — introduces
  BQSR and the covariate model.
- McKenna et al. (2010), *The Genome Analysis Toolkit* — the GATK engine.
- Ewing & Green (1998), *Base-calling … quality values (PHRED)* — the quality score.
- NVIDIA Parabricks BQSR docs — the GPU implementation matching GATK output.
- NVIDIA CUDA C++ Programming Guide — atomics and the reduction pattern.
