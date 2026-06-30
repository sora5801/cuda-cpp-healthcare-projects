# THEORY — 3.22 RNA-seq Quantification / Pseudo-alignment

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

A cell's identity and state are largely set by **which genes it transcribes and
how much**. RNA-seq measures this: it shatters the cell's RNA into millions of
short fragments, sequences ~100-base **reads** from them, and from the read counts
we estimate each transcript's **abundance** (expression level). That number drives
differential-expression studies, cancer subtyping, single-cell atlases, and more.

The complication that makes quantification interesting is **isoforms**. One gene
produces several transcript variants (isoforms) by alternative splicing, and they
**share exons** — long stretches of identical sequence. A read from a shared exon
is genuinely compatible with *several* transcripts; you cannot tell which one it
came from. So "count reads per transcript" is ill-posed: many reads are ambiguous.

```
gene A                       reads
 ┌── exon1 ──┬── exon2 ──┐    r1 (exon1 only) -> only isoform t0   (unique)
 t0:  [=========exon2====]    r2 (exon2)      -> t0 AND t1         (ambiguous!)
 t1:  [exon1==============]    r3 (exon1)      -> only isoform t1   (unique)
        shared region ^^^
```

**Pseudo-alignment** (kallisto, 2016; Salmon, 2017) is the fast modern answer.
Rather than align each read to the genome base-by-base, it:

1. breaks each read into **k-mers** (length-k substrings),
2. looks each k-mer up in a hash/de-Bruijn index of the transcriptome to find the
   set of transcripts that k-mer occurs in,
3. intersects those sets over the read's k-mers to get the read's **compatibility
   set** — the transcripts the *read* could have come from,
4. collapses all reads with the **same compatibility set** into one **equivalence
   class (ec)**, keeping only a per-ec read count.

That is 10–100× faster than alignment and loses almost no quantification accuracy.
What remains is a statistical problem: **given the ec counts, estimate the
transcript abundances** — the job of this project's EM.

## 2. The math

**Notation.**

- `T` transcripts, indexed `t = 0..T-1`. `M` equivalence classes, indexed `e`.
- `ℓ_t` = **effective length** of transcript `t` (positions a read can start =
  length − mean fragment length + 1). Units: bases.
- `n_e` = number of reads in ec `e` (observed). `N = Σ_e n_e` = total reads.
- `E_t` = the set of ecs whose compatibility set contains `t`; `M_e` = the members
  of ec `e`.
- **Unknowns** `ρ_t ∈ [0,1]`, `Σ_t ρ_t = 1` — the **abundance**: the probability a
  random read originates from transcript `t`.

**Generative model.** A read is produced by (i) picking a transcript `t` with
probability proportional to its *available sequence*, `ρ_t · ℓ_t`… no — we model
abundance as *molecular* abundance, so a read is produced by picking transcript
`t` with probability `ρ_t`, then a uniformly random start position among its `ℓ_t`
positions. The read's compatibility set (hence its ec) is then determined by where
it landed. The probability that a read from transcript `t` is *seen as* ec `e`
(i.e. `t ∈ M_e`) is governed by which region it hit; aggregating, the chance a read
belongs to ec `e` AND truly came from member `t` is proportional to the
**length-normalised weight**

```
    w_t = ρ_t / ℓ_t .
```

(The `1/ℓ_t` is the key correction: at equal molecular abundance a longer
transcript yields *more* reads, so to recover molecular `ρ` we divide length out.)

**Likelihood.** Treating each read's true origin as a hidden variable, the
log-likelihood of the observed ec counts under abundances `ρ` is

```
    L(ρ) = Σ_e  n_e · log( Σ_{t ∈ M_e}  w_t )          (up to constants),
```

a mixture model. We want the **maximum-likelihood** `ρ̂ = argmax_ρ L(ρ)` subject
to `Σ ρ_t = 1`. There is no closed form (the log of a sum), so we use EM.

## 3. The algorithm

**Expectation-Maximisation** alternates a closed-form E-step and M-step that are
each easy, and is guaranteed to never decrease `L`.

Initialise `ρ_t = 1/T` (uniform). Then repeat:

- **E-step (responsibilities).** For each ec `e`, split its `n_e` reads among its
  members in proportion to their current weights. The expected reads transcript
  `t ∈ M_e` receives from ec `e` is

  ```
      c_{e,t} = n_e ·  w_t / Σ_{s ∈ M_e} w_s ,        w_t = ρ_t / ℓ_t .
  ```

- **M-step (re-estimate).** Sum each transcript's expected reads over all its ecs,
  then renormalise to a distribution:

  ```
      α_t = Σ_{e ∈ E_t} c_{e,t} ,        ρ_t ← α_t / Σ_s α_s .
  ```

Singleton ecs (unique regions) have one member, so all their reads go straight to
that transcript — these are the **anchors** that let EM disambiguate the shared
ecs. Iterate to convergence (the `α` stop changing). Output `ρ`, the per-transcript
read counts `α`, and usually **TPM** = transcripts per million:

```
    TPM_t = 1e6 · (ρ_t / ℓ_t) / Σ_s (ρ_s / ℓ_s) .
```

**Complexity.** Let `Z = Σ_e |M_e|` be the total membership (the number of nonzeros
in the ec×transcript matrix). One EM iteration is `O(Z)` work — each member is
touched once in the E-step and once in the M-step. With `I` iterations the serial
cost is `O(I·Z)`. Crucially `Z ≪ M·T` (each ec has only a few members), so this is
a **sparse** computation; treating it as a dense `M×T` GEMV would be `O(M·T)` and
hopelessly wasteful. Memory access is **irregular** (gather member `ρ`, scatter to
member `α`), which is what makes the GPU mapping interesting rather than trivial.

## 4. The GPU mapping

One EM iteration is exactly the pattern from flagship `11.09` (k-means): a
**parallel per-item step** followed by an **atomic reduction**.

**Thread-to-data mapping.** One thread owns **one equivalence class** `e`:

```
    e = blockIdx.x * blockDim.x + threadIdx.x        // guard e >= M
```

Thread `e` reads its member slice from the CSR arrays, runs the E-step into a tiny
per-thread register array `contrib[k]` (length `k = |M_e| ≤ PSA_MAX_EC_SIZE`), then
`atomicAdd`s each member's expected reads into the global per-transcript
accumulator. Because many ecs share a popular transcript, those adds **collide** —
hence atomics.

```
   ecs (one thread each)            per-transcript accumulator (global)
   ┌──────────────┐                      α[0] α[1] α[2] ... α[T-1]
   │ thread e=0 {0}      ──n_0──────────▶ +
   │ thread e=1 {1}      ──n_1──────────────▶ +
   │ thread e=6 {0,1}    ──split──▶ + ───────▶ +     (atomicAdd collisions)
   │ thread e=8 {2,3,4}  ──split──▶ + ──▶ + ──▶ +
   └──────────────┘
   E-step: independent per thread        M-step: atomic scatter-reduction
```

**Launch configuration.** `block = 128`, `grid = ⌈M/128⌉`. The kernel is **latency-
bound** on the irregular gather of member `ρ` values (a handful of scattered
`double` loads per thread), not compute-bound, so 64/128/256 threads/block all
perform about the same — an easy thing to sweep (Exercise 5). No shared memory is
used: the per-ec work is tiny and independent, so there is nothing to stage.

**Memory hierarchy.** Inputs (`ρ`, `ℓ`, ec counts, CSR offsets/members) live in
**global** memory, marked `__restrict__` so the compiler may route the read-only
loads through the read-only cache. The `contrib[]` scratch lives in
**registers/local** memory (size known at compile time via `PSA_MAX_EC_SIZE`). The
only writes are `atomicAdd` into the **global** accumulator.

**Determinism via fixed point.** The catalog says "warp-level reduction for
abundance accumulation". A naive `float`/`double` `atomicAdd` reduction is **not
deterministic**: floating-point addition is non-associative, so the result depends
on the (race-determined) order threads arrive — and it would *not* match the CPU.
We instead quantise each contribution to a **fixed-point integer** (×2²⁴) and
`atomicAdd` on `unsigned long long`. Integer addition commutes, so the reduction is
order-independent → reproducible AND bit-identical to the CPU (PATTERNS.md §3, same
trick as `5.01` and `11.09`).

**Where a library would go.** Each EM iteration is a **sparse matrix–vector
product**: let `A` be the `M×T` membership matrix and `w` the weight vector; the
E-step numerators are row-wise `A·w`, and the M-step is `Aᵀ·(scaled counts)`. This
is precisely a **cuSPARSE SpMV** (`cusparseSpMV` on a CSR matrix). We hand-roll it
so the gather/scatter and the determinism trick are visible (no black box, CLAUDE.md
§6.1.6); writing it by hand also lets us fuse the row-normalisation into the same
pass, which a generic SpMV cannot. THEORY §7 says what production does instead.

## 5. Numerical considerations

- **Precision.** We use **FP64** everywhere (`ρ`, weights, the E-step). Abundances
  span several orders of magnitude and the E-step divides by a sum of small
  weights; double precision keeps the ratios accurate and makes exact CPU/GPU
  parity easy. The memory cost is irrelevant at this scale.
- **Fixed-point M-step.** Contributions are stored as `round(value · 2²⁴)` in
  `unsigned long long`. 2²⁴ ≈ 1.7e7 gives ~7 significant digits per contribution;
  the largest accumulated value is `N · 2²⁴`, which for `N` up to ~10¹¹ reads stays
  under the 1.8e19 `ull` ceiling. Rounding is to-nearest (`+0.5` before truncation)
  identically on host and device.
- **Race conditions.** The *only* shared write is the `atomicAdd` reduction, and it
  is safe and deterministic because it is integer (commutes). The E-step has no
  cross-thread sharing.
- **Stability / convergence.** EM monotonically increases the likelihood and
  cannot diverge; we run a **fixed** iteration count (100) rather than a
  data-dependent stop so the CPU and GPU execute identically and stdout is
  reproducible (an early stop on a floating threshold could trip at a different
  iteration on each side — see Exercise 3).
- **Degenerate ecs.** If every member of an ec has zero weight (cannot happen after
  uniform init, but we guard anyway), the reads are spread uniformly so no count is
  lost and no division by zero occurs.

## 6. How we verify correctness

Two independent checks, both in `src/main.cu`:

1. **GPU vs CPU.** `src/reference_cpu.cpp` is a plain serial EM that calls the
   *same* `__host__ __device__` E-step (`pseudoalign.h::psa_ec_contributions`) and
   the *same* host renormalise (`counts_to_rho`) as the GPU path. Because every
   arithmetic operation is identical and the only reduction uses commuting integer
   atomics, the two final `ρ` vectors are **bit-identical**. We assert
   `max_t |ρ_t^CPU − ρ_t^GPU| ≤ 1e-12`; the measured value is `0`. Tolerance choice:
   PATTERNS.md §4 "exact" tier — same exact operations on both sides.
2. **Recovery of a known truth.** The synthetic sample embeds the abundances used
   to generate it, and the ec counts are produced by the very model EM inverts, so
   EM recovers them to `L1 ≈ 2e-5` (residual is integer rounding of the counts). We
   print `recovery: L1(estimated ρ, truth ρ)`, which rounds to `0.0000`. This
   validates the **science** (we get the right answer), not just CPU=GPU agreement.

Why this is convincing: an independent serial implementation agreeing to machine
precision rules out kernel bugs (indexing, races, missed members), and recovering a
known ground truth rules out a *correct implementation of the wrong algorithm*.

## 7. Where this sits in the real world

Production pseudo-aligners do far more than this teaching EM:

- **They build the ecs.** kallisto constructs a **colored de Bruijn graph** /
  hashed k-mer index of the transcriptome and pseudo-aligns reads against it to
  *produce* the equivalence classes. That index + lookup (the catalog's "GPU hash
  table for k-mer to equivalence class look-up") is the heavy data-structure work
  we skip — we start from ecs. The 2026 "RNA-seq analysis in seconds using GPUs"
  work (Melsted et al.) is exactly about moving that lookup *and* the EM onto the
  GPU.
- **Salmon** adds a **variational-Bayes** EM and rich bias models (sequence-specific
  and GC bias, positional bias, fragment-length distribution) that materially
  improve accuracy on real data; it also offers online/streaming inference.
- **Bootstrap / Gibbs** resampling gives per-transcript **uncertainty** (kallisto's
  bootstraps, used downstream by sleuth) — an ensemble pattern (cf. flagship `9.02`).
- **Scale & layout.** Real runs have 10⁵–10⁷ ecs; kallisto stores them compactly and
  the GPU version expresses the EM update as a fused sparse operation (cuSPARSE-style
  SpMV) with CUDA **streams** overlapping read I/O with compute. Single-cell tools
  (**bustools**, **alevin-fry**) run a closely related per-cell EM over a BUS file.

Our version keeps the EM mathematics exactly right while deliberately omitting the
index construction, bias modelling, and uncertainty — the right scope for learning
*why the EM works and how it parallelises*.

---

## References

- **N. Bray, H. Pimentel, P. Melsted, L. Pachter (2016), "Near-optimal probabilistic
  RNA-seq quantification," *Nat. Biotechnol.*** — the kallisto paper; defines
  pseudo-alignment, equivalence classes, and the EM used here.
- **R. Patro et al. (2017), "Salmon provides fast and bias-aware quantification of
  transcript expression," *Nat. Methods*** — quasi-mapping + bias-aware VB-EM; the
  accuracy refinements we omit.
- **B. Li & C. Dewey (2011), "RSEM…," *BMC Bioinformatics*** — the classic
  alignment-based EM; the same mixture-model likelihood, derived in full.
- **kallisto GPU branch** <https://github.com/pachterlab/kallisto> — how the lookup
  and EM are redesigned for the GPU; the target of "RNA-seq analysis in seconds".
- **Salmon** <https://github.com/COMBINE-lab/salmon> — read the VB-EM and bias
  modules to see what a production quantifier adds.
- **bustools** <https://github.com/BUStools/bustools> and **alevin-fry**
  <https://github.com/COMBINE-lab/alevin-fry> — the single-cell EM over BUS records.
- **cuSPARSE `cusparseSpMV`** (NVIDIA docs) — the library SpMV the hand-rolled EM
  update corresponds to (§4 "Where a library would go").
