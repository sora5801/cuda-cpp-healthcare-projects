# THEORY — 2.15 Antibody Structure Prediction (reduced-scope: CDR screening)

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

> **Scope contract (CLAUDE.md §13).** This project does **not** predict 3-D
> antibody structures. It implements the **library-screening** step that sits
> under high-throughput antibody work: rank a library of antibodies by how
> similar their CDR loops are to a query's, weighting CDR-H3 most. The final
> section explains what the full IgFold/ABodyBuilder3 pipeline does and exactly
> where this piece fits.

---

## The science

An **antibody** (immunoglobulin) is a Y-shaped protein the immune system uses to
recognize foreign molecules (**antigens**). The business end is the **Fv region**:
the variable domains of the **heavy** and **light** chains (VH + VL). Most of each
variable domain is a conserved **β-sandwich framework**; grafted onto it are three
hypervariable loops — the **complementarity-determining regions (CDRs)**:

```
          VH (heavy)                         VL (light)
   ┌───────────────────────┐         ┌───────────────────────┐
   │ framework  CDR-H1      │         │ framework  CDR-L1      │
   │ framework  CDR-H2      │         │ framework  CDR-L2      │
   │ framework  CDR-H3 ◄────┼─ longest│ framework  CDR-L3      │
   └───────────────────────┘  most    └───────────────────────┘
                              variable
        the six CDR loops together form the PARATOPE (antigen-binding surface)
```

The six CDRs fold together into the **paratope**, the surface that grips the
antigen. **CDR-H3** is special: it sits at the center of the paratope, varies most
in length and sequence (it is assembled by V(D)J recombination with junctional
diversity), and contributes the most binding energy and specificity. That single
biological fact — *CDR-H3 dominates* — is what we encode as a ×3 weight.

To compare CDRs across antibodies you must first **delimit** them consistently.
The **IMGT numbering scheme** assigns every residue in a variable domain a
canonical position, so "CDR-H3" means the same span in every antibody. Tools like
**ANARCI** do this numbering. In this project the dataset is **pre-delimited**:
each antibody is given as its six CDR strings directly (see `data/README.md`).

**Why this matters in practice.** Antibody engineering — humanization, affinity
maturation, developability triage — routinely screens large libraries: "which of
these 10⁶ candidate antibodies have CDRs like my lead?" That screen is a
similarity ranking, and it is exactly what we build here.

## The math

Encode each residue as an integer `0..19` (the 20 amino acids) plus `20` for a
gap/pad. A **substitution matrix** `S` (we use **BLOSUM62**) gives `S[i][j]`, the
log-odds score of aligning residue `i` with residue `j`: identical/conservative
substitutions score positive, disruptive ones negative. BLOSUM62 is symmetric and
integer-valued.

**Per-CDR score.** Two CDR fields `a, b`, each padded to `L = 24` residues, score
by an **ungapped column sum**:

```
    cdr_score(a, b) = Σ_{p=0}^{L-1} S[a_p][b_p]
```

We pad with gaps and set `S[gap][gap] = +1`, so equal-length pads add a small
uniform constant and a residue-vs-gap is penalized (`S[·][gap] = −4`).

**Antibody score.** Let the six CDRs be indexed `c = 0..5` (H1,H2,H3,L1,L2,L3) and
let `w_c` be the CDR weight (`w_2 = 3` for CDR-H3, else `1`). The query-vs-library
score is

```
    score(query, lib) = Σ_{c=0}^{5} w_c · cdr_score(query_c, lib_c)
```

Every term is a sum of small integers, so `score` is an **exact integer** — there
is no floating-point rounding anywhere. That is what makes CPU and GPU agree
bit-for-bit (see "How we verify correctness").

**The screen.** Compute `score(query, lib_i)` for all `N` library antibodies `i`,
then report the `K` largest (ties broken by lower index for determinism).

## The algorithm

```
load:   parse query + N library antibodies; encode + pad to 144 residues each
for each library antibody i in 0..N-1:        # independent across i
    s_i = 0
    for each CDR c in 0..5:
        cdr = Σ_p S[query[c][p]][lib_i[c][p]]  # 24-term column sum
        s_i += weight(c) * cdr                 # CDR-H3 counts triple
report: top-K of {s_i}
```

**Complexity.** Each antibody is `O(6 · 24) = O(144)` integer ops, so the whole
screen is `O(N · 144) = O(N)` work — linear in the library size. The serial CPU
reference runs this loop directly; the GPU runs the `i`-loop in parallel.

**Why it parallelizes perfectly.** The outer loop over `i` has **no
cross-iteration dependencies**: `s_i` depends only on the query (shared, read-only)
and on library row `i`. This is the textbook *embarrassingly parallel* shape —
"score one query against N independent items" (PATTERNS.md §1, exemplar 1.12).

## The GPU mapping

We use the **"independent jobs + constant-memory query"** pattern:

- **One thread per library antibody.** Thread `(blockIdx.x, threadIdx.x)` owns
  index `i = blockIdx.x·blockDim.x + threadIdx.x`, then strides by the total
  thread count (a **grid-stride loop**) so a fixed grid covers any `N`.
- **Query in `__constant__` memory.** The 144-byte query record is read by *every*
  thread, identically, and never written. Constant memory's broadcast cache serves
  one address to a whole 32-lane warp in a single transaction — ideal here.
  Allocated as `__constant__ uint8_t c_query[144]`, filled with
  `cudaMemcpyToSymbol`.
- **Library in global memory, row-major.** Antibody `i` occupies
  `lib[i·144 .. i·144+143]`. Consecutive threads read consecutive rows.
- **No shared memory, no atomics.** Outputs are fully independent — one `int32`
  score per antibody — so there is nothing to reduce across threads. (Contrast
  project 11.09, where centroid accumulation *does* need atomics.)
- **The scoring math is shared.** The per-pair score lives in
  `src/antibody.h` as `__host__ __device__` functions (the HD-core idiom,
  PATTERNS.md §2). The CPU reference and the kernel call the **same**
  `ab_cdr_score()` — one definition of "the score," so the two paths cannot drift.

```
  grid:   ceil(N/128) blocks (capped at 1024; grid-stride covers the rest)
  block:  128 threads (multiple of the warp; good occupancy on sm_75..sm_89)
  thread: i -> s[i] = Σ_c w_c · Σ_p S[c_query[c][p]][lib[i][c][p]]
              └ c_query from constant cache; lib row i from global memory
```

**Occupancy / bandwidth note.** Each thread does only ~144 integer adds and reads
144 bytes, so the kernel is **memory/launch-bound**, not compute-bound. On the
tiny demo (`N = 24`) the launch + copy overhead dwarfs the compute, which is why
the timing is a teaching artifact. The GPU's advantage appears at library scale
(millions of antibodies streamed through), where the independent per-row work
saturates memory bandwidth.

## Numerical considerations

- **Integer everywhere.** BLOSUM62 entries and the weights are small integers; a
  full antibody score fits comfortably in `int32` (max ≈ `6·24·11·3 ≈ 4752`). No
  floating point, no rounding, no `--use_fast_math` subtlety. This is the cleanest
  numerics in the repo.
- **Determinism.** The kernel writes each `s[i]` from exactly one thread (no
  atomics, no reduction reordering), so the device result is reproducible
  regardless of warp scheduling. The host top-K breaks ties by lower index, so the
  printed ranking is byte-identical every run (the basis of `expected_output.txt`).
- **Encoding edge cases.** Unknown letters, lowercase, and `-` all map to the gap
  symbol, so malformed input degrades gracefully instead of reading out of bounds.
  CDR tokens longer than 24 residues are truncated and the count is reported to
  stderr.

## How we verify correctness

Two checks, in increasing strength:

1. **CPU == GPU, exactly.** `main.cu` runs `score_cpu()` and `score_gpu()` and
   computes `max_abs_err = max_i |s_cpu[i] − s_gpu[i]|`. Because both call the same
   integer `ab_cdr_score()`, the tolerance is **0** — any nonzero difference is a
   bug, not a rounding artifact (PATTERNS.md §4, "exact" case).
2. **Recovering a planted answer.** The synthetic dataset embeds two known "hits"
   (PATTERNS.md §6): `mAb_07` is a near-copy of the query (2 point mutations per
   CDR) and `mAb_18` shares the query's *exact* CDR-H3. A correct screen must rank
   these two at the top, far above the random antibodies — and it does (516 and
   373 vs ≤25). The per-hit breakdown also confirms CDR-H3 supplies most of
   `mAb_18`'s score (354 of 373), validating the weighting, not just the agreement.

## Where this sits in the real world

This project is the **similarity-ranking** rung of the ladder; the full tools
climb much higher:

- **IgFold / ABodyBuilder3** predict *3-D coordinates*. They embed the antibody
  sequence with a protein **language model** (ESM-2 / IgLM) — a multi-head
  attention transformer run on the GPU with **cuDNN / Flash-attention** in FP16 —
  then a structure module (Evoformer/IPA, vectorized from **OpenFold**) folds the
  Fv, with extra care for the flexible **CDR-H3** loop (loop sampling / diffusion)
  and **disulfide** geometry. That is hundreds of MB of trained weights and a
  multi-kernel pipeline — deliberately out of scope for a single didactic kernel.
- **Our scoring is sequence-only and ungapped.** Production similarity uses gapped
  alignment (Needleman-Wunsch per CDR — see project 3.01 for the GPU wavefront) and
  often compares *structures* (loop RMSD) rather than sequences.
- **Our CDR weights are illustrative.** Real pipelines learn position-specific
  importance from data rather than hard-coding CDR-H3 ×3.
- **What transfers directly:** the GPU *pattern*. "Broadcast a small query from
  constant memory; give each library item its own thread; keep the per-item math in
  a shared host/device core for exact verification" is exactly how a library-scale
  antibody screen (or a Tanimoto search, 1.12, or a spectral search, 12.01) is
  structured. Master it here on integer arithmetic, then scale the *math* up.

**Exercises** to deepen this are listed in the [README](README.md#exercises):
gapped CDR alignment, tunable weights, on-GPU top-K, a warp-per-antibody reduction,
and screening real ANARCI-numbered CDRs.
