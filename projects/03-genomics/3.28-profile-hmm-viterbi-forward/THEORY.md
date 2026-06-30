# THEORY — 3.28 Profile HMM (Viterbi / Forward)

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

Proteins in the same **family** share a common ancestry and therefore a common
*shape* of sequence: certain positions are highly conserved (a catalytic residue
that must be there), others tolerate substitutions, and some positions are prone
to insertions or deletions. A **multiple-sequence alignment (MSA)** of a family
lines up these positions into columns. A **profile** captures, per column, *which
amino acids appear and how often* — a far more sensitive search tool than a single
representative sequence, because it knows that (say) column 12 is almost always a
hydrophobic residue while column 13 is anything.

A **profile hidden Markov model (profile HMM)** is the probabilistic version of a
profile. It is the engine behind **Pfam** (the protein-family database) and
**HMMER** (the search tool). The biological question it answers is:

> *Given a profile of a known family and a database of millions of unknown
> protein sequences, which sequences are members of the family — including
> distant homologs a simple BLAST search would miss?*

This matters because **function follows family**: if an uncharacterised protein
from a metagenome scores well against the "ATP-binding" profile, you have a strong
hint about what it does. Metagenomic surveys produce **billions** of protein
fragments, so the search must be fast — hence the GPU.

## 2. The math

### 2.1 The model (a teaching subset of HMMER's "Plan 7")

A profile HMM of length `M` (number of **match columns**) has, per column `k`,
three states:

- **Match `M_k`** — emits one residue, drawn from column `k`'s amino-acid
  distribution. This is where the family signal lives.
- **Insert `I_k`** — emits one residue from the *background* distribution (an
  insertion relative to the family consensus; carries no family signal).
- **Delete `D_k`** — a **silent** state, emits nothing (a deletion: the sequence
  skips column `k`).

Symbols (all probabilities; we store their natural logs):

| Symbol | Meaning | Range |
|---|---|---|
| `e_M(k, a)` | P(emit residue `a` ∣ match column `k`) | `a ∈ {0..19}` (20 amino acids) |
| `e_I(a)` | P(emit residue `a` ∣ any insert state) | background, here uniform |
| `t_MM(k), t_MI(k), t_MD(k)` | transitions out of `M_k` (sum to 1) | — |
| `t_IM(k), t_II(k)` | transitions out of `I_k` (sum to 1) | — |
| `t_DM(k), t_DD(k)` | transitions out of `D_k` (sum to 1) | — |

The input is a database sequence `x = x_1 x_2 … x_L` of residue codes
(`x_i ∈ {0..19}`). The output is a **score**: how well the model explains `x`.

### 2.2 Two scores

For a fixed model and sequence, a **state path** `π` is a sequence of states that
emits exactly `x`. Its probability is the product of all transition and emission
probabilities along it. Two different summaries of "how well the model fits":

- **Viterbi score** = log-probability of the **single most likely path**:
  `S_vit(x) = max_π log P(x, π)`.
  This corresponds to the best **alignment** of `x` to the profile.

- **Forward score** = log-probability **summed over all paths**:
  `S_fwd(x) = log Σ_π P(x, π)`.
  This is the total support — more sensitive for remote homologs, because it adds
  up many mediocre alignments instead of trusting one.

We work in **log space** (natural log, units = *nats*) because `P(x, π)` is a
product of hundreds of small probabilities that would underflow a `double` to 0.

## 3. The algorithm

Both scores are computed by **dynamic programming** over a lattice indexed by
sequence position `i ∈ {0..L}` and profile column `k ∈ {0..M}`, with three planes
`M[i][k]`, `I[i][k]`, `D[i][k]` (the best/total log-prob of being in that state
having emitted `x_1..x_i`). The recurrences (with `⊕` = the **combine** operator):

```
Match :  M[i][k] = e_M(k, x_i) + ( (M[i-1][k-1] + t_MM(k-1))
                                 ⊕ (I[i-1][k-1] + t_IM(k-1))
                                 ⊕ (D[i-1][k-1] + t_DM(k-1)) )

Insert:  I[i][k] = e_I(x_i)    + ( (M[i-1][k] + t_MI(k))
                                 ⊕ (I[i-1][k] + t_II(k)) )

Delete:  D[i][k] =               ( (M[i][k-1] + t_MD(k-1))     <-- same row i
                                 ⊕ (D[i][k-1] + t_DD(k-1)) )   (silent state)
```

The single difference between the two algorithms is the **combine operator** `⊕`:

- **Viterbi:** `a ⊕ b = max(a, b)` (max-sum semiring).
- **Forward:** `a ⊕ b = log(e^a + e^b)` = **log-sum-exp** (log-prob semiring).

That is the central teaching point: *Viterbi and Forward are the same recurrence
in two different semirings.* Our code factors `⊕` into one function
([`phmm.h`](src/phmm.h) `max2`/`max3` vs `log_sum_exp`) selected by a compile-time
flag, so the body is written once.

**Boundary (begin/end).** Before any residue (`i = 0`) a path enters match column
1 for free (`M[0][1] = log 1 = 0`) and may walk a silent **delete** chain along
row 0 to skip leading columns. The score is read from `M[L][M]` (end in the final
match column). This is a simplified begin/end; §7 covers the full Plan-7 version.

**Complexity.**

- *Serial, one sequence:* the lattice has `O(L·M)` cells, each `O(1)` work →
  `O(L·M)` time. The whole database of `N` sequences is `O(N·L·M)`.
- *Memory:* the recurrence only reaches back **one residue** (rows `i` and `i-1`),
  so we keep just two rolling rows of `M+1` cells per plane → `O(M)` memory per
  sequence, not `O(L·M)`. This is the rolling-row trick in both the CPU and GPU
  code.
- *Parallel:* the `N` sequences are independent → **work** `O(N·L·M)`, **depth**
  `O(L·M)` (one sequence's DP is sequential in `i`). Across the database the
  parallelism is `N`-wide and embarrassingly parallel.

Arithmetic intensity is high relative to memory traffic: each thread reads its
sequence's `L` residue bytes once and the (cached, constant-memory) profile, then
does `O(L·M)` floating-point work in registers/local memory — so the kernel is
**compute-bound**, not bandwidth-bound (§4).

## 4. The GPU mapping

**Pattern (PATTERNS.md §1): independent jobs + constant-memory query.** This is
the *same shape* as flagship `1.12` (one query vs N fingerprints) and `12.01` (one
spectrum vs N library spectra) — only the per-item work differs (a DP here, a
popcount/dot-product there).

- **Thread-to-data map:** **one thread ↔ one database sequence**. Thread
  `i = blockIdx.x·blockDim.x + threadIdx.x` runs the *entire* Viterbi (or Forward)
  DP for sequence `i`. A **grid-stride loop** lets a fixed-size grid cover an
  arbitrarily large database.
- **Launch config:** `256` threads/block (a multiple of the 32-lane warp, good
  occupancy on sm_75…sm_89); `blocks = ceil(N / 256)`, capped at 1024 with the
  grid-stride loop covering the rest.
- **Memory hierarchy:**
  - **Constant memory** holds the whole `ProfileHMM` (emission tables + transition
    logs). Every thread reads the same model but never writes it, so the constant
    cache **broadcasts** one address warp-wide in a single transaction — exactly
    the constant-memory query trick from `1.12`.
  - **Global memory** holds the ragged database: one flat `res[]` buffer of all
    residue codes plus `off[]`/`len[]` arrays (CSR-style), uploaded in one copy.
    Each thread streams its own `len[i]` residues once.
  - **Local memory / registers** hold each thread's six rolling DP rows
    (`M/I/D` × prev/cur, each `MAX_M+1` doubles). This is the *occupancy limiter*:
    bigger profiles → more local memory per thread → fewer resident warps. For
    `M ≤ 64` it is comfortable; for production-size profiles you switch to the
    *one-block-per-sequence* scheme (§7, exercise 3) that puts the lattice in
    **shared memory** shared by a cooperating block.
- **No CUDA library here, on purpose.** The DP is a custom recurrence with no
  off-the-shelf primitive (it is not a GEMM/FFT/sort). So we **hand-roll** the
  kernel — and that *is* the lesson. Where a library *would* fit (a warp-level max
  for the cooperative scheme), CUB's `WarpReduce` would do it; we note that in the
  exercises rather than hide the recurrence behind it.

```
  Database (N sequences, ragged)            Profile HMM (M columns)
  res[]: [seq0 | seq1 | seq2 | ... ]        in __constant__ memory,
  off[]:  ^0    ^l0    ^l0+l1               broadcast to every thread
                                            ┌───────────────────────┐
        grid-stride over sequences          │ e_M(k,a), e_I(a),     │
   ┌────────┬────────┬────────┬────────┐    │ t_MM..t_DD per column │
   │thread 0│thread 1│thread 2│  ...   │    └───────────────────────┘
   │ seq0   │ seq1   │ seq2   │        │
   │  DP    │  DP    │  DP    │  each thread runs the full
   │ M/I/D  │ M/I/D  │ M/I/D  │  L×M recurrence in local memory
   └───┬────┴───┬────┴───┬────┘
       v        v        v
     out[0]   out[1]   out[2]        (one score per sequence)
```

## 5. Numerical considerations

- **Precision = FP64 (double).** The DP accumulates `O(L·M)` log-space additions;
  the magnitudes (tens of nats) and the `log1p`/`exp` in log-sum-exp are
  comfortably accurate in double. We *store* the final score as `float` only at the
  very end (it is compared as `float`), which is where the only rounding happens.
- **`-inf` handling.** Impossible states are a finite sentinel `LOG_ZERO = -1e30`
  rather than true `-inf`, so `(-inf) + finite` and `max(-inf, x)` stay clean and
  identical on host and device, and we never form `(-inf) - (-inf) = NaN`. The
  `log_sum_exp` helper short-circuits when either argument is the sentinel.
- **Stable log-sum-exp.** `log(e^a + e^b) = max(a,b) + log1p(e^(-|a-b|))` factors
  out the larger term so `exp()` never sees a positive argument (no overflow), and
  `log1p` is accurate when the correction is tiny.
- **Determinism — and why there are no atomics.** Each thread writes exactly one
  independent output `out[i]`; there is **no cross-thread reduction**, so there are
  no `atomicAdd`s and no float-reordering nondeterminism (contrast `5.01`/`11.09`,
  which *must* use integer/fixed-point atomics for determinism). The per-thread DP
  performs its additions in a fixed order on both CPU and GPU. As a result stdout
  is byte-identical every run, and **Debug and Release produce identical output**
  (verified). The only ordering subtlety is *ties* in the ranking — two decoys can
  share a Viterbi score — which we break by lower index so the printed order is
  deterministic.

## 6. How we verify correctness

- **Independent CPU reference.** [`reference_cpu.cpp`](src/reference_cpu.cpp)
  implements the same DP with a plain serial loop and `std::vector` rolling rows.
  It is written to be *obviously* correct, with no parallelism.
- **Shared `__host__ __device__` core (the key to exactness).** Both the CPU
  reference and the GPU kernel call the *same* `phmm.h` primitives (`emit_match`,
  `emit_insert`, `max2/max3`, `log_sum_exp`) in the *same* order. So they execute
  byte-for-byte identical floating-point operations — and indeed the measured
  `max_abs_err` is **`0.0e+00`** for both Viterbi and Forward. We still set a
  nonzero **tolerance of `1.0e-4` nats** to allow for the final double→float store
  and any platform FMA differences; agreement is far inside it (PATTERNS.md §4,
  "exact / same exact operations" bucket).
- **A scientific check, not just CPU==GPU.** Agreement only proves the two
  implementations match. The *model* is validated by the **ranking**: the planted
  homolog (a 3-substitution mutant of the consensus) scores `-31.85` nats, while
  every random decoy scores below `-85` — a margin of ~54 nats. A correct profile
  HMM *must* separate true family members from random sequences, and it does.
- **Edge cases exercised by the sample:** sequences with substitutions (the
  homolog), sequences with no signal (decoys), ties in the score (two decoys),
  and the begin/end boundary (every sequence is the full profile length).

## 7. Where this sits in the real world

Production profile-HMM search differs from this teaching version in scope, not in
the core recurrence:

- **Full Plan-7 architecture.** HMMER's model adds flanking states **N, B, E, C, J**
  that implement **local vs glocal** alignment (a hit can start/end mid-profile)
  and **multi-hit** scoring (a sequence can contain several copies of the domain).
  Our begin = "enter at column 1", end = "exit from column M" is the simplest case.
- **The acceleration cascade.** HMMER3 does not run full Forward on everything. It
  runs cheap filters first — **SSV/MSV** (ungapped/multi-segment Viterbi),
  then **P7Viterbi**, then **Forward-Backward** — passing only survivors to the
  expensive stage. MSV/SSV alone is ~72% of runtime, which is exactly what CUDAMPF
  targets on the GPU.
- **SIMD / cooperative parallelism.** HMMER vectorises the DP across profile
  columns with SSE/AVX in **reduced precision** (`int8`/`int16` for MSV). **CUDAMPF**
  maps **one block per sequence**, each thread owning a strip of columns and
  cooperating along anti-diagonals with **warp-level reductions** and **`float4`**
  vectorised cells — higher throughput than our one-thread-per-sequence scheme for
  long sequences, at the cost of more complex code (exercise 3).
- **Statistics.** Real hits are reported with **bit scores** (vs a null model) and
  **E-values** (Gumbel/exponential fits to the score distribution), not raw nats
  (exercise 2). And emissions come from a real MSA with **Dirichlet-mixture priors**,
  not a single consensus residue.
- **Alternatives.** **MMseqs2** prefilters with a fast **k-mer** match before
  profile scoring, trading a little sensitivity for a large speed-up — a different
  point on the speed/accuracy curve.

This project keeps the **recurrence and the GPU data-parallel decomposition**
faithful while simplifying the architecture, parameters, statistics, and
cooperative scheme — the right trade for *learning how Viterbi/Forward run on a
GPU*.

---

## References

- S. R. Eddy, *Accelerated Profile HMM Searches*, PLoS Comput. Biol. (2011) — the
  HMMER3 algorithm and the MSV/Viterbi/Forward cascade.
- HMMER user guide & source (<https://github.com/EddyLab/hmmer>) — Plan-7
  architecture, the flanking states, and the vectorised DP.
- H. Jiang & N. Ganesan, *CUDAMPF*, BMC Bioinformatics (2016)
  (<https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-016-0946-4>)
  — one-block-per-sequence GPU MSV/Viterbi; study its memory layout and reductions.
- L. Rabiner, *A Tutorial on Hidden Markov Models* (1989) — the canonical
  derivation of Viterbi and Forward (the speech-recognition origin of the algorithms).
- M. Steinegger & J. Söding, *MMseqs2* (<https://github.com/soedinglab/MMseqs2>) —
  k-mer prefiltering as a faster route to profile search.
