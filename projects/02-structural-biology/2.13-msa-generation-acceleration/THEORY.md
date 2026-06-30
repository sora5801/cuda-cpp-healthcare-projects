# THEORY — 2.13 MSA Generation Acceleration

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See
> [README.md](README.md) for the quick tour and build steps.
>
> _Educational only — not for clinical use. All data here is synthetic._

---

## 1. The science

To predict a protein's 3-D structure, **AlphaFold2** does not look at the query
sequence alone — it first finds **homologs** (evolutionarily related proteins)
and stacks them into a **multiple sequence alignment (MSA)**. The pattern of which
columns co-vary across that alignment is the single most informative signal AF2
uses: residues that mutate *together* are usually in contact in the folded
structure. **Better MSA → better structure prediction.**

Building that MSA is a *search* problem. Given a query protein, you scan a giant
database of known sequences (UniRef90 is ~210 GB) and pull out the ones similar
enough to be relatives. The gold-standard tools for this — **HMMER/Jackhmmer** and
**HHblits** — do not do a naive pairwise comparison. They turn the query (and its
early hits) into a **profile hidden Markov model (HMM)**: a position-by-position
statistical model of the protein family that knows, for each column, which
residues are conserved and how tolerant each position is to insertions and
deletions. They then **score every database sequence against that profile HMM**
with a dynamic program (Viterbi).

This database scan is **the bottleneck**. In the AlphaFold2 pipeline it is one of
the last steps that runs on the **CPU**, taking minutes to hours per query, while
the neural network runs on the GPU in seconds. The research direction (GPU-HMMER,
MMseqs2-GPU, ColabFold's MSA server) is to move the Viterbi scan onto the GPU.
**That inner loop is what this project teaches.**

> **Reduced-scope teaching version.** A production tool builds the profile HMM
> from data, uses per-column transition probabilities, local/glocal modes, null
> models, E-values, k-mer prefiltering, and bit-score calibration. We ship a
> *clearly simplified* profile HMM and a direct Viterbi scan so the **GPU
> parallelisation** is unmistakable. §7 describes what the real tools add.

## 2. The math

A **profile HMM** of length `L` is a left-to-right chain of `L` **match** states
(`M₁…M_L`), each with a companion **insert** state (`Iₖ`) and **delete** state
(`Dₖ`). It generates a protein sequence by walking from the begin state to the end
state, emitting one residue per match/insert state visited.

**Emissions.** Match state `k` emits amino acid `a` with probability `eₖ(a)`. We
score against a background distribution `p(a)` using a **log-odds**:

```
  emit_score(k, a) = log( eₖ(a) / p(a) )
```

Positive means "residue `a` is more common in family column `k` than by chance"
(a good match); negative means rarer. Insert states emit at the background rate,
so in log-odds terms their emission is `0` and drops out.

**Transitions.** Moving between states costs a log-odds transition score. We use
seven (shared across columns for teaching simplicity):

```
  t_mm  M→M   t_mi  M→I   t_im  I→M   t_ii  I→I
  t_md  M→D   t_dm  D→M   t_dd  D→D
```

**The Viterbi recurrence.** To score a database sequence `x = x₁…x_T` against the
profile, define `M[i][k]`, `I[i][k]`, `D[i][k]` = the best (max) log-odds of any
state path that has consumed the first `i` residues and ends in `Mₖ / Iₖ / Dₖ`:

```
  M[i][k] = emit_score(k, xᵢ) + max( M[i-1][k-1] + t_mm,
                                     I[i-1][k-1] + t_im,
                                     D[i-1][k-1] + t_dm )

  I[i][k] = max( M[i-1][k] + t_mi,        // (insert emission = 0 in log-odds)
                 I[i-1][k] + t_ii )

  D[i][k] = max( M[i][k-1] + t_md,        // deletes consume NO residue:
                 D[i][k-1] + t_dd )        //   they move within row i
```

The **hit score** of the sequence is the best match score anywhere in the grid,
`max over i,k of M[i][k]` (a local-style score — the alignment may begin anywhere,
because the begin state is reachable for free at every row, `M[i][0] = 0`).

**Symbols:** `L` = profile length (columns); `T` = database-sequence length; `xᵢ ∈
{0..20}` = amino-acid index (20 = unknown). All scores are **dimensionless
log-odds**, pre-multiplied by `SCORE_SCALE = 1000` and rounded to **integers**
(see §5). `NEG_INF ≈ −10⁹` represents an unreachable state (`log 0`).

## 3. The algorithm

For each of the `N` database sequences we fill its `(T+1)×(L+1)` Viterbi grid and
take the max. The grid has a **row dependency** (row `i` reads row `i-1`) and,
within a row, the delete chain `D[k]` reads `D[k-1]` — so:

- **Across sequences:** completely independent → `N`-way parallel.
- **Within one sequence, across rows:** strictly sequential (row `i` needs `i-1`).
- **Within one row, across columns:** `M[k]` and `I[k]` depend only on the
  *previous* row → all `L` columns are independent and can be done in parallel.
  `D[k]` depends on the *current* row's `D[k-1]` → a left-to-right scan.

**Memory.** We never store the full grid: a row only needs the previous row, so we
keep two rows of `M,I,D` (length `L+1`) and ping-pong → **O(L) memory** per
sequence instead of O(T·L). This is the key that lets a whole sequence's working
set fit in fast on-chip memory.

**Complexity.** One sequence costs `O(T·L)` integer max/add operations; the whole
scan is `O((Σ Tᵢ)·L)`. Serial **work** and parallel **work** are the same; the
parallel **depth** for one sequence is `O(T)` (the unavoidable row chain) with each
row's M/I done in `O(1)` span across `L` threads.

## 4. The GPU mapping

The decomposition above maps directly onto CUDA:

```
   grid   = N blocks            ← one block per database sequence (independent)
   block  = 128 threads         ← cooperate on ONE sequence's Viterbi DP
   thread → a strided set of profile columns k

   ┌──────────────────────────── grid (N blocks) ───────────────────────────┐
   │ block 0      block 1      block 2     ...     block N-1                  │
   │  seq[0]       seq[1]       seq[2]              seq[N-1]                  │
   └──────────────────────────────────────────────────────────────────────-─┘
         │ each block, per database residue r = 0..T-1:
         │   Phase 1 (parallel over k): M[k], I[k]  ← read previous row
         │   __syncthreads()
         │   Phase 2 (one thread):      D[k] chain  ← left-to-right scan
         │   __syncthreads()
         │   Phase 3 (parallel over k): fold best M[k] into a running max
         │   cooperative copy cur → prev  (ping-pong),  __syncthreads()
         └─ tree-reduce the per-thread maxima → this sequence's hit score
```

**Why one block per sequence (not one thread per sequence)?** A thread-per-sequence
mapping (like 1.12's Tanimoto) is simplest, but each Viterbi DP has internal
parallelism (`L` columns) and a working set (`6·(L+1)` ints) better served by a
block's shared memory than by one thread's registers/local memory. Blocks also let
us scale `L` without spilling.

**Memory hierarchy and why:**

- **Constant memory** holds the emission table `c_emit[L*21]`. Every thread of
  every block reads it and nothing writes it during the launch, so the constant
  cache *broadcasts* one address to a whole warp in a single transaction — the same
  trick flagship **1.12** uses for its query fingerprint.
- **Shared memory** holds the two ping-pong rows (`prev/cur` × `M/I/D`, each
  `L+1`) plus a small reduction scratch. The per-residue sweep then touches only
  fast on-chip memory; global memory is read only to stream in residues.
- **Global memory** holds the database in **CSR layout**: all residues
  concatenated into one buffer `res`, plus an `offset[N+1]` index, so a block finds
  "its" sequence with a single offset lookup — exactly the ragged-array layout a
  GPU wants.

**The delete-chain subtlety.** `D[k]` depends on `D[k-1]` of the *same* row, an
inherently serial scan. We let **one thread** resolve the whole delete row after
the parallel M/I phase. It is `O(L)` and cheap, and — crucially — doing it the same
way as the CPU keeps the two results **identical** (§6). A production kernel would
parallelise the delete chain with a *scan* (prefix-max) algorithm; we keep it
serial because the recurrence stays legible (the teaching goal).

**`__syncthreads()` between phases** is mandatory: M/I must be fully written before
the delete scan reads `curM[k-1]`, and the current row must be complete before we
copy it into the previous-row buffer for the next residue.

## 5. Numerical considerations

**Integers, not floats — on purpose.** Viterbi scores are sums of log-odds. If we
accumulated them in `float`, the GPU (which fuses multiply-adds and may reorder
reductions) and the host compiler would diverge by ~`1e-6` per cell, accumulating
over a long sequence. CPU and GPU would then *disagree*, and the demo's stdout
would not be reproducible.

We avoid this entirely by **pre-quantising** every log-odds to a scaled integer
(`round(logodds · 1000)`) and running the whole DP in `int`. Integer `max` and `+`
are **associative and exact**, so:

- the **tree reduction** of per-thread maxima is order-independent (deterministic),
- the **CPU and GPU produce bit-identical scores**, and
- we can verify with **tolerance exactly 0** (PATTERNS.md §4, the "exact" case).

**Overflow.** A path of a few hundred residues sums at most a few hundred ×
(~2000 units) ≈ `10⁶`, far inside `int`'s `±2.1·10⁹`. `NEG_INF = −10⁹` is far below
any reachable score yet far enough from `INT_MIN` that adding a negative transition
penalty cannot wrap; we also guard with `if (in ≤ NEG_INF/2) stay NEG_INF` so the
sentinel never drifts.

**No atomics, no races.** Within a block, the three phases are separated by
barriers and threads write disjoint columns, so there is no write-after-write
hazard. Across blocks, each writes only its own `out[i]`.

## 6. How we verify correctness

`src/reference_cpu.cpp` runs the **same** Viterbi search serially: a single
readable loop over sequences, calling the **identical** shared recurrence
(`viterbi_step` / `best_in_row` in `src/hmm_core.h`) that the kernel calls. Because
the math lives in exactly one place (the `__host__ __device__` header — PATTERNS.md
§2) and the arithmetic is integer, the two implementations must agree
**bit-for-bit**.

`main.cu` computes `max |score_cpu[i] − score_gpu[i]|` and **requires it to be 0**.
An independent serial implementation agreeing exactly with the parallel one is
strong evidence both are correct — a transcription bug in the kernel's indexing or
phase ordering would almost certainly perturb at least one of the `N` scores.

**A second, stronger check (the science, not just CPU==GPU):** the synthetic data
*plants the query motif* in sequences `0,1,2`, so a correct search must rank those
three on top. It does: clean motif (`20.000`) > one mutation (`17.500`) >
2-residue insertion (`15.500`) > random background. Recovering the embedded answer
validates that the model and DP actually score homology, not noise.

**Edge cases handled:** the ragged last block of columns (strided loops guard
`k ≤ L`), all-unreachable predecessors (the `NEG_INF` floor), non-standard residues
(mapped to the catch-all index 20), and Windows/Unix line endings in the loader.

## 7. Where this sits in the real world

Production MSA tools do far more than this teaching scan:

- **HMMER / Jackhmmer** estimate the profile HMM's emissions and *per-column*
  transitions from a seed alignment, add a proper begin/end and a **null model**,
  compute the **Forward** score (sum-over-paths, not just the max path) for
  sensitivity, and report calibrated **E-values**. Jackhmmer *iterates*: hits from
  one round rebuild the profile for the next.
- **HHblits** aligns **profile against profile** (HMM-HMM), far more sensitive for
  remote homologs, over the UniClust database.
- **MMseqs2 / Linclust** win on **speed** with a **k-mer prefilter**: a GPU/SIMD
  hash-table seed lookup discards almost all of the database before any expensive
  alignment, then runs a vectorised **Smith-Waterman** (à la CUDASW++) on the
  survivors. ColabFold's MSA server is built on this and is what most people use to
  feed AlphaFold2 today.
- **GPU-HMMER** does on the GPU what we do here — parallelise the Viterbi/Forward
  recurrence over many targets — but with the full Plan7 model and striped/SIMD
  cell layouts for throughput.

What we omit for clarity: the k-mer prefilter, Forward scoring + E-values, profile
*estimation*, per-column transitions, banding, and a parallel (scan-based) delete
chain. The **GPU pattern** — independent per-sequence DPs, constant-memory profile,
shared-memory ping-pong rows, integer determinism — is exactly the one those tools
build on.

---

## References

- **Durbin, Eddy, Krogh, Mitchison — *Biological Sequence Analysis* (1998).** The
  canonical text on profile HMMs and the Viterbi/Forward algorithms; chapters 3–5
  are the math of §2.
- **Eddy, "Accelerated Profile HMM Searches" (HMMER3, PLoS Comp Biol 2011).** How
  real HMMER scores sequences (MSV filter, Forward, vectorised cells) — the model
  this project simplifies.
- **MMseqs2 — <https://github.com/soedinglab/MMseqs2>.** Ultra-fast protein search:
  study the k-mer prefilter + vectorised alignment (the speed story §7 omits).
- **ColabFold — <https://github.com/sokrypton/ColabFold>.** GPU-accelerated MSA for
  AlphaFold2; shows where this scan sits in the structure-prediction pipeline.
- **CUDASW++** — GPU Smith-Waterman; the warp/striped cell layouts a throughput
  version of this kernel would adopt.
- **AlphaFold2 (Jumper et al., *Nature* 2021).** Why the MSA matters: co-evolution
  signal drives the structure prediction this whole search feeds.
