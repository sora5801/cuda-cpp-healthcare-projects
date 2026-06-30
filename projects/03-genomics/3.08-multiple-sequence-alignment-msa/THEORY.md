# THEORY — 3.8 Multiple Sequence Alignment (MSA)

> Read [README.md](README.md) first for the one-paragraph picture. This document
> is the deep dive: the biology, the math, the algorithm with complexity, how it
> maps onto the GPU, the numerics, how we verify, and where this sits relative to
> production MSA tools. Audience: comfortable with C++, new to CUDA and to
> bioinformatics.

---

## The science

A **multiple sequence alignment** arranges three or more biological sequences
(DNA, RNA, or protein) into a grid so that each **column** holds residues that are
**homologous** — descended from the same position in a common ancestor. Gaps
(`-`) are inserted to account for insertions and deletions ("indels") that
accumulated as the sequences diverged. MSAs are foundational:

- **Phylogenetics** — the columns are the characters from which evolutionary trees
  are inferred.
- **Conserved-site discovery** — columns that stay constant across a family mark
  functionally important residues (active sites, binding pockets).
- **Structure prediction** — a *deep* MSA of a protein family is the single most
  important input to AlphaFold2-class predictors; covariation between columns
  encodes 3-D contacts.
- **Variant analysis / primer design** — aligning patient or strain sequences to a
  reference reveals where they differ.

The catalog frames MSA as **Active R&D**: aligning *N* sequences optimally is
NP-hard in *N*, so every practical tool is a heuristic. The dominant heuristic
family is **progressive alignment** (ClustalW, Clustal Omega, MAFFT, T-Coffee):
align the most similar sequences first, then progressively add the rest, guided by
a tree built from pairwise similarities. This project implements the simplest
correct member of that family.

---

## The math

### Pairwise global alignment (Needleman-Wunsch)

For two sequences `a` (length `la`) and `b` (length `lb`) over an alphabet, define
a **score matrix** `H` of size `(la+1) × (lb+1)`. With a **linear gap** penalty
`g` (here `NW_GAP = −2`) and a substitution score `s(x,y)` (`+2` match, `−1`
mismatch), the **Needleman-Wunsch** recurrence is:

```
H[0][0] = 0
H[i][0] = i · g                 (leading gaps in b)
H[0][j] = j · g                 (leading gaps in a)
H[i][j] = max( H[i-1][j-1] + s(a_i, b_j),   // align a_i with b_j   (diagonal)
               H[i-1][j]   + g,             // gap in b             (up)
               H[i][j-1]   + g )            // gap in a             (left)
```

The optimal **global** alignment score is `H[la][lb]` (bottom-right corner).
Unlike Smith-Waterman (project `3.01`) there is **no `max(0, …)`**: global
alignment must consume *every* residue of both sequences, so it cannot "restart".

### From score to distance

We turn a pairwise score into a **distance** in `[0,1]`. The maximum achievable
score for a sequence is its *self*-score (all matches, no gaps) `self = L·MATCH`.
Normalising by the shorter sequence's self-score:

```
D[a][b] = clamp01( 1 − score(a,b) / min(self_a, self_b) )
```

Identical sequences → `D = 0`; highly divergent ones → `D → 1`. This crude
"fractional-identity" distance is enough to order the sequences for the guide
step; real tools use corrected evolutionary distances (e.g. Kimura) — see
*real world*.

### Center-star guide

The **center-star** method picks the single most representative sequence: the
**center** `c` is the one whose total distance to all others is smallest,

```
c = argmin_c  Σ_b D[c][b]            (ties broken by lowest index)
```

Every other sequence is then aligned **to the center**, and the pairwise
alignments are merged in the center's coordinate frame. Star alignment has a
classic guarantee: its Sum-of-Pairs score is within a factor `2 − 2/N` of optimal.

### Sum-of-Pairs (SP) score

We grade a finished MSA column-by-column. For a column, sum the pairwise scores
over all `C(N,2)` row pairs: two residues score `s(x,y)`; a residue vs a gap
scores `g`; gap vs gap scores `0`. The MSA's SP score is the sum over all columns.
Higher is better; it is the standard objective progressive aligners approximate.

---

## The algorithm

```
STAGE 1  distance matrix          for each pair (a,b):  score = NW(a,b);  D = 1 - score/self
STAGE 2  guide (center-star)      c = argmin_c Σ_b D[c][b]
STAGE 3  progressive assembly     for each b≠c: align b to c; merge into the center frame
         grade                    SP = sum_of_pairs(MSA)
```

### Worked example of the merge ("once a gap, always a gap")

Suppose the center is `ACGT`. Aligning sequence `b1` to it yields
`A C G T` / `A - G T` (b1 has a deletion). Aligning `b2` yields
`A C G T` / `A C G A`. Aligning `b3` inserts a base *into the center*:
`A C - G T` / `A C T G T`. The center is gapped differently by each pairwise
alignment. The merge takes the **union** of all center-gap positions (here a
single inserted column before `G`, demanded by `b3`) and re-threads every row
through that common layout. `b1`/`b2`, which had no insertion there, get a `-` in
the new column. The result is one grid where every row has equal width and the
center's residues line up in shared columns.

> **Honest caveat.** Residues that different sequences insert into the *same*
> merged gap column are merely co-located, not mutually aligned — star alignment
> only guarantees alignment *to the center*, never insertion-vs-insertion. Tree-
> based progressive alignment with profile-profile steps fixes this; we keep the
> simpler star method because it makes the GPU stage and the merge legible.

### Complexity

- **Stage 1:** `P = N(N-1)/2` alignments, each `O(L²)` time and (score-only)
  `O(L)` space ⇒ **`O(N²·L²)` time**. This is the cost that dominates and the
  cost the GPU attacks.
- **Stage 2:** `O(N²)` to sum each row of the distance matrix.
- **Stage 3:** `N−1` alignments with traceback (`O(L²)` memory each) + an
  `O(N·width)` threading pass; SP scoring is `O(width·N²)`. All cheap vs Stage 1
  for large N.

Serial total is `O(N²·L²)`; the GPU keeps the same *work* but does the `O(N²)`
independent alignments concurrently.

---

## The GPU mapping

The catalog prescribes **"one CUDA thread block per pairwise alignment"** for the
distance-matrix phase. We follow it exactly.

```
flat pair list (a<b):   p=0 -> (0,1)   p=1 -> (0,2)  ...  p=P-1 -> (N-2,N-1)
grid:                   P blocks, one per pair
block:                  32 threads (one warp); lane 0 drives the serial DP
shared memory / block:  2·(max_len+1) ints  = the two rolling DP rows (prev,curr)
output:                 d_score[a*N+b] = d_score[b*N+a] = NW score   (no races)
```

**Why a block per pair (not a thread per pair)?** Each NW alignment needs `O(L)`
working memory (two DP rows). Owning a whole block lets that scratch live in fast
on-chip **shared memory** (≈100× lower latency than global), and gives the
scheduler a full warp to swap in while another block waits on memory. With one
*thread* per pair (the `1.12` pattern), the rows would have to sit in slow global
memory or burn a huge register budget.

**Why lane 0 only, in this teaching version?** The DP recurrence is itself
serial along a row; parallelising *within* a pair needs the anti-diagonal
wavefront of `3.01`. Doing both levels at once would bury the lesson. So here the
block's *role* is to own the shared scratch for its pair, lane 0 runs the
recurrence, and the **across-pairs** parallelism — the catalog's point — is what
the grid expresses. Exercise 1 upgrades the within-pair work to the wavefront.

**The shared `__host__ __device__` core (`nw_core.h`).** The function
`nw_score_core()` *is* the recurrence, compiled by **both** the host compiler
(for `reference_cpu.cpp`) and nvcc (for the kernel). The CPU reference loops it
over all pairs; the kernel calls it once per block. Because it is the same integer
code, the two distance matrices are **bit-identical** (PATTERNS.md §2). The
`NW_HD` macro expands to `__host__ __device__` under nvcc and to nothing under the
host compiler.

**Memory hierarchy used:** sequences and the pair list live in **global** memory
(read-only, `__restrict__`); the DP rows live in **dynamic shared** memory
(requested as the third `<<<>>>` argument); the score is written straight back to
**global** memory. No atomics or `__syncthreads` are needed — independent blocks
never touch the same output cell.

**Stages 2–3 stay on the host.** They are `O(N²)` and `O(N·L²)` bookkeeping, not
the throughput bottleneck, and they involve data-dependent control flow
(traceback, gap merging) that is awkward and pointless to put on the GPU at this
scale. Keeping them on the host also means the *same* assembly code consumes
either matrix — which is how we know the GPU-driven and CPU-driven pipelines
produce the identical alignment.

---

## Numerical considerations

- **Integer scoring ⇒ exact, deterministic.** Every score is an `int`; `max` and
  `+` on integers commute and associate exactly, independent of thread/lane/order.
  There is **no floating-point nondeterminism** anywhere in the scored quantity,
  so CPU and GPU agree to the bit (PATTERNS.md §3, §4). The derived *distance* is a
  `double`, but it is computed by the **same formula** on both paths from the same
  integer score, so it too matches.
- **No atomics.** Distinct pairs write distinct `d_score` cells; the symmetric
  `(a,b)`/`(b,a)` writes come from the *same* block, so there is no inter-block
  race. This is why we can claim exact reproducibility without integer fixed-point
  tricks (contrast `5.01`/`11.09`, which *do* reduce with atomics and therefore
  must use integer accumulation).
- **Overflow.** Scores are bounded by `L·MATCH` (≤ a few hundred here), far inside
  `int`; SP is a `long long` to be safe for larger N.
- **Determinism of the alignment itself.** No RNG; all tie-breaks are fixed
  (center: lowest index; traceback: diagonal > up > left). So stdout is
  byte-identical every run — the demo can diff it.

---

## How we verify correctness

1. **GPU == CPU, exactly.** `main.cu` builds the Stage-1 score matrix on both the
   CPU (`distance_matrix_cpu`) and the GPU (`distance_matrix_gpu`) and asserts
   **every cell is equal** (`mismatches == 0`, `max_abs_diff == 0`). Tolerance is
   literally **zero** because the scored quantity is integer and the math is the
   shared `nw_score_core()`.
2. **The science is recoverable.** The committed sample is a family of mutated
   descendants of one ancestral motif. A correct MSA lines the conserved core up
   into **starred** (fully conserved) columns — visible in the demo output. If the
   alignment logic were wrong, the core would not align and the star line would be
   sparse.
3. **Cross-config determinism.** The `Release` and `Debug` binaries produce
   byte-identical stdout (checked during development), confirming no
   optimisation-dependent or undefined behaviour leaked in.

`demo/expected_output.txt` was **captured from a real run**, never hand-written.

---

## Where this sits in the real world

This is a deliberately **reduced-scope teaching version**. Production MSA differs
in every stage:

| Stage | This project | Production (MAFFT / Clustal Omega / T-Coffee) |
|---|---|---|
| Distance | full NW, fractional identity | `k`-mer / PartTree fast distances; corrected (Kimura) distances |
| Guide | center-star, lowest index | neighbor-joining or UPGMA **tree**; `mBed` for huge N |
| Assembly | align each seq to the center | **profile-profile** alignment along the guide tree |
| Gaps | linear (`g` per gap) | **affine** (open + extend), position-specific gap penalties |
| Refinement | none | iterative refinement (tree splits, consistency objectives) |
| Scoring | Sum-of-Pairs | SP + consistency (T-Coffee), column reliability |

On the **GPU** side, the catalog points to real systems: **MAFFT-PartTree** has a
GPU distance-phase prototype (the ~6× figure); **CUK-Band (2024)** puts center-
star MSA with *banded* DP on CUDA (it bounds the DP to a diagonal band, turning
each `O(L²)` alignment into `O(L·w)`); **MMseqs2-GPU** accelerates the iterative
*search* that builds the deep MSAs feeding AlphaFold2 — often the single most
time-consuming step of a structure-prediction pipeline. The pattern we teach here
— *the all-vs-all pairwise phase is the embarrassingly parallel part* — is exactly
why those tools target it first.

A faithful next step (Exercise 1) is to parallelise *within* each pair with the
anti-diagonal wavefront of `3.01`, giving a two-level GPU MSA: blocks across pairs,
warps across each pair's diagonal. Add banding (CUK-Band) and affine gaps and you
have the skeleton of a modern GPU progressive aligner.
