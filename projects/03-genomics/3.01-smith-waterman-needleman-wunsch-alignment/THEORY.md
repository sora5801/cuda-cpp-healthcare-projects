# THEORY — 3.01 Smith-Waterman / Needleman-Wunsch Alignment

> For a reader who knows C++ but is new to CUDA and to sequence alignment.
> See [README.md](README.md) for the tour and build. _Educational only._

## 1. The science

Biological sequences (DNA, RNA, protein) evolve by substitutions, insertions, and
deletions. To compare two sequences we **align** them — line them up, allowing
gaps, to maximize matched residues. Alignment underlies homology search (BLAST),
genome assembly, variant calling, and phylogenetics. **Global** alignment
(Needleman-Wunsch) lines up the sequences end to end; **local** alignment
(Smith-Waterman) finds the single best-matching *sub*-region, which is what you
want when a short motif sits inside longer, unrelated flanks.

## 2. The math

Let `q` (length `M`) and `t` (length `N`) be the sequences. Define a substitution
score `s(a,b)` (here `+2` if `a=b`, `-1` otherwise) and a linear gap penalty
`g = -2`. The Smith-Waterman score matrix `H ∈ ℤ^{(M+1)×(N+1)}` is

```
H[i][0] = 0,   H[0][j] = 0
H[i][j] = max( 0,
               H[i-1][j-1] + s(q_i, t_j),     # substitution (diagonal)
               H[i-1][j]   + g,               # deletion: gap in t (up)
               H[i][j-1]   + g )              # insertion: gap in q (left)
```

The optimal local score is `max_{i,j} H[i][j]`; the alignment is recovered by
**traceback** from that cell, following whichever predecessor produced it, until
a `0` is reached. **Needleman-Wunsch** (global) drops the `max(0, …)`, initializes
`H[i][0]=i·g`, `H[0][j]=j·g`, and reads the score from `H[M][N]`.

## 3. The algorithm

```
fill H row by row (serial reference), OR diagonal by diagonal (parallel):
  for each cell (i,j): H[i][j] = recurrence above
score = max cell ;  alignment = traceback(score cell)
```

**Complexity.** Filling is `Θ(M·N)` work. Serially it is `M·N` dependent steps.
The key observation for parallelism: `H[i][j]` depends only on `(i-1,j-1)`,
`(i-1,j)`, `(i,j-1)` — all with a smaller `i+j`. So if we group cells by
`d = i+j` (**anti-diagonals**), every cell on diagonal `d` depends only on
diagonals `d-1` and `d-2`. Cells *within* a diagonal are independent. The
**critical path** (depth) is `M+N-1` diagonals; the **work** is still `M·N`, now
spread across the up-to-`min(M,N)` cells of each diagonal.

## 4. The GPU mapping

We sweep `d = 2 … M+N`. For each `d` we launch `sw_diagonal_kernel` with one
thread per cell on that diagonal:

```
  thread k  ->  i = i_lo + k,   j = d - i
  valid rows:   i in [ max(1, d-N) .. min(M, d-1) ]   (so 1 <= j <= N)
```

```
   d=2  d=3  d=4  d=5 ...           each launch fills one frontier; it READS
    \    \    \    \                only cells written by earlier launches
   H[1,1]                           (diagonals d-1, d-2) -> no intra-launch
        H[1,2] H[2,1]               hazard, no atomics, no __syncthreads.
              H[1,3] H[2,2] H[3,1]
```

**Memory.** `H` lives in global memory (`(M+1)(N+1)` ints, row stride `N+1`); the
two sequences are small global arrays read as `q[i-1]`, `t[j-1]`. No shared memory
is used in this teaching version (Exercise 5 adds tiling). Because each launch
only reads finalized cells, there are **no race conditions** and no
synchronization primitives — the *ordering between diagonals* is provided for
free by launching them in sequence.

**Why integers / determinism.** Scores are integers, so the GPU and CPU compute
the **identical** matrix bit-for-bit (`matrix mismatches = 0`). There is no
floating-point and no cross-thread reduction during the fill, so nothing can
reorder.

**Occupancy & the honest caveat.** Early and late diagonals are short (few
cells), so they underutilize the GPU; only the middle diagonals are wide. Worse,
we pay a **kernel-launch cost** `M+N-1` times. For a single modest pair this
overhead dominates and the GPU is *slower* than the CPU — a genuinely useful
lesson about when GPUs help. Production fixes: a **single** persistent kernel that
loops over diagonals with a grid-wide barrier (cooperative groups); **shared-
memory tiling** of the active band; and, above all, **batching many pairs** (one
block per query-target pair), which turns alignment back into an embarrassingly
parallel workload like `1.12`.

## 5. Numerical considerations

Pure integer arithmetic: no precision or stability concerns, fully deterministic.
The only subtlety is **traceback tie-breaking**: when several predecessors yield
the same score there are multiple optimal alignments. We pick a fixed priority
(diagonal > up > left) and traceback once, on the host, from the GPU matrix — so
the reported alignment is deterministic and independent of GPU thread scheduling.

## 6. How we verify correctness

`main.cu` fills the matrix twice — `sw_cpu` (a plain triple loop) and `sw_gpu`
(the wavefront) — and compares **every cell**. Equality of two independent
implementations (one obviously correct, one parallel) is strong evidence the
kernel is right, and it directly validates the wavefront ordering. The synthetic
sample embeds a mutated motif so the optimal local alignment is non-trivial
(it exercises matches, mismatches, and gaps in the traceback).

## 7. Where this sits in the real world

CUDASW++4.0, GenomeWorks, and WFA-GPU add what this teaching version omits:
**affine gaps** (Gotoh's H/E/F recurrence), **substitution matrices** (BLOSUM/PAM)
for proteins, **striped/SIMD inter-sequence** parallelism, **DPX** hardware DP
instructions on Hopper, and **database tiling** so a query streams against
millions of targets. The **wavefront alignment algorithm (WFA)** reformulates the
problem to work proportional to the *alignment score* rather than `M·N`, a large
win for similar long sequences. The core recurrence you see here is unchanged —
everything else is engineering for scale.

## References

- Smith & Waterman (1981), *Identification of Common Molecular Subsequences* — the algorithm.
- Needleman & Wunsch (1970) — global alignment. Gotoh (1982) — affine gaps.
- Schmidt et al., **CUDASW++4.0** (2024) — modern GPU SW with DPX.
- NVIDIA CUDA C++ Programming Guide — grid-stride/wavefront patterns, cooperative groups.
