# THEORY — 3.23 Splice-Aware RNA Alignment

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

This project ships a **reduced-scope teaching version** (CLAUDE.md §13): the
*scientific heart* of a splice-aware aligner — the spliced dynamic-programming
recurrence with a canonical-splice-site-scored intron move — applied to a batch
of short reads against one short reference. The genome-scale machinery (suffix
arrays, FM-indexes, chaining) is **described** here, not implemented; see §7.

---

## 1. The science

A eukaryotic **gene** is not one contiguous block of coding sequence. It is a
mosaic of **exons** (kept) and **introns** (removed). When the gene is
transcribed, the cell produces a pre-mRNA containing both, then the **spliceosome
splices out the introns** and joins the exons into a **mature mRNA**:

```
 genomic DNA :  [== exon1 ==]---intron1---[= exon2 =]---intron2---[== exon3 ==]
                                   |  splicing removes introns  |
 mature mRNA :  [== exon1 ==][= exon2 =][== exon3 ==]
```

**RNA-seq** sequences the mature mRNA, yielding short **reads**. A read that
happens to straddle an **exon-exon junction** contains sequence from two exons
that are *adjacent in the mRNA* but *far apart in the genome* (separated by the
spliced-out intron). To map such a read back onto the **genome**, the aligner
must place part of the read on one exon, **skip the intron**, and place the rest
on the next exon. An aligner that cannot do this either fails to map junction
reads or mis-maps them — and junction reads are exactly the ones that tell you
*which* splice isoform was expressed. Hence **splice-aware alignment** is the
foundation of transcript quantification, isoform discovery, and fusion detection.

The biology gives us a crucial clue: introns are not random. ~99% of human
introns obey the **GT-AG rule** — the intron begins with the dinucleotide `GT`
(the **donor** / 5′ splice site) and ends with `AG` (the **acceptor** / 3′ splice
site). A splice-aware aligner uses this signal to prefer biologically plausible
intron boundaries.

---

## 2. The math

We align a **read** `q = q_1 … q_M` to a **reference** `r = r_1 … r_N` (both over
the DNA alphabet `{A,C,G,T}`, with RNA `U` folded to `T`). We seek the best
**local** alignment (Smith-Waterman), extended with an **intron skip**.

Define the score matrix `H ∈ ℤ^{(M+1)×(N+1)}`, where `H[i][j]` is the best score
of a local alignment of the read prefix `q_1…q_i` ending with `q_i` aligned to
reference position `j`. With integer substitution scores

```
 s(a,b) = MATCH   (+2)  if a = b
        = MISMATCH (-1) otherwise
```

and a linear gap penalty `GAP = -2`, the recurrence is:

```
 H[i][j] = max {
     0,                              // (local restart — the Smith-Waterman floor)
     H[i-1][j-1] + s(q_i, r_j),      // (M)  align q_i to r_j      (diagonal)
     H[i-1][j]   + GAP,              // (I)  q_i is an insertion    (up)
     H[i][j-1]   + GAP,              // (D)  one-base ref deletion   (left)
     E[i][j]                         // (N)  intron-spliced match
 }
```

with row 0 and column 0 initialised to 0. The novelty is the **intron term**:

```
 E[i][j] = max over k of  H[i-1][k] + s(q_i, r_j) + φ(k, j)
```

read as: *q_i matches r_j (the `s` term), and immediately before column j there
is a spliced-out intron `r[k+1 … j-1]`, whose left edge connects to `q_{i-1}`
aligned at column k.* The **intron penalty** `φ` is

```
 φ(k, j) = INTRON_OPEN (-6)                 if the intron is canonical-free
         = INTRON_OPEN + CANON_BONUS (-6+4) if r[k+1..k+2]=GT and r[j-2..j-1]=AG
         = -∞                               if the span is shorter than MIN_INTRON
```

and `k` is restricted to a **band** `j-1-MAX_INTRON ≤ k ≤ j-1-MIN_INTRON`
(`MIN_INTRON = 4`, `MAX_INTRON = 64` here). Two design facts matter:

- `φ` is **flat in intron length** — skipping a 40-base or a 40 000-base intron
  costs the same. That is the entire point: a base-by-base gap of a 40-base
  intron would cost `40·GAP = -80`, versus `-6` (or `-2` canonical) for the `N`
  move.
- A canonical intron nets `INTRON_OPEN + CANON_BONUS = -2 < 0`. It is still a
  *penalty*, never a reward — otherwise the optimiser would **fabricate** introns
  to harvest the bonus. But a canonical jump (`-2`) beats a non-canonical one
  (`-6`), so the aligner is steered to the biologically correct boundary.

The **alignment score** is the maximum cell of `H`; the **alignment** is the path
from that cell back to a 0, read off as a **CIGAR** string where the `N`
operation denotes the skipped intron (e.g. `12M40N12M`).

---

## 3. The algorithm

**Per read** (the serial reference, `align_one_cpu` + `cell_recurrence`):

```
for i in 1..M:                         # read positions (rows)
  for j in 1..N:                       # reference positions (columns)
    v = max(0, M-move, I-move, D-move)
    for k in [j-1-MAX_INTRON .. j-1-MIN_INTRON]:   # the intron band
      v = max(v, H[i-1][k] + s(q_i,r_j) + φ(k,j))
    H[i][j] = v
    track global best cell
```

**Complexity.** The three classic moves make the base DP `O(M·N)`. The intron
move scans a band of width `B = MAX_INTRON - MIN_INTRON + 1` at every cell, so a
read costs `O(M·N·B)`. For a **batch** of `R` reads it is `O(R·M·N·B)`. The band
`B` is the key knob: an unbounded intron search would be `O(M·N²)` (real introns
*are* long, which is why production tools cap intron length and/or seed first).

**Traceback** (`traceback_cigar`) re-derives each move by checking which
predecessor reproduces `H[i][j]` exactly (integers ⇒ no floating-point
ambiguity), with a fixed preference order **M > N > I > D** so the path — and thus
the printed CIGAR — is deterministic.

ASCII picture of an intron move (the `N` jump on the diagonal):

```
       k                         j
 r:  …A G | g t …………………… a g | C T…       (intron r[k+1..j-1], GT…AG)
        \__________ skip ________/  \
 q:  …q_{i-1}                        q_i        q_i matches r_j; the intron
                                                connects to q_{i-1} at column k
```

---

## 4. The GPU mapping

The independence axis is **across reads**: each read is a self-contained DP
problem that reads the shared reference and writes its own table. So:

- **grid:** `R` blocks — `blockIdx.x` = read index. No cross-block communication,
  no atomics: each block owns a disjoint slice of the global DP buffer.
- **block:** one active thread (`threadIdx.x == 0`) runs that read's serial DP.
  *Why not many threads per read?* Within a row, the **D (left) move** reads
  `H[i][j-1]`, a left-to-right serial chain; filling a row in parallel needs an
  anti-diagonal **wavefront** (as in project `3.01`). The intron `N` move adds a
  long-range read of the **previous** row `H[i-1][k]`. Bolting a wavefront *and* a
  banded intron scan onto one kernel is correct but hard to read, so for these
  short reads we keep the per-read DP serial and harvest parallelism from the
  thousands of blocks instead. That is the honest mapping for *many small
  independent alignments* (`docs/PATTERNS.md §1`).

**Memory hierarchy.** The DP table lives in **global memory** — a long reference
makes `(M+1)(N+1)` ints too large for the 48–100 KB of shared memory per block.
The reference and reads are also in global memory; the reference is read by every
block, so on a real workload it is a prime candidate for **constant** or
**texture** memory (broadcast/cache friendly) — an exercise. The per-read scalar
outputs (score, endpoint) are tiny.

**The shared core.** `cell_recurrence`, `is_canonical_intron`, and `intron_score`
live in `reference_cpu.h` decorated `__host__ __device__` (the `HD` macro). The
CPU loop and the GPU kernel call **the same functions**, so they cannot drift
(`docs/PATTERNS.md §2`). This is what makes the verification *exact* rather than
approximate.

```
 reference_cpu.h  ── HD math (cell_recurrence, splice scoring) ──┐
        │                                                        │
        ▼ (host compiler)                                        ▼ (nvcc)
 reference_cpu.cpp  align_one_cpu()                 kernels.cu  align_batch_kernel()
        └──────────────── identical integers ───────────────────┘
                                  │
                         main.cu verifies cell-by-cell
```

---

## 5. Numerical considerations

- **Everything is integer.** Scores, penalties, and the table are `int`. Integer
  addition is associative and order-independent, so the CPU and GPU produce
  **bit-identical** results regardless of scheduling — there is no
  floating-point reduction, no atomics, and therefore no nondeterminism
  (`docs/PATTERNS.md §3`). This is why we can verify with an **exact** tolerance.
- **No race conditions.** Each block writes only its own table slice and its own
  scalar outputs; nothing is shared-and-mutated across blocks, so no
  `__syncthreads`, no atomics, no memory fences are needed.
- **Overflow.** With `MATCH = +2` and reads of tens to a few hundred bases, scores
  stay far below `INT_MAX`; an `int` table is ample. (A genome-scale tool with
  very long reads would still be safe in 32-bit, but it is worth checking.)
- **The `-∞` sentinel.** `intron_score` returns a large negative number for a
  span too short to be an intron; we test `<= -1000000` rather than using actual
  `INT_MIN` so adding it to a finite score cannot underflow.
- **Determinism of ties.** When two paths score equally (a real phenomenon at
  ambiguous splice boundaries), the fixed traceback preference order makes the
  reported CIGAR reproducible; the *score* is unaffected by the choice.

---

## 6. How we verify correctness

Three nested checks, all **exact** (`== 0` tolerance — justified because the same
integer operations run on both sides, `docs/PATTERNS.md §4`):

1. **Scores** — `res_cpu[r].score == res_gpu[r].score` for every read.
2. **Endpoints** — the best-cell `(i,j)` matches for every read.
3. **Every DP cell** — the full `R·(M+1)·(N+1)` table buffers are compared
   element-by-element (`cell_mismatches == 0`, `max_abs_cell_diff == 0`). This is
   the strongest check: it proves the GPU reproduced the *entire* computation,
   not just the summary.

A second, **scientific** check is built into the synthetic data (`docs/PATTERNS.md
§6`): the reads are engineered fragments of a known mRNA, so the *expected* CIGAR
is known a priori (`12M40N12M` for a one-junction read, `6M40N24M48N6M` for the
double-junction read). The demo recovering exactly those CIGARs validates the
*science*, not merely CPU == GPU agreement. The traceback runs on the **GPU**
tables, so a correct printed CIGAR also confirms the GPU table is right.

Edge cases covered: exon-internal reads (must yield pure `M`, no spurious
intron), reads at different junction offsets, and a read crossing **two** introns.

---

## 7. Where this sits in the real world

This teaching version implements the **scoring + DP** core. A production
splice-aware aligner adds several layers we deliberately omit:

- **Seeding, not full DP.** STAR builds an **uncompressed suffix array** (~28 GB
  for the human genome) and finds *Maximal Mappable Prefixes* to seed alignments
  in `O(read length)`; HISAT2 uses a **graph FM-index** (BWT) that *encodes known
  splice sites into the index itself*. Only a band around each seed is then
  refined by DP. We DP the whole read×reference rectangle — fine for a short
  reference, hopeless for a 3 Gb genome. The catalog's "page-locked host memory
  for the suffix array loaded by GPU" targets exactly this seeding/querying step.
- **Chaining.** minimap2 (`-ax splice`) finds many short anchors and **chains**
  them across introns with a dynamic program over anchors, then fills gaps with a
  banded/wavefront extension — the natural home of a *GPU wavefront* kernel for
  long reads.
- **Better splice models.** Real tools score donor/acceptor sites with
  **maximum-entropy** position models, handle the minor `GC-AG` and `AT-AC`
  intron classes, use known-junction annotation (GENCODE GTF), and require a
  minimum overhang on each side of a junction. Our single canonical bonus is a
  one-line caricature of this.
- **Scale & engineering.** Multi-sample pipelining via **CUDA streams**, GPU
  **hash tables** for the splice-junction index, **Thrust** sorts to cluster
  seeds, paired-end constraints, multi-mapping resolution, and quality-aware
  scoring. None of these change the *idea* taught here; they make it fast and
  accurate at genome scale.

**Bottom line:** what you learn here — the spliced DP recurrence, the flat
canonical-scored `N` move, the CIGAR-with-N output, and the batched
one-block-per-read GPU mapping — is the conceptual core that the production tools
wrap in heavy indexing and chaining machinery.
