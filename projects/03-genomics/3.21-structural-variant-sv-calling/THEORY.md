# THEORY — 3.21 Structural Variant (SV) Calling

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use. This is a **reduced-scope teaching
> version**; §7 describes the full pipeline._

---

## 1. The science

A genome is read in pieces. A **reference genome** is the agreed-upon sequence for
a species; an individual's DNA is sequenced into millions of **reads** (sub-strings
of their genome) that we *align* back to the reference to find where each came from
and how it differs.

Small differences — a single changed base (SNP) or a few inserted/deleted bases —
are **small variants**. A **structural variant (SV)** is a *large* rearrangement,
conventionally ≥50 bp:

- **Deletion (DEL):** a chunk of the reference is missing in the individual.
- **Insertion (INS):** extra sequence is present that the reference lacks.
- **Inversion (INV):** a chunk is present but reverse-complemented.
- **Translocation (BND):** two distant loci are joined.

SVs matter clinically and biologically far out of proportion to their number: they
cause disorders (e.g., large deletions in cancer driver genes), drive evolution,
and are the hardest class to call accurately. They are detected from **read-support
signatures**: a read that crosses an SV breakpoint cannot align in one piece. For a
**deletion**, a long read aligns as two collinear pieces separated by a jump:

```
reference :  ... A C G T | x x x x x x x x x x | G C A ...     (the x's are deleted)
                         ^ left breakpoint     ^ right breakpoint
read      :  ... A C G T                         G C A ...     (the read "skips" the x's)
```

The read is a **split read**: its left part ends at the *left breakpoint*; its right
part resumes at the *right breakpoint*. Each split read gives a **noisy** estimate of
where the breakpoint is (aligners are imperfect near indels) and how long the
deletion is. A real caller must (1) **refine** each read's breakpoint by careful
re-alignment, and (2) **cluster** the many reads that support the same event into one
confident call with a *genotype* (is the variant on one chromosome copy or both?).

This project models exactly that two-step recipe for deletions.

## 2. The math

**Inputs.**
- Reference window $R = r_0 r_1 \dots r_{L-1}$, each $r_i \in \{A,C,G,T,N\}$ encoded
  as an integer code $\{0,1,2,3,4\}$; length $L$ bp.
- $N$ candidate reads. Read $k$ carries: a raw breakpoint guess $g_k \in [0,L)$, a
  deletion-length estimate $\ell_k$, and a left flank $f_k$ of $F=\texttt{SV\_FLANK}$
  bases (the read bases ending at its breakpoint).

**Step 1 — breakpoint refinement (banded local alignment).**
For a candidate breakpoint position $p$, let $R[p{-}F : p]$ be the $F$ reference
bases ending at $p$. Define the **banded Smith-Waterman** score between the read
flank $f_k$ and that reference flank:

$$
H_{i,j} = \max\!\begin{cases}
0 \\
H_{i-1,j-1} + s(f_k[i], R[p{-}F+j]) \\
H_{i-1,j} + g \\
H_{i,j-1} + g
\end{cases}
\qquad |i-j| \le B,
$$

with substitution score $s(a,b)=+2$ if $a=b$ (and known), $-1$ otherwise; linear gap
penalty $g=-2$; band half-width $B=\texttt{SV\_BAND}$. The local-alignment score is
$\mathrm{SW}(f_k, p) = \max_{i,j} H_{i,j}$. The **refined breakpoint** of read $k$ is

$$
\hat{p}_k = \arg\max_{p \in [g_k - W,\; g_k + W]} \mathrm{SW}(f_k, p),
$$

searching a window of half-width $W=\texttt{SV\_SEARCH}$ around the raw guess; ties
break toward the smaller $p$ (determinism).

**Step 2 — clustering by histogram.**
Build the integer **support histogram** $h$ and the **length-sum** $\sigma$ over
1-bp bins:

$$
h[b] = \big|\{\,k : \hat{p}_k = b\,\}\big|, \qquad
\sigma[b] = \sum_{k:\hat{p}_k = b} \ell_k .
$$

**Step 3 — calls.**
Emit a call at every bin $b$ that is a local maximum of $h$ within $\pm M$
($M=\texttt{SV\_MERGE}$) and clears the support floor $h[b]\ge\tau$. Its support is the
window mass $\mathrm{sup}(b)=\sum_{|o|\le M} h[b+o]$, its consensus length is the
integer mean $\big\lfloor \sum_{|o|\le M}\sigma[b+o] \,/\, \mathrm{sup}(b)\big\rfloor$,
and its genotype follows an integer variant-allele-fraction rule on $(\mathrm{sup}, N)$
(see §6).

**Output.** A list of $(\text{breakpoint}, \text{length}, \text{support}, \text{genotype})$
deletion calls, sorted by breakpoint.

## 3. The algorithm

```
for each read k (INDEPENDENT):                 # Step 1  -- the parallel part
    p_hat = refine_breakpoint(f_k, R, g_k)     #   2W+1 banded SW alignments
    atomic: h[p_hat] += 1                       # Step 2  -- scatter-reduction
    atomic: sigma[p_hat] += l_k
calls = histogram_to_calls(h, sigma, tau)       # Step 3  -- tiny serial merge
```

**Complexity.**
- One banded SW alignment is $O(F\cdot B)$ (only band cells, two rolling rows →
  $O(F)$ memory). Refining one read scans $2W{+}1$ candidates → $O(W F B)$ per read.
- Serial total: $O\!\big(N\,W F B\big)$ for refinement + $O(N)$ voting + $O(L\,M)$
  merge. With $F,B,W,M$ small constants this is **$O(N)$ in the read count** with a
  modest constant — i.e. *linear work, fully parallel across reads*.
- **Parallel depth:** $O(W F B)$ (one read's refinement) — constant in $N$. The
  histogram votes are $O(1)$ atomics. So the whole thing is *work-efficient* and has
  *constant depth per read*: the ideal shape for a GPU.

**Arithmetic intensity / access pattern.** Each thread reads its own flank ($F$
bytes) and a $(2W{+}F)$-byte slice of the reference, does $\sim W F B$ integer
max/add ops, and writes two atomics. The reference slice is the only shared data and
is tiny and hot (rides in L2), so the kernel is compute-bound on the integer DP, not
bandwidth-bound — exactly where a GPU's many integer ALUs shine.

## 4. The GPU mapping

**Thread-to-data mapping.** One **thread per read**:
`k = blockIdx.x * blockDim.x + threadIdx.x`. Thread $k$ owns read $k$, refines its
breakpoint, and casts its vote. There is no cooperation between threads except the
atomic writes into the shared histogram.

**Launch configuration.** `block = 256` threads (multiple of the 32-lane warp; 8
warps give the scheduler enough work to hide the reference's global-memory latency);
`grid = ceil(N / 256)`. The ragged last block is guarded by `if (k >= N) return;`.

**Memory hierarchy.**
- **Registers / local memory:** the two rolling DP rows of banded SW ($F{+}1$ ints
  each) and the read's flank live per-thread. With $F=24$ this is a handful of
  registers — no spilling on sm_75+.
- **Global memory (read-only):** the reference codes and the flattened read arrays.
  Marked `__restrict__` so the compiler may keep loads in registers; the reference
  is small and reused by all threads, so it stays resident in L2 (a natural
  broadcast cache here).
- **Global memory (atomic writes):** the support histogram `hist[L]` and length-sum
  `len_sum[L]`. These are the only writes, via `atomicAdd`.

```
            reads (Struct-of-Arrays in global memory)
   guess[] : g0 g1 g2 ... gN-1
   dellen[]: l0 l1 l2 ... lN-1          reference R[0..L)  (global, read-only, hot in L2)
   flank[] : f0 | f1 | f2 | ... (row-major, F bytes each)
                 │
   grid ───────►│ block 0      block 1            block ceil(N/256)-1
                ▼ ┌───────────┐┌───────────┐ ... ┌───────────┐
   thread k = ... │t0 t1 ..t255││t0 t1 ..t255│    │.. tN-1 ...│   each: banded SW refine
                  └─────┬─────┘└─────┬─────┘     └─────┬─────┘
                        │  atomicAdd │                 │
                        ▼            ▼                 ▼
                    hist[ p̂ ] += 1   and   len_sum[ p̂ ] += l_k   (shared histogram)
                                     │
                                     ▼
                        histogram_to_calls()  (host: peak-merge → SV calls)
```

**Why no CUDA library here.** The catalog's full pattern names cuDNN (CNN
genotyping) and Thrust (read sorting). In this reduced-scope version both the banded
SW and the histogram are **hand-rolled** on purpose — that is the teaching content
(no black boxes, CLAUDE.md §6). To *write Thrust's sort by hand* you would radix-sort
reads by refined breakpoint then segment-reduce; we instead use an atomic histogram,
which is simpler and, with integer bins, exactly as deterministic. See §7 for where
cuDNN/Thrust re-enter at full scope.

## 5. Numerical considerations

**Everything that affects the result is INTEGER.** Base codes, the SW substitution
and gap scores, the DP cells, the breakpoint bins, and the histogram counts are all
integers. The only floating-point in the whole program is the genotype's VAF
comparison — and even that is rephrased as integer inequalities
(`8*support < total`, `4*support >= 3*total`) so **no float ever enters a decision**.

**Why this matters for the GPU — determinism.** Many reads from the same SV vote
into the same bin, so their `atomicAdd`s **collide**. Floating-point atomic adds are
*not associative*: the sum depends on the (nondeterministic) order threads arrive, so
a float histogram would differ run-to-run and would not match the CPU. **Integer
atomic adds commute** — order is irrelevant — so the GPU histogram equals the CPU
histogram *bit-for-bit*, every run (PATTERNS.md §3/§4). This is the same lesson as
flagships 5.01 (integer energy quanta) and 11.09 (fixed-point centroid sums).

**Race conditions.** The only shared writes are the two `atomicAdd`s; refinement is
purely local to each thread. There are no read-modify-write hazards on the reference
(read-only) or the flanks (each thread reads only its own).

**Precision of the DP.** Scores are small ($\le 2F = 48$), so 32-bit `int` cannot
overflow. The length-sum uses `unsigned long long` to stay safe when scaled to
millions of reads.

## 6. How we verify correctness

**The CPU reference** (`src/reference_cpu.cpp`, `sv_call_cpu`) runs the identical
pipeline with a plain serial loop: for each read it calls the *same* `sv.h`
functions the kernel calls, votes into a `std::vector` histogram, and runs the *same*
`histogram_to_calls` merge. Because the per-read math is shared via the
`__host__ __device__` header (`src/sv.h`, the HD-macro idiom, PATTERNS.md §2), the
CPU and GPU evaluate byte-identical formulas.

**The check.** `main.cu` compares the two histograms bin-by-bin (`hist` and
`len_sum`) and the two call lists field-by-field. The **tolerance is exactly zero** —
integer arithmetic plus commuting integer atomics means agreement is *exact*, not
approximate (PATTERNS.md §4, "exact"). A single mismatched bin is a real bug.

**A second, stronger check — recovering the planted truth.** The synthetic sample
bakes in a *known* deletion (breakpoint 120, length 50). The demo reports whether the
top call lands within $\pm M$ of that truth (`recovered: YES`). That validates the
*science* — the refinement actually pulls jittered guesses back to the true
breakpoint — not merely that CPU==GPU.

**Genotype, honestly.** `sv_geno_from_vaf` uses fixed integer VAF cutoffs
(`VAF<1/8`→0/0, `≥3/4`→1/1, else 0/1). In the synthetic mix the denominator (total
reads) is dominated by supporting reads, so VAF≈0.75 and the call is `1/1`. This is a
*teaching* genotype, not a likelihood model — Exercise 3 adds reference-supporting
reads to produce a `0/1`.

**Edge cases handled.** ragged last block (`k>=N`), refined breakpoints that fall off
the reference (dropped), empty bins in the merge (skipped), and divide-by-zero in the
consensus length (guarded).

## 7. Where this sits in the real world

Production SV callers do far more than this teaching version:

- **Input.** They read aligned **BAM/CRAM** (not a toy text file), extract split
  reads, **discordant read pairs** (mates farther/closer/wrong-orientation than
  expected), and **read-depth** changes — multiple orthogonal signatures, not just
  split reads.
- **All SV types.** Insertions, inversions, duplications, and translocations, each
  with its own breakpoint signature; we do deletions only.
- **Clustering at scale.** **Sniffles2** and **cuteSV** cluster millions of
  signatures with fast spatial data structures (the catalog's *Thrust read cluster
  sorting* would radix-sort signatures then segment-reduce — the GPU version of our
  atomic histogram). **pbsv** does careful **local re-assembly + realignment** at each
  breakpoint, a heavier version of our banded SW.
- **Genotyping with deep learning.** **DeepVariant**/**DeepSV** render the pileup
  around a candidate as an *image* and classify it with a CNN; at population scale
  this is **batched cuDNN inference** across millions of candidates — the second GPU
  hotspot the catalog names, and a natural follow-on project (cf. flagship 7.10's
  1-D conv and the medical-AI domain).
- **Merging across samples.** **SURVIVOR** merges per-sample callsets into a cohort
  matrix — the population-scale step that motivates multi-GPU.

The full pipeline is *Active R&D*: long-read SV calling accuracy is still improving,
and GPU genotyping is an emerging direction. This project teaches the two load-bearing
GPU ideas (independent per-read realignment + deterministic atomic clustering) on a
problem small enough to verify exactly.

---

## References

- Smith, T.F. & Waterman, M.S. (1981), *Identification of common molecular
  subsequences* — the local-alignment recurrence we band.
- Sedlazeck et al., **Sniffles** / Sniffles2 — fast long-read SV clustering:
  https://github.com/fritzsedlazeck/Sniffles (study the clustering heuristics).
- PacBio **pbsv** — local-realignment SV calling: https://github.com/PacificBiosciences/pbsv.
- Jiang et al., **cuteSV** — clustering-based SV caller: https://github.com/tjiangHIT/cuteSV.
- **NGSEP** — variant suite with GPU-amenable CNN scoring: https://github.com/NGSEP/NGSEPcore.
- Poplin et al., **DeepVariant** (2018) — CNN pileup-image genotyping (the model
  DeepSV-style SV genotyping borrows).
- **GiaB** HG002 SV benchmark (NIST): https://www.nist.gov/programs-projects/genome-bottle —
  the gold-standard truth set to evaluate a real caller against.
- repo docs: `docs/PATTERNS.md` §1 (independent jobs + atomic reduce), §2 (the
  `__host__ __device__` core), §3/§4 (determinism & exact tolerance).
