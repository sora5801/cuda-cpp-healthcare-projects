# THEORY — 3.17 CRISPR Guide Design & Off-Target Scoring

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See
> [README.md](README.md) for the quick tour and build steps.
>
> _Educational only — not for clinical use. The CFD weights here are a synthetic
> teaching model, not the published table; see §7._

---

## 1. The science

**CRISPR–Cas9** is a programmable DNA-cutting system. The Cas9 protein is
guided to a target by a ~20-nucleotide **guide RNA (gRNA) spacer**: wherever the
genome contains a 20-base "protospacer" that base-pairs with the spacer **and**
is immediately followed by a short **PAM** ("protospacer adjacent motif"), Cas9
can bind and cut. For the common *S. pyogenes* Cas9 (SpCas9) the PAM is **NGG**
— any base, then two guanines.

The catch that makes this a computational problem: **Cas9 tolerates mismatches.**
A spacer designed to cut gene X will also cut, more weakly, at other genomic
sites that resemble it. Those unintended cuts are **off-targets**, and they are
the central safety concern in any CRISPR experiment or therapy. So designing a
good guide requires, for every candidate spacer:

1. **Find every near-match in the genome** — every window that has a PAM and
   differs from the spacer in only a few positions. For a 3 Gb human genome that
   is hundreds of millions of candidate windows *per guide*.
2. **Score how strongly Cas9 would cut at each** — not all mismatches are equal.
   A mismatch in the **seed region** (the ~10 bases nearest the PAM) cripples
   cutting; a mismatch at the distal 5′ end barely matters. This position
   dependence is the single most important fact in off-target prediction.

This project implements the core of step 2 — a **per-window mismatch + CFD
(Cutting Frequency Determination) scorer** — over a whole genome on the GPU, and
aggregates the result into a guide-level specificity score. It is the
GPU-parallel heart of tools like Cas-OFFinder and CRISPOR (§7).

## 2. The math

**Inputs.**
- A guide spacer `g = g₀ g₁ … g₁₉`, each `gₖ ∈ {A,C,G,T}` (20 bases).
- A reference genome `G = G₀ G₁ … G_{L-1}`, `L` bases.

We index positions in the spacer so that **position 19 is PAM-proximal** (the
seed end) and **position 0 is PAM-distal** (the 5′ end).

**Candidate windows.** A window starting at genome position `i` spans 23 bases:
the 20-base protospacer `p = G_i … G_{i+19}` and the 3-base PAM
`G_{i+20} G_{i+21} G_{i+22}`. There are `n = L − 23 + 1` windows.

**PAM gate.** Window `i` is a Cas9 target site only if its PAM is NGG:

```
isPAM(i)  =  (G_{i+21} = G)  ∧  (G_{i+22} = G)         (the first base is "N": unconstrained)
```

**Mismatch count.** For a window that passes the PAM gate,

```
m(i)  =  Σ_{k=0}^{19} [ gₖ ≠ p_k ]            (Iverson bracket; an exact integer in 0..20)
```

**CFD off-target score.** The Cutting Frequency Determination model scores the
predicted cutting at a site as a **product of per-position penalty weights**:

```
CFD(i)  =  Π_{k=0}^{19}  w_k^{[ gₖ ≠ p_k ]}
        =  Π over mismatched positions k of  w(k)          (matches contribute factor 1)
```

where `w(k) ∈ [0,1]` is the weight for a mismatch at position `k`. A **perfect
match** (the on-target) gives `CFD = 1`; each mismatch multiplies in a weight
`< 1`. In the real CFD model `w` also depends on the *identity* of the
rRNA:dDNA mispair; **here we use a synthetic position-only model** (§7) that
captures the essential seed effect:

```
w(k) = W_distal − (W_distal − W_seed) · (1 − d_k²),     d_k = (19 − k)/19,
       W_seed = 0.05  (mismatch at the PAM-proximal seed end)
       W_distal = 0.95 (mismatch at the far 5′ end)
```

So `w(19) = 0.05` (a seed mismatch nearly abolishes cutting) and `w(0) = 0.95`
(a distal mismatch barely dents it) — a smooth, monotone stand-in for the
experimentally-measured curve.

**Guide-level specificity.** Summing CFD over all *off-target* windows (PAM,
≥1 mismatch) gives the off-target burden; the CRISPOR/MIT specificity score is

```
S  =  100 / (100 + 100 · Σ_offtargets CFD(i))             (100 = perfectly specific; → 0 as burden grows)
```

## 3. The algorithm

The serial algorithm is a single sliding-window pass:

```
for i in 0 .. n-1:                 # each of the n = L-23+1 windows
    if not isPAM(i): mark "no site"; continue
    m   = count mismatches g vs G[i..i+19]      # O(20)
    cfd = product of w(k) over mismatched k     # O(20)
    store (m, cfd) for window i
then (host reduction):
    classify windows into on-target (m=0) and off-target (m>=1),
    sum off-target CFD, rank off-targets, compute specificity S.
```

**Complexity.** Each window costs `O(GUIDE_LEN) = O(20) = O(1)`, so the scan is
`Θ(n) = Θ(L)` work — linear in genome length. The windows are **mutually
independent** (each reads only its own 23 bases and writes only its own outputs),
so the *parallel depth* is `O(1)` and the *parallel work* is `Θ(L)`: an
embarrassingly parallel map. The final reduction (sum, rank) is `Θ(n)` work and
is done once, on the host, to keep it deterministic (§5).

**Arithmetic intensity / access pattern.** Each window reads 23 bytes of genome
(heavily overlapping its neighbours) and does ~40 integer compares + ≤20 double
multiplies. It is **memory-light and compute-light** — the genome stays hot in
cache because consecutive windows overlap by 22 of 23 bases. The real cost at
genome scale is simply the sheer *number* of windows, which is exactly what the
GPU parallelizes.

> *Aside — the harder problem this teaches toward.* Production tools first
> **enumerate** candidate windows with a bounded-mismatch search (a BFS over the
> mismatch graph / an FM-index lookup) so they never touch the vast majority of
> windows that can't possibly match. This teaching version scores **every**
> window directly, which is simpler to follow and still perfectly parallel; §7
> describes the enumeration step the catalog mentions.

## 4. The GPU mapping

The map in §3 is the textbook CUDA pattern, the same one as flagship **1.12**
(Tanimoto: one query vs N items): **one thread per genome window.**

- **Thread-to-data mapping.** Thread `t` (global index
  `i = blockIdx.x*blockDim.x + threadIdx.x`) owns window `i`. A **grid-stride
  loop** (`i += blockDim.x*gridDim.x`) lets one modest grid cover an arbitrarily
  long genome.
- **Launch configuration.** `block = 256` threads (a multiple of the 32-lane
  warp; 8 warps to hide latency; good occupancy on sm_75..sm_89). `grid =
  ceil(n/256)`, capped at 1024 blocks with the grid-stride loop covering the
  remainder.
- **Memory hierarchy and why:**
  - **Constant memory** holds the 20-base guide (`__constant__ uint8_t
    c_guide[20]`). Every thread reads all 20 guide bases but none writes them and
    they never change during the launch — the constant cache *broadcasts* one
    address to a whole warp in a single transaction, far cheaper than 20 global
    loads per thread. (Identical trick to the query fingerprint in 1.12.)
  - **Global memory** holds the genome (1 byte/base) and the two output arrays
    (`int mismatches[n]`, `double cfd[n]`, Structure-of-Arrays so writes
    coalesce). Overlapping windows keep the genome resident in L1/L2.
  - **Registers** hold each thread's running mismatch count and CFD product; no
    shared memory or atomics are needed because every window's output is
    independent.
- **No CUDA library is used** for the scoring kernel — the per-window math is a
  short hand-written loop, which is the teaching point. (The catalog also
  mentions cuDNN/transformer inference for *learned* efficiency models; that is
  the part this reduced-scope version deliberately omits — see §7.)

```
        guide (20 B)  ───────────────►  __constant__ c_guide   (broadcast to every warp)

genome  G0 G1 G2 ............................ G_{L-1}   (global memory, 1 byte/base)
        └─ window 0 ─┘
           └─ window 1 ─┘                 windows overlap by 22/23 bases
              └─ window 2 ─┘
                 ...
 thread:  0      1      2   ...            i = block*blockDim + thread  -> window i
 writes:  mismatches[i] (int),  cfd[i] (double)        ── independent, coalesced ──
```

Why this is a *good* GPU problem: hundreds of millions of independent windows,
trivial per-window state, no cross-thread communication. Why the **demo** doesn't
show a speed-up: a 396-window toy genome is dominated by launch + PCIe-copy
overhead (the honest-timing rule, PATTERNS.md §7). The GPU's edge appears at
chromosome scale (10⁸ windows).

## 5. Numerical considerations

- **Mismatch counts are integers** — computed by identical compare-and-add logic
  on both sides, so they are **bit-exact** between CPU and GPU by construction.
- **CFD is a short product of doubles.** We compute it in **FP64**, iterating
  `k = 0..19` in a **fixed order** and multiplying in that order. Crucially the
  inner computation is a pure multiply chain with **no adjacent add**, so there
  is **no fused-multiply-add (FMA) contraction** for the host and device
  compilers to disagree on (FMA divergence is the usual reason GPU and CPU
  doubles differ by ~1 ulp — see flagship 10.02). In practice CPU and GPU
  therefore produce **bit-identical** CFD scores here.
- **The reduction is done on the host, deterministically.** Summing the
  off-target CFD scores in floating point is *order-dependent*. If we summed them
  on the GPU with `atomicAdd`, the nondeterministic accumulation order would make
  the total — and thus the specificity score and the diffed stdout —
  irreproducible (PATTERNS.md §3). So the GPU only does the *independent*
  per-window scoring; the host walks the per-window arrays **left to right** in a
  single thread to sum and rank. Single-threaded fixed-order summation is
  deterministic and matches run-to-run. The top-K ranking breaks score ties by
  lower genome position, so the printed order is fully determined.
- **Why double, not float, for CFD?** A product of up to 20 factors each
  `~0.05..0.95` can underflow a float's precision quickly; FP64 keeps the small
  off-target scores meaningful and removes any precision difference as a
  confounder in verification.

## 6. How we verify correctness

`src/reference_cpu.cpp::scan_cpu()` is an independent serial implementation. The
key design choice (PATTERNS.md §2): **CPU and GPU call the *same*
`__host__ __device__` `score_window()`** from `src/cfd_score.h`, so a
disagreement can only come from a GPU-mechanics bug (indexing, memory, launch),
never from a different formula. `main.cu` then asserts two things:

- **Mismatch arrays are exactly equal** (`mm_diffs == 0`) — integers must match
  to the bit.
- **CFD arrays agree within `1e-12`** — chosen as a tiny *physical* tolerance
  ~10⁴ ulps below the `[0,1]` score range and far below any biological meaning.
  In practice the observed error is **0.0** (see the demo's stderr:
  `CFD max_abs_err = 0.000e+00`), consistent with the no-FMA argument in §5. We
  verify to a stated tolerance rather than *claim* bit-exactness blindly
  (PATTERNS.md §4).

**A stronger, science-level check** is baked into the synthetic data (§"Data"):
the scan must **recover exactly the engineered answer** — one on-target site at
the known position with `CFD = 1`, and off-targets whose ranking reproduces the
seed effect (the 1-mismatch *distal* site outscoring the 1-mismatch *seed* site
by ~17×). That validates the *science*, not just CPU==GPU agreement. Edge cases
covered by the loader: wrong-length guide, non-ACGT guide base, genome too short
for one window, and PAM gating on invalid/masked bases.

## 7. Where this sits in the real world

This is a deliberately **reduced-scope teaching version** (CLAUDE.md §13). A
production off-target pipeline differs in several ways:

- **Enumeration, not brute force.** Cas-OFFinder and FlashFry first *enumerate*
  only the windows within a mismatch budget — Cas-OFFinder via a GPU/CPU bounded
  search (and it also models RNA/DNA **bulges**, not just substitutions);
  FlashFry via a precomputed compressed binary index for fast lookups. They never
  score the overwhelming majority of windows that can't match. We score every
  window because it is simpler to read and still perfectly parallel.
- **The real CFD weights.** The genuine CFD score (Doench et al., *Nat.
  Biotechnol.* 2016) uses an **experimentally measured** weight that depends on
  *both* the position *and* the specific rRNA:dDNA mispair (a 20 × mismatch-type
  table), plus a separate penalty for PAM variants. We use a **synthetic
  position-only curve** so the project redistributes no copyrighted table while
  still teaching the seed effect. To use the real model, replace only
  `cfd_position_weight()` (and pass the mispair identity) in `src/cfd_score.h` —
  the kernel and CPU reference are untouched.
- **On-target efficiency models.** Choosing a *good* guide also needs an
  on-target **efficiency** prediction (Azimuth / Rule Set 2 CNNs; protein
  language models like PLM-CRISPR for Cas9 variants). Those are batched neural
  inferences (cuDNN / transformer kernels) over candidate guides — a whole
  separate GPU workload this project does not implement.
- **Strandedness and genomics plumbing.** Real tools scan both DNA strands
  (reverse-complement), handle alternative PAMs and Cas variants, soft-masked
  repeats, and annotate hits against gene models. We scan one forward strand of a
  toy genome.

The transferable lesson is the **GPU pattern**: a genome-wide per-window scan is
an embarrassingly parallel map (one thread per position, query in constant
memory), and the *order-dependent* aggregation is kept off the GPU to stay
deterministic.

---

## References

- **Doench, Fusi, et al. (2016)**, "Optimized sgRNA design to maximize activity
  and minimize off-target effects of CRISPR-Cas9," *Nat. Biotechnol.* — defines
  the real CFD score and the position/mispair weight table.
- **Cas-OFFinder** — Bae, Park, Kim (2014); <https://github.com/snugel/cas-offinder>
  — GPU-accelerated bounded-mismatch + bulge enumeration; the canonical fast
  off-target searcher. Study its enumeration step (the part we skip).
- **FlashFry** — <https://github.com/aaronmck/FlashFry> — scalable guide design
  with a precomputed binary index; study the index that makes lookups fast.
- **CRISPOR** — <https://crispor.gi.ucsc.edu/> and the paper repo
  <https://github.com/maximilianh/crisporPaper> — an end-to-end on/off-target
  scoring pipeline; the specificity-score formula here follows its MIT-style
  aggregate.
- **PLM-CRISPR** — <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12254127/> —
  protein language model for Cas9-variant activity; the learned-model direction
  the catalog points to (GPU transformer inference).
- **UCSC Genome Browser** — <https://genome.ucsc.edu/> — hg38/mm10 reference
  genomes to scan against (see `scripts/download_data.*`).
