# THEORY — 6.13 Gene Regulatory Network Inference

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

Every cell carries the same DNA, yet a neuron and a muscle cell behave nothing
alike. The difference is **which genes are expressed** and how strongly — and that
is orchestrated by a **gene regulatory network (GRN)**. Some genes encode
**transcription factors (TFs)**: proteins that bind DNA and turn other genes up or
down. A GRN is the directed graph of those control relationships: an arrow
`TF → gene` means "this factor regulates that target".

Reconstructing the GRN from data is a foundational problem in systems biology: it
tells us how cells make decisions, how disease dysregulates them, and where a drug
might intervene. The raw material is an **expression matrix**: rows are genes,
columns are samples (increasingly *single cells*, via scRNA-seq), and each entry
is how much of that gene's mRNA was measured. If two genes' expression levels move
together across many cells — beyond what chance would produce — they are probably
*coupled* in the network. The inference challenge is (a) measuring "move together"
in a way that catches **nonlinear** relationships, and (b) telling **direct**
regulation apart from **indirect** knock-on correlation.

**ARACNE** (Algorithm for the Reconstruction of Accurate Cellular Networks) answers
both. It measures coupling with **mutual information** — a distribution-free
dependence score that catches nonlinearities a correlation coefficient misses —
and it removes indirect edges using an information-theoretic identity, the **Data
Processing Inequality**. This project implements exactly that, on the GPU.

## 2. The math

**Setup.** Let the expression matrix be `X ∈ ℝ^{G×S}`: `G` genes, `S` samples,
`X[g,s]` the expression of gene `g` in sample `s` (row-major, `X[g,s] = expr[g·S+s]`).

**Discretization.** MI on continuous data needs a density estimate; the simplest
is a histogram. For each gene `g` we find its range `[lo_g, hi_g]` over the `S`
samples and map each value to one of `B` equal-width bins:

```
    bin(x) = clamp( floor( (x − lo)/(hi − lo) · B ), 0, B−1 )
```

producing a discretized matrix `D ∈ {0,…,B−1}^{G×S}` (`B = 8` here).

**Mutual information.** For a gene pair `(i,j)`, count the `B×B` **joint
histogram** `n_{ab}` = number of samples with gene `i` in bin `a` and gene `j` in
bin `b`. Its **marginals** are the row/column sums `r_a = Σ_b n_{ab}` and
`c_b = Σ_a n_{ab}`, and `Σ_a r_a = Σ_b c_b = S`. The plug-in (maximum-likelihood)
MI estimate, in **nats** (natural-log units), is

```
    Î(i;j) = Σ_{a,b}  p_{ab} · ln( p_{ab} / (p_a · p_b) ),
             where  p_{ab} = n_{ab}/S,  p_a = r_a/S,  p_b = c_b/S.
```

Rearranged to keep everything integer until the log (what the code computes):

```
    Î(i;j) = (1/S) · Σ_{a,b : n_{ab}>0}  n_{ab} · ln( (n_{ab}·S) / (r_a·c_b) ).
```

`Î ≥ 0`, and `Î = 0` **iff** the two genes are independent. Larger = stronger
statistical coupling. Empty cells contribute 0 (the limit `x ln x → 0` as `x→0`),
so they are skipped — which also dodges `ln 0`.

**Data Processing Inequality (DPI).** The key identity: if the dependence between
`A` and `B` is entirely *mediated* by a third gene `C` (a Markov chain
`A → C → B`), then information cannot increase along the chain:

```
    I(A;B) ≤ min( I(A;C), I(C;B) ).
```

So in any triangle of edges, the **weakest** one is the prime suspect for being
indirect. ARACNE examines every triangle `(i,j,k)` and removes edge `(i,j)` if

```
    Î(i,j) < Î(i,k) − τ   AND   Î(i,j) < Î(j,k) − τ
```

for a small slack `τ` (so near-ties are not pruned). An MI floor `θ` first discards
edges too weak to be more than noise. Outputs: the `G×G` MI matrix and a `keep`
mask of the surviving **direct** edges.

## 3. The algorithm

```
1. discretize        each gene -> B bins                    O(G·S)
2. MI matrix         for each pair (i<j):
                        build B×B joint histogram            O(S)   per pair
                        Î from marginals                     O(B²)  per pair
                     over all G(G−1)/2 pairs                 O(G²·S + G²·B²)
3. DPI prune         for each edge (i<j), scan mediators k   O(G³)
```

The dominant term is the **MI matrix**: `O(G²·S)`. It is *embarrassingly parallel*
— each pair's histogram is independent, needs only two rows of `D`, and writes only
its own matrix cell. That independence is exactly what makes the GPU mapping
trivial and correct. The DPI step is `O(G³)` but reads a small, finished matrix.

**Serial vs. parallel.** The serial cost is `O(G²·S)`; the parallel *work* is the
same but the *depth* (critical path) is `O(S)` — one thread's histogram scan —
because all `G(G−1)/2` pairs run at once. With `G = 20 000`, that is ~200M
independent jobs, the catalog's headline number.

## 4. The GPU mapping

Three kernels, in `src/kernels.cu`:

**(a) `discretize_kernel` — one thread per gene.** Thread `g` scans its row for
`[lo,hi]` and bins it, calling the shared `discretize_value()` so the result is
identical to the CPU's. Reads/writes only gene `g`'s contiguous row → coalesced,
no sharing.

**(b) `mi_kernel` — one thread per gene pair.** We enumerate unordered pairs
`(i<j)` as a flat index `p ∈ [0, G(G−1)/2)` and map `p → (i,j)` by walking the
triangular row offsets (an integer scan — deliberately *not* a floating `sqrt`,
which could misround and break determinism; `G` is small so the scan is cheap).
Each thread keeps a **private** `B×B = 64`-int histogram in local memory, tallies
one increment per sample, and calls `mi_from_joint()`. It writes `mi[i,j]` and
`mi[j,i]` — cells no other thread touches, so **no atomics, no races**. A
grid-stride loop lets a capped grid cover any pair count.

**(c) `dpi_kernel` — one thread per candidate edge.** Reads the *finished,
unmutated* MI matrix and, for edge `(i,j)`, scans all mediators `k`, applying the
DPI test. Because it reads a frozen matrix, the outcome is order-independent →
matches the CPU exactly.

```
   pairs p = 0..G(G-1)/2-1        (flatten the strict upper triangle)
   ┌──────────────────────────────────────────────┐
   │ block 0        block 1        block 2   ...    │  256 threads each
   └──┬──────────┬──┬──────────┬──┬──────────┬──────┘
      ▼          ▼  ▼          ▼  ▼          ▼
   thread→pair (i,j): private 8×8 histogram → mi_from_joint → mi[i,j]=mi[j,i]
                          (grid-stride if pairs > total threads)
```

**Launch config.** 256 threads/block (multiple of the 32-lane warp; 8 warps to
hide latency; good occupancy on sm_75–sm_89). Blocks = `ceil(n_pairs/256)`, capped
at 1024 with the grid-stride loop covering the rest.

**Memory.** `D` is small and read-only, shared by all threads → it rides the L2 /
read-only cache well. Each histogram lives in per-thread local memory
(`64·4 = 256 B`); at `B=8` this stays in registers/L1 and is the reason `B` is a
compile-time constant. No shared memory is needed at this size — a *tiling* variant
that stages `D` rows in shared memory is Exercise 4 and matches the catalog's
"one-tile-per-gene-pair block" note.

**No black boxes (CLAUDE.md §6.1.6).** The catalog suggests *cuBLAS* for a pairwise
**correlation** matrix (`C = XXᵀ`, an `N×N` outer product / GEMM) and *Thrust* for
per-gene ranks. Those accelerate a *linear* (correlation) or *rank-MI* variant. We
deliberately hand-roll the **histogram-MI** path instead, because (i) it captures
nonlinear dependence that correlation misses, and (ii) integer histograms keep the
math exactly reproducible and CPU↔GPU bit-comparable — the whole verification
story. If you wanted the dense-correlation route at genome scale, `cublasDsyrk`
computing `XXᵀ` would be the tool (one call, tuned GEMM), and hand-rolling it means
a tiled shared-memory matrix multiply (see flagship `3.11`).

## 5. Numerical considerations

- **Precision.** Histogram counts are **exact integers**. The only floating-point
  work is the `Σ n·ln(·)` accumulation, done in **FP64** (`double`) on both sides.
- **Determinism.** Integer addition is associative and commutative, so the joint
  histogram is identical regardless of thread/sample order — no floating-point
  reduction reordering anywhere (contrast the atomic-float hazard in PATTERNS.md
  §3). Each thread owns disjoint outputs, so **no atomics** are needed. stdout is
  therefore byte-identical every run.
- **The single transcendental.** `logval()` in `grn.h` wraps `::log`, which under
  nvcc resolves to CUDA's device `log()` (device code) or the CRT `log()` (host
  code); both are IEEE-754 correctly-rounded double log, agreeing to ≈1 ULP. That
  ~`1e-16` disagreement, summed over `≤ B²` terms, is why we verify to `1e-9`
  rather than claiming bit-identity (PATTERNS.md §4).
- **Estimator bias.** The plug-in MI is biased *upward* on finite samples (empty
  cells vanish but sparse ones inflate). We manage it with a coarse `B` and an MI
  floor; the Miller–Madow correction is Exercise 2.
- **Edge cases.** A constant gene (`hi == lo`) maps to a single bin → `Î = 0` with
  every partner (handled in `discretize_value`); the all-zero-sample guard in
  `mi_from_joint` returns 0.

## 6. How we verify correctness

Two independent checks, both in `src/main.cu`:

1. **Continuous:** `max |mi_cpu − mi_gpu|` over all `G²` cells must be `≤ 1e-9`.
   Because the two paths share `grn.h`'s `mi_from_joint()` and build identical
   integer histograms, they differ only in the last bits of `log`; the observed
   error is ~`2.2e-16` (machine epsilon), far inside the tolerance.
2. **Discrete:** the DPI `keep` masks must be **bit-identical** (`==`). This is an
   exact check — the pruned network the GPU reports must be the *same set of edges*
   the trusted serial code reports.

Why convincing: the CPU reference is written to be *obviously* correct (plain
nested loops, no parallelism) and is a completely separate implementation of the
control flow. When an independent serial code and the parallel GPU code agree — to
machine precision on a continuous quantity *and* exactly on a discrete one — a
shared bug would have to corrupt both identically, which is vanishingly unlikely.

A **second, scientific** check (PATTERNS.md §4): the synthetic data has a *planted*
network, so we do not merely confirm CPU==GPU — we confirm the algorithm recovers
the **right biology**. The demo shows raw MI proposing 7 edges (including the
indirect `TF–B`, `A–C`, `B–C`) and DPI pruning to exactly the 4 true direct edges
`TF–A, TF–C, D–E, A–B`. Noise genes stay below the floor (their max MI ≈ 0.11 nats
vs. the 0.20 threshold).

## 7. Where this sits in the real world

This is a **reduced-scope teaching version**: undirected, histogram-MI ARACNE on a
clean synthetic matrix. Production GRN inference goes further:

- **GENIE3** casts inference as regression — for each target gene, fit a random
  forest predicting it from all other genes and read off feature importances. It
  often tops the **BEELINE** benchmark; GPU random forests are the accelerator.
- **PANDA** message-passing fuses expression with TF-motif and protein-interaction
  priors, iterating to a consensus network.
- **Neural-ODE** methods (`torchdiffeq`) fit `dx/dt = f_θ(x)` to *time-series*
  expression, inferring dynamics and hence directed, signed edges — GPU autograd
  through an ODE solver.
- **scVI** learns a probabilistic latent representation of scRNA-seq (a VAE) that
  denoises and batch-corrects before any network step; `rapids-singlecell` runs the
  Scanpy pipeline on GPU.

Real data adds the hard parts we skip: single-cell **dropout / zero-inflation**,
batch effects, cell-type mixing, and — worst — **no ground-truth network**, so
methods are ranked on curated benchmarks (BEELINE) or against ChIP-seq TF-binding
evidence (ENCODE). ARACNE-style MI remains a strong, interpretable baseline and the
clearest lens on *why* indirect-edge removal matters.

---

## References

- Margolin et al., *ARACNE: An Algorithm for the Reconstruction of Gene Regulatory
  Networks* (BMC Bioinformatics, 2006) — the MI + DPI method implemented here.
- Cover & Thomas, *Elements of Information Theory* — mutual information and the Data
  Processing Inequality, from first principles.
- **BEELINE** — <https://github.com/Murali-group/BEELINE> — how GRN methods are
  benchmarked against reference networks; the fair way to compare inference methods.
- **GENIE3** (Huynh-Thu et al., 2010) — the random-forest regression alternative;
  study it to see a completely different framing of the same problem.
- **scVI / scverse** — <https://github.com/scverse/scvi-tools> — deep generative
  modeling of scRNA-seq; the modern preprocessing layer before network inference.
- **torchdiffeq** — <https://github.com/rtqichen/torchdiffeq> — GPU neural ODEs for
  inferring *dynamics* (directed edges) from time-series expression.
- **Scanpy / rapids-singlecell** — <https://github.com/scverse/scanpy> — the
  standard scRNA-seq analysis toolkit and its GPU backend.
