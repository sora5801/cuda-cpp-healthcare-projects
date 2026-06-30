# THEORY — 3.29 Motif Finding in Genomic Sequences

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

A genome is mostly *regulatory* text, not protein-coding genes. **Transcription
factors (TFs)** are proteins that turn genes on and off by binding short, specific
DNA patches — typically **6–20 base pairs** long. The set of sequences a given TF
will bind is its **binding motif**. Crucially, a motif is *not* a fixed word: the
TF tolerates variation, so real binding sites look like `TGACGTCA`, `TGACGTGA`,
`TTACGTCA`, … — the same pattern with substitutions.

How do we learn a TF's motif experimentally? **ChIP-seq** pulls down the DNA a TF
was bound to and sequences it, yielding thousands of ~200 bp "peak" regions, each
containing (somewhere, at an unknown offset) one binding site, surrounded by
biologically irrelevant flanking DNA. **Motif finding** is the inverse problem:
given many such sequences that share a hidden, variable pattern at unknown
positions, *recover the pattern*. The answer — a probabilistic description of the
motif — lets us scan the genome for other binding sites, compare TFs, and
interpret regulatory mutations.

This is hard because (a) the motif is short relative to the sequence (signal
buried in noise), (b) every copy is different (no exact substring to search for),
and (c) the position in each sequence is unknown (a latent variable). MEME's
insight: treat it as a **missing-data maximum-likelihood** problem and solve it
with Expectation-Maximisation.

## 2. The math

**Encoding.** DNA over the alphabet Σ = {A, C, G, T}, encoded A=0, C=1, G=2, T=3.
We are given N sequences; sequence *s* has length Lₛ. The motif width is **W**.

**The model.** Two pieces:

- A **position weight matrix** (PWM) **θ**: a W×4 matrix where
  θ[p][b] = P(base b at motif column p), each row summing to 1. This is the motif.
- A **background** distribution **f**: f[b] = P(base b) for non-motif positions
  (i.i.d.), estimated from the observed base composition.

**Window likelihoods.** A *window* is a length-W substring starting at offset j in
sequence s, with bases x₀,…,x_{W-1}. Its likelihood under the motif vs. the
background is

```
P(window | motif) = ∏_{p=0}^{W-1} θ[p][x_p]
P(window | bg)    = ∏_{p=0}^{W-1} f[x_p]
```

Taking the **log-odds** turns the products into sums (cheaper and stabler):

```
score(window) = Σ_{p=0}^{W-1} log2( θ[p][x_p] / f[x_p] )          ... (★)
```

(★) is the single hot computation of the whole method — `window_score()` in
`src/motif_core.h`. We precompute the W×4 **log-odds table**
`logodds[p][b] = log2(θ[p][b]/f[b])` once per iteration, so each window score is
just W table lookups and adds.

**The OOPS objective.** Under the *One Occurrence Per Sequence* model, each
sequence contains exactly one motif instance at a latent offset Z_s, uniform a
priori over its Mₛ = Lₛ−W+1 windows. The likelihood of sequence s is the average
window likelihood. Working with the score (★), the per-sequence **log evidence**
is the log-sum-exp of its window scores:

```
LL_s = log Σ_{j=0}^{Mₛ-1} exp( score(window_{s,j}) )            (up to a constant)
total LL = Σ_s LL_s
```

EM maximises `total LL` over θ. The **responsibility** of window (s,j) is the
posterior probability that the motif sits there:

```
r_{s,j} = exp(score_{s,j}) / Σ_{j'} exp(score_{s,j'})    (softmax over a sequence's windows)
```

## 3. The algorithm

**MEME OOPS Expectation-Maximisation.** Repeat until `total LL` converges:

1. **Build log-odds** from θ, f  — O(W·4).
2. **E-step:** for every window, compute score (★); per sequence, softmax the
   scores → responsibilities r; accumulate `total LL`.
3. **M-step:** new motif counts `c[p][b] = pseudocount + Σ_{windows} r·[x_p = b]`,
   then renormalise each column → new θ.

Then read off the **consensus** (argmax base per column), the **information
content** `Σ_p (2 + Σ_b θ[p][b]·log2 θ[p][b])` bits, and each sequence's predicted
site (argmax-responsibility window).

**Complexity.** Let L = Σ Lₛ be the total sequence length, so there are
~L windows. One EM iteration costs:

| Step | Work | Notes |
|------|------|-------|
| E-step scoring (★) | **O(L · W)** | the bottleneck; ~L independent dot products |
| E-step softmax | O(L) | cheap, per-sequence |
| M-step counts | O(L · W) | scatter-add into a tiny W×4 table |

Over T iterations: **O(T · L · W)**. The exhaustive-search alternative (test all
4^W candidate words) is O(L · 4^W) — exponential in W and why EM (polynomial) is
used. The E-step scoring is **embarrassingly parallel**: window scores are
mutually independent, read-only over the sequence and the (small) log-odds table.
That is exactly the step we move to the GPU.

## 4. The GPU mapping

We accelerate the E-step scoring; the host keeps the cheap softmax + M-step (the
same division of labour mCUDA-MEME uses). This matches the **"many independent
jobs · constant-memory broadcast"** pattern (PATTERNS.md §1, shared with 1.12
Tanimoto and 12.01 spectral search).

**Data layout.** Sequences are concatenated into one flat byte buffer `data`
(encoded 0–4) with CSR offsets, and the loader precomputes a flat list of all
valid (all-ACGT) windows as absolute start indices `start_of_win[]`. So the
irregular "N ragged sequences" workload becomes **one 1-D array of independent
jobs** — the cleanest possible grid.

**Thread-to-data mapping.** One thread per window:

```
win = blockIdx.x * blockDim.x + threadIdx.x   (then grid-stride: win += blockDim.x*gridDim.x)
start = start_of_win[win]
out[win] = window_score(data, start, W, c_logodds)   // Σ of W log-odds lookups
```

**Launch configuration.** `block = 256` threads (a multiple of the 32-lane warp;
8 warps to hide memory latency; good occupancy on sm_75…sm_89). `grid` = enough
blocks to cover the windows, capped at 1024 — a **grid-stride loop** lets that
fixed grid cover an arbitrarily large window count.

**Memory hierarchy — and why:**

- **Constant memory** holds the W×4 **log-odds table** (`__constant__ c_logodds`).
  Every thread reads the *whole* table but never writes it, and it is identical
  for the launch → the constant cache **broadcasts** one address to a full warp in
  a single transaction. The table is tiny (≤ MAX_W·4 floats = 1 KiB) and
  fixed-size — a textbook constant-memory use, just like the query in 1.12.
- **Global memory** holds `data` and `start_of_win`. Adjacent threads read
  adjacent `start_of_win` entries (coalesced). The W sequence-byte reads per
  thread overlap heavily between neighbouring windows and are served from L2.
- **Registers** hold the running sum. No shared memory or atomics are needed:
  every output is independent.

This kernel is **memory-light and arithmetic-trivial** (W adds), so *occupancy and
launch overhead*, not compute, set its throughput — which is exactly why it is
launch-bound on the tiny sample and only wins at scale (§7, honest-timing rule).

```
 concatenated sequences (global)        constant log-odds table (broadcast)
 [ ...seq s bytes... ]                   logodds[W][4]
        ^                                     |
        | start_of_win[win]                   v
   win 0  win 1  win 2 ...            +-----------------------------+
   +----+ +----+ +----+   ...         | thread win: Σ_{p<W} logodds |
   |t0  | |t1  | |t2  |  one thread    |          [p][ data[start+p]]|
   +----+ +----+ +----+  per window -> +-----------------------------+ -> out[win]
   \__________ 1-D grid of independent jobs (grid-stride) __________/
```

> The catalog also mentions *Thrust for top-k* and *warp-level sums*. We
> deliberately keep the kernel to the pure scoring step (no Thrust dependency):
> the per-sequence "best site" is a tiny argmax done on the host after the scores
> return, and W is small enough that a per-thread serial sum beats a warp
> reduction (a warp reduction shines when each *score* is itself a long vector;
> here each score is a short W-term sum owned by one thread). Both are noted as
> exercises, not black boxes.

## 5. Numerical considerations

- **Precision.** The log-odds table and per-window scores are **FP32** — the same
  type on both sides — because (★) is a short sum (W≤64) of small magnitudes;
  double precision buys nothing observable here and would only slow the GPU. The
  EM *bookkeeping* (softmax, log-sum-exp, count accumulation) runs in **FP64** on
  the host for a stable likelihood, but that is outside the verified kernel.
- **Stability.** The per-sequence softmax subtracts the max score before `exp`
  (log-sum-exp trick) so no term overflows. PWM columns are kept strictly positive
  by the M-step pseudocount (0.25), so `log2(θ/f)` is always finite — no `-inf`.
- **Determinism / races.** The kernel writes each `out[win]` from exactly one
  thread — **no atomics, no shared accumulators, no reordered reductions**. Float
  addition is non-associative, but `window_score()` sums `p = 0…W-1` in a *fixed*
  order on both CPU and GPU, so the bits are identical. The M-step *does* sum many
  `r` values, but on the host in a fixed loop order — also deterministic. Result:
  stdout is byte-identical every run (PATTERNS.md §3).

## 6. How we verify correctness

The CPU reference (`src/reference_cpu.cpp::score_windows_cpu`) and the GPU kernel
call the **identical** `__host__ __device__ window_score()` from `motif_core.h`,
with the same fixed summation order. So for the final converged model their
per-window scores are **bit-for-bit equal**, and `main.cu` verifies with an
**exact** tolerance (`max_abs_err == 0`) — the strongest honest claim (PATTERNS.md
§4: exact when the same operations run on both sides). A nonzero `max_abs_err`
would mean a real bug (a layout error, an indexing slip, a divergent code path),
not floating-point noise.

A second, *scientific* check beyond CPU==GPU: the demo plants a known motif
(`TGACGTCA`) and reports the recovered consensus + sites. EM recovers the motif's
core (`TGACGT`) at a consistent 2 bp-shifted register across all 12 sequences,
with ~8.3 bits of information content — confirming the method finds the real
signal, not noise. The 2 bp **phase shift** is a known EM behaviour (the
likelihood has near-equal optima at neighbouring registers), reported honestly and
left as a refinement exercise.

Edge cases handled by the loader: sequences shorter than W (rejected), non-ACGT
characters (encoded as 4 and their windows excluded), and CRLF line endings.

## 7. Where this sits in the real world

This is a **reduced-scope teaching version**. Production motif finders add:

- **Occurrence models beyond OOPS:** ZOOPS (zero-or-one per sequence) and TCM
  (any number per sequence) — MEME fits all three and picks the best.
- **Automatic width selection:** MEME searches a range of W and scores models by
  an **E-value** (statistical significance against a random model), not just
  likelihood. We fix W=8 and report information content.
- **Multiple motifs:** after finding one, MEME *erases* its sites and re-runs to
  find the next.
- **Both strands:** TF sites occur on either strand; real tools also score reverse
  complements.
- **Scale & the GPU:** **mCUDA-MEME** distributes MEME's EM across GPU cores *and*
  multiple GPUs/nodes, turning multi-day genome-scale runs into hours — our
  single-E-step kernel is the didactic kernel of that idea. **Argo_CUDA** takes the
  opposite tack (exhaustive GPU enumeration). **FIMO** (also MEME Suite) does the
  *scan* problem — score a *known* PWM across a genome — which is essentially our
  E-step kernel run once. **HOMER** does enrichment-based discovery on ChIP-seq.
  **JASPAR** is the curated PWM database you would validate a recovered motif
  against.

The honest takeaway: the math here (PWM log-odds + EM) *is* the math of the real
tools; what production adds is statistical rigour, model variety, and the
distributed-GPU engineering to run it on millions of peaks.

---

## References

- Bailey & Elkan (1994), *Fitting a mixture model by expectation maximization to
  discover motifs in biopolymers* — the original MEME paper; the OOPS/ZOOPS/TCM
  models and the EM derivation implemented here.
- **MEME Suite** <https://meme-suite.org/> — reference implementation, FIMO,
  E-value statistics; the canonical "what production does."
- **CUDA-MEME / mCUDA-MEME** <https://cuda-meme.sourceforge.io/homepage.htm> — how
  MEME's EM is mapped onto GPUs and GPU clusters (the scaled-up version of §4).
- **Argo_CUDA** <https://pubmed.ncbi.nlm.nih.gov/29281953/> — an alternative,
  exhaustive GPU motif-discovery strategy.
- **JASPAR 2024** <https://jaspar.elixir.no/> — curated PWMs; validate recovered
  motifs against known TF profiles.
- **HOMER** <http://homer.ucsd.edu/> — ChIP-seq motif enrichment; complementary
  enrichment-based discovery.
