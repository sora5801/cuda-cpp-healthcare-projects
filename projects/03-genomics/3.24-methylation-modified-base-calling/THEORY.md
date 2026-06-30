# THEORY — 3.24 Methylation / Modified-Base Calling

> The deep dive. Read [README.md](README.md) first for the one-paragraph picture,
> then this for the science → math → algorithm → GPU mapping → numerics →
> verification → real-world chain. Code references: the shared physics is in
> [`src/meth_core.h`](src/meth_core.h), the CPU baseline in
> [`src/reference_cpu.cpp`](src/reference_cpu.cpp), the GPU twin in
> [`src/kernels.cu`](src/kernels.cu).

---

## 1. The science

### DNA methylation, biologically

DNA is more than its four bases. The most common **epigenetic** mark in mammals is
**5-methylcytosine (5mC)**: a methyl group (`–CH₃`) added to the 5-carbon of a
cytosine, almost always in a **CpG** context (a cytosine immediately followed by a
guanine). CpG methylation silences genes, defines cell identity, and goes awry in
cancer — yet it does not change the A/C/G/T sequence, so ordinary sequencing is
blind to it. Reading it is its own measurement problem.

### Nanopore sequencing, physically

An Oxford Nanopore device pulls a single DNA strand through a nanometer-scale
protein pore embedded in a membrane, with a voltage across it. Ions flowing through
the pore make a tiny **ionic current** (picoamps). At any instant the current is
throttled by the **~k bases** physically occupying the pore's narrowest
constriction — a **k-mer**. As the strand ratchets base by base, the current steps
through a sequence of levels; segmentation collapses each dwell into one **event**
(a mean current + duration).

A **pore model** is a calibration table: for each of the `4^k` k-mers it stores the
**expected current** as a Gaussian (`level_mean`, `level_stdv`). A methylated
cytosine is bulkier and sits slightly differently, so the k-mers that *contain* the
5mC produce a **shifted** current. Keep two pore models — canonical and methylated —
and methylation calling becomes a hypothesis test: *which model better explains the
observed events over this site?*

### Where the GPU comes in

A 30× human methylome has ≈28 million CpG sites, each covered by ≈30 reads; scoring
each read under two models is ≈1.7 **billion** small alignments over billions of
signal samples. The alignments are mutually independent — the textbook case for
data-parallel hardware (this is why f5c exists).

---

## 2. The math

### Emission: how well does an event fit a k-mer?

If event `i` has mean current `xᵢ` and aligns to a reference k-mer with model
Gaussian `N(μ, σ)`, its **emission log-probability** is

```
log p(xᵢ | k-mer) = −½·((xᵢ − μ)/σ)²  −  log σ  −  ½·log(2π)
```

This is [`gaussian_logpdf`](src/meth_core.h) in the code. Higher (less negative) =
better fit. Working in **log space** turns the products of a path's probabilities
into sums, which is numerically stable and lets us use `max` instead of repeated
multiplication.

### Alignment: threading events onto k-mers

A read does not give us a clean one-event-per-k-mer correspondence: the pore can
**dwell** (extra events on one k-mer) or **skip** (a base passes without a clear
event). We model the correspondence as a path through a grid and score the best one.
Let `E` = number of events, `K` = number of reference k-mers. Define

```
dp[i][j] = log-likelihood of the best alignment of the first i events
           to the first j k-mers, with event (i−1) emitted by k-mer (j−1).
```

The recurrence takes the best of three moves into cell `(i,j)`:

```
dp[i][j] = max(
    dp[i−1][j−1] + emit(i,j),                 # MATCH: consume event, advance k-mer
    dp[i−1][j]   + emit(i,j) + STAY_PENALTY,   # STAY : extra event, same k-mer
    dp[i][j−1]              + SKIP_PENALTY     # SKIP : advance k-mer, no event
)
```

where `emit(i,j) = log p(xᵢ₋₁ | k-mer_{j−1})`. This is **Viterbi** (a max-product
HMM decode in log space): the single most-likely alignment, not the marginal. The
final score is `dp[E][K]`.

### The band

Events track the reference roughly 1:1, so the alignment never wanders far from the
diagonal. We forbid cells with `|i − j| > BAND_WIDTH` (set to `−∞`). This **band**
turns an `O(E·K)` full table into `O(E·band)` work and reflects the biology. f5c
uses an **adaptive** band that re-centers on the best cell each row; we use the
simplest correct version — a **fixed** band — and explain the upgrade in §7.

### The decision: log-likelihood ratio

For one (read, site) job, run the DP twice — once with each pore model — and take

```
LLR = logL_meth − logL_canon.
```

`LLR > 0` ⇒ the events favor the methylated hypothesis. Per site, average the LLRs
over its reads; **call 5mC iff mean LLR > 0**. (A real caller would calibrate a
threshold and report a probability; mean-LLR-sign is the transparent teaching
version.)

---

## 3. The algorithm (and complexity)

```
for each (read, site) job j:                       # ~N_sites · coverage jobs
    derive K k-mer codes from the reference window  # O(K)
    logL_meth  = banded_dp(events, kmers, meth_model)    # O(E · band)
    logL_canon = banded_dp(events, kmers, canon_model)   # O(E · band)
    llr[j] = logL_meth − logL_canon
for each site s:                                   # aggregate
    mean_llr[s] = mean over reads of llr;  call[s] = (mean_llr[s] > 0)
```

- **Per job:** two banded DPs, each `O(E·(2·BAND_WIDTH+1))`. With `E=K=10` and
  `BAND_WIDTH=6` that is a few hundred FMAs — tiny.
- **Total:** `O(jobs · E · band)`. Serial CPU cost grows linearly in jobs; the GPU
  does all jobs concurrently, so wall-time is ~constant until the device saturates.
- **Memory per job:** two DP rows of `K+1` doubles (the recurrence reaches back one
  row only), i.e. a handful of registers — no per-job heap allocation.

---

## 4. The GPU mapping

| Decision | Choice | Why |
|---|---|---|
| **Parallel unit** | one **thread per (read, site) job** | jobs are independent; this is the `1.12`/`12.01` batched-jobs pattern (PATTERNS.md §1) |
| **Block size** | 128 threads | good occupancy on sm_75–sm_89 for a small, register-heavy DP |
| **Grid** | `ceil(num_jobs / 128)` blocks | covers all jobs; a guard `if (j >= num_jobs) return;` handles the ragged last block |
| **Pore models** | `__constant__` memory | every thread reads the same table, never writes it → the constant cache broadcasts each entry warp-wide instead of 32 separate global loads |
| **DP scratch** | two on-stack rows of `double` | tiny (`K+1` doubles); lives in registers/local memory, so **no shared memory** is needed at this size |
| **Reduction** | **none** | each thread writes one independent `llr[j]`; no atomics, no cross-thread sync |

Thread-to-data mapping: thread `(blockIdx.x, threadIdx.x)` owns job
`j = blockIdx.x*blockDim.x + threadIdx.x`. It derives the window's k-mer codes
(`kmer_code`), runs `banded_align_core` twice (reading `c_meth` / `c_canon` from
constant memory), and writes `llr[j]`. See [`src/kernels.cu`](src/kernels.cu).

Because there is no inter-thread communication and no floating-point reduction
order to worry about, the kernel is **deterministic**: the same input yields the
same `llr` every run (PATTERNS.md §3).

```
        constant memory                 global memory
     ┌──────────────────┐          ┌───────────────────────┐
     │ c_canon[64]      │          │ jobs[num_jobs]  (in)   │
     │ c_meth [64]      │   read   │ llr [num_jobs]  (out)  │
     └────────┬─────────┘  ──────▶ └───────────┬───────────┘
              │  broadcast                      │ 1 thread : 1 job
              ▼                                 ▼
   thread j : kmer_code ▸ banded_dp(meth) − banded_dp(canon) ▸ llr[j]
```

---

## 5. Numerical considerations

- **Double-precision DP.** Emission/transition sums accumulate in `double`. The
  inputs (events, model means) are `float`; doing the arithmetic in `double` keeps
  the CPU and GPU within FMA-rounding of each other and avoids catastrophic
  cancellation in `logL_meth − logL_canon` (two ~tens-of-units logs).
- **`−∞` sentinel.** Unreachable cells (outside the band, or invalid boundary) are
  a large finite negative (`−1e18`), not IEEE `−inf`, so a stray `(−inf)+(finite)`
  never produces a `NaN` that would poison the `max`.
- **Shared `__host__ __device__` core.** The exact same `banded_align_core` and
  `gaussian_logpdf` run on both sides (`MC_HD` expands to `__host__ __device__`
  under nvcc, to nothing under the host compiler). The only divergence is the GPU's
  fused multiply-add contracting `a*b + c` slightly differently from the host —
  here that is `0` to the printed precision (measured `max_abs_err = 0.0`).
- **Determinism.** No atomics, no reduction, no RNG on the device → byte-identical
  `stdout` every run. Timings (which vary) are printed to `stderr` and not diffed.

---

## 6. How we verify correctness

Three independent checks:

1. **GPU vs CPU (the gate).** `main.cu` computes every per-job LLR on both the CPU
   reference and the GPU and asserts `max_abs_err(llr_cpu, llr_gpu) ≤ 1.0e-3`
   (PATTERNS.md §4: a small physical tolerance for an FMA-sensitive computation;
   the actual error is `0.0` here). Both call the same shared core, so this is a
   true CPU/GPU consistency check, not a tautology — a bug in the kernel's k-mer
   indexing or constant-memory upload would break it.
2. **Recovery of planted truth (the science).** The synthetic instance plants a
   known 5mC/canonical label per site. The program reports
   `calls matching ground truth: 12 of 12` — a correct DP + LLR recovers what we
   built, validating the *modeling*, not just CPU==GPU agreement.
3. **Deterministic stdout.** `demo/run_demo` diffs stdout against
   `demo/expected_output.txt`; any nondeterminism would fail the demo.

Edge cases handled: the ragged last block (thread guard), malformed input
(`load_meth_data` throws), `σ ≤ 0` rejected at load, and the band boundary rows
initialized explicitly.

---

## 7. Where this sits in the real world

This is a **reduced-scope teaching version** (CLAUDE.md §13). How production tools
differ:

- **k-mer length.** Real ONT R10.4.1 models are **9-mers** (262 144 entries, ~2 MB)
  — too big for constant memory. Production f5c keeps the model in global/texture
  memory. We use **3-mers** (64 entries) so the table is human-readable and fits
  constant memory; nothing else in the algorithm changes.
- **Adaptive band.** f5c's band **re-centers** each row on the running maximum
  (the "adaptive banded event alignment" of nanopolish/f5c), which tracks indels in
  long reads. We use a **fixed** band — correct for our clean, centered windows, and
  the cleanest starting point. Exercise 1 upgrades it.
- **Segmentation & basecalling.** We assume known reference windows and a 1:1
  event-to-k-mer layout. Real pipelines basecall the raw signal first (CTC over a
  modification-aware alphabet) and segment variable-length events; f5c re-aligns
  events to the basecalled reference before scoring.
- **Learned models.** Remora/Dorado replace the fixed Gaussian pore model with a
  trained **CNN/LSTM** that classifies signal windows directly, then call
  modifications during basecalling on the GPU (cuDNN). The likelihood-ratio idea
  here is the classical predecessor and the clearest way to *understand* the signal.
- **Statistics.** Production callers calibrate a probability per call, model
  strand/allele effects (binomial tests for allele-specific methylation), and
  aggregate into bedMethyl (Modkit). We stop at mean-LLR-sign.
- **Comparison ground truth.** ENCODE **WGBS** (whole-genome bisulfite sequencing)
  is the orthogonal reference methylation calls are benchmarked against.

The takeaway the learner should carry away: **a modified base is detectable because
it shifts a measurable physical signal, and "is it modified?" is a likelihood-ratio
between two generative models of that signal — a question that factorizes into
millions of independent GPU jobs.**
