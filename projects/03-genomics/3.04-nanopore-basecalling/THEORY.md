# THEORY — 3.4 Nanopore Basecalling (CTC greedy decode)

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

> **Scope.** Full basecalling = neural network (squiggle → posteriors) + CTC
> decoder (posteriors → bases). This project implements **the CTC decoder**; the
> network is described in §7 but not built (CLAUDE.md §13). Everything below
> distinguishes the two clearly.

---

## 1. The science

**How a nanopore sequences DNA.** A nanopore sequencer (Oxford Nanopore: MinION,
GridION, PromethION) threads a single strand of DNA through a nanometer-scale
protein pore embedded in a membrane, across which a voltage is applied. Ions flow
through the pore and produce a measurable **ionic current** (picoamps). As each
short stretch of bases (a *k-mer*) occupies the pore's narrowest point, it partially
blocks the current by a characteristic amount. A motor protein ratchets the DNA
through one base at a time; the current is sampled at a fixed rate (e.g. 4–5 kHz).
The result is a 1-D time series — the **"squiggle"** — whose shape encodes the
base sequence.

**The problem: squiggle → sequence ("basecalling").** Recovering the bases is hard
because (a) the current depends on a *window* of bases, not one, (b) the DNA moves
at a variable speed, so each base dwells in the pore for a *variable, unknown number*
of samples, and (c) the signal is noisy. Modern callers (ONT's Dorado, formerly
Guppy) solve this with a **neural network** that reads the squiggle and emits, at
each time step, a probability distribution over base classes — and then a **CTC
decoder** turns that distribution sequence into the called bases.

**Why basecalling matters.** It is the very first computational step of every
nanopore experiment; its accuracy bounds everything downstream (assembly, variant
calling, methylation). A run produces *millions* of reads per hour, so basecalling
must keep up with the sequencer in real time — which is why it runs on the GPU.

**What this project models.** The *decode* stage. We assume the network has already
run and given us, for each read, a `T × C` posterior matrix; we recover the bases.
This isolates a clean, fully-understandable, embarrassingly-parallel algorithm.

## 2. The math

**The CTC label space.** Connectionist Temporal Classification (Graves et al., 2006)
solves the "unknown alignment between a long input and a short output" problem. Let
the output alphabet be the 4 DNA bases `{A, C, G, T}`. CTC augments it with a special
**blank** symbol `∅`, giving `C = 5` classes (in our code: index `0 = blank`,
`1 = A`, `2 = C`, `3 = G`, `4 = T`).

The network emits a matrix **P** of shape `T × C`, where `T` is the number of time
steps for this read and

  `P[t, c] = probability that class c is emitted at step t`,  with `Σ_c P[t, c] = 1`.

A **path** `π = (π₀, π₁, …, π_{T−1})` assigns one class to each time step. CTC defines
a deterministic **collapse map** `B` from a path to a label sequence:

1. **Merge** maximal runs of the *same* class into one symbol.
2. **Delete** all blanks.

Example (`∅` = blank):  `B(∅ A A ∅ A C C) = B-merge(∅ A ∅ A C) = (A A C)`. Note the
two `A`s survive because a blank separates the runs — this is precisely how CTC
represents a **homopolymer** (a repeated base) and the **dwell** of one base across
many steps.

**The decoding objective.** The "best" label sequence is
`ŷ = argmax_y P(y | P) = argmax_y Σ_{π : B(π)=y} Π_t P[t, π_t]` — summing over *all*
paths that collapse to `y`. Computing this exactly needs a dynamic program (the CTC
forward algorithm) or a beam search.

**Greedy ("best-path") decoding — what we implement.** Instead of summing over paths,
take the single most-likely path and collapse it:

  `π̂_t = argmax_c P[t, c]`   (independently per step),  then  `ŷ = B(π̂)`.

This is the maximum-*path* (not maximum-*label*) decode. It is exact when one path
dominates the posterior mass — common when the network is confident — and is the
"fast" decode used in practice for high-quality models. Our synthetic data is built
so one path clearly dominates, making greedy exact.

## 3. The algorithm

**Greedy CTC decode of one read** (`ctc_greedy_decode` in `src/ctc_core.h`):

```
input  : P[0..T-1][0..C-1]   (this read's posteriors)
output : bases[]             (the called DNA sequence)
prev = -1                    # argmax class of the previous step (none yet)
for t in 0 .. T-1:
    cls = argmax_c P[t][c]                 # step 1: best class this step
    if cls != prev and cls != BLANK:       # step 2: collapse on the fly
        emit base(cls)                      #         (change to a non-blank)
    prev = cls
return bases
```

The collapse is done **incrementally in a single pass**: by tracking only the
previous step's argmax class we both merge repeats (skip if `cls == prev`) and drop
blanks (skip if `cls == BLANK`). A base is emitted exactly when the argmax *changes
to a non-blank value*. Setting `prev = cls` even for blanks is what lets `A ∅ A`
emit two `A`s (the blank resets `prev`).

**Complexity.**

| | Per read | Whole batch (`N` reads) |
|---|---|---|
| Serial work | `O(T · C)` (argmax) + `O(T)` (collapse) = `O(T·C)` | `O(Σ_r T_r · C)` |
| Parallel depth | `O(T)` (the collapse is a sequential scan) | `O(max_r T_r)` |

`C = 5` is a small constant, so each read is essentially `O(T)`. The collapse is a
left-to-right scan with a loop-carried dependence (`prev`), so a single read is
inherently sequential — but the **reads are independent**, which is where all the
parallelism comes from.

**Data-access pattern.** Each read reads its `T·C` posteriors once, in order, and
writes `≤ T` bytes of output. Arithmetic intensity is low (a few compares per float
loaded) → the kernel is **memory-bound**, as decode kernels generally are.

## 4. The GPU mapping

**Pattern: independent jobs, one thread per item** (PATTERNS.md §1; exemplar `1.12`
Tanimoto). Decoding read `r` is independent of every other read, so:

- **Thread-to-data mapping:** thread `i = blockIdx.x·blockDim.x + threadIdx.x`
  decodes read `i`. A **grid-stride loop** (`i += blockDim.x·gridDim.x`) lets a
  fixed-size grid cover a batch of any size (a real run has millions of reads).
- **Launch configuration:** `THREADS_PER_BLOCK = 128` (a multiple of the 32-lane
  warp, good occupancy on sm_75–sm_89). Blocks = `ceil(N / 128)`, capped at 1024;
  the grid-stride loop absorbs any remainder. We use 128 (not 256) because each
  thread runs an independent serial decode — there is no shared-memory cooperation
  to amortize over a bigger block, and a smaller block keeps register pressure low.
- **Memory hierarchy:** posteriors, offsets, and lengths live in **global memory**;
  each thread keeps `prev`, `cls`, and the output cursor in **registers**. There is
  **no shared memory** and **no constant memory** here — unlike `1.12`, there is no
  single shared "query" broadcast to all threads; each read reads its own slice.
- **No atomics, no races:** each read writes to its **own private output row**
  (`d_bases[i·max_T …]`), so threads never touch each other's memory. This disjoint
  output is exactly why the decode parallelizes with zero synchronization.
- **Jagged batch layout:** reads have different `T`, so all posteriors are packed
  into one flat array with a per-read `offset[]` (prefix sum of `T`). This keeps the
  host→device upload a *single* contiguous `cudaMemcpy` instead of one per read.
- **No CUDA library** is needed for the decode (it is integer compares + writes).
  The *full* pipeline uses cuBLAS/cuDNN/TensorRT for the network GEMMs — see §7 — but
  to hand-roll *those* you would implement batched matrix multiply and LSTM/attention
  kernels, which is the research-grade part we omit.

```
   batch of N reads (jagged T_r)            one thread per read
   ┌───────── flat probs[] (T·C floats) ─────────┐
   │ read0 (T0·C) │ read1 (T1·C) │ read2 (T2·C) … │
   └──────────────┴──────────────┴────────────────┘
        ▲offset0       ▲offset1       ▲offset2
        │              │              │
     thread0        thread1        thread2     …  (grid-stride for N > grid)
        │              │              │
        ▼              ▼              ▼
   ctc_greedy_decode → bases row 0 / 1 / 2 (private, no overlap)
```

## 5. Numerical considerations

- **Precision.** Posteriors are FP32 (`float`). But the *decode itself uses no
  float arithmetic that affects the result* — only `argmax` **comparisons** and
  integer index/byte writes. Argmax is invariant to any monotonic transform, so the
  result is identical whether the inputs are probabilities, logits, or log-probs.
- **Determinism (the key point).** PATTERNS.md §3 warns that floating-point
  `atomicAdd` reductions are non-deterministic because float addition is not
  associative. **We have no reductions at all.** The argmax breaks ties by the
  **lowest class index** (strict `>` in `ctc_argmax_step`), and every output write
  happens in a fixed order. The per-read checksum is an **integer** polynomial hash.
  So the output is bit-identical across runs *and* across CPU/GPU.
- **Race conditions.** None: disjoint per-read output rows, no shared accumulators.
- **Edge cases.** All-blank steps → empty output (`len = 0`); the loader rejects
  `T ≤ 0` and a class-count mismatch; `ctc_class_to_base` guards out-of-range classes.

## 6. How we verify correctness

- **The CPU reference** (`basecall_cpu` in `src/reference_cpu.cpp`) loops over reads
  and calls the **identical** `ctc_greedy_decode` the GPU kernel calls — the
  `__host__ __device__` shared-core idiom (PATTERNS.md §2). Both compile the *same
  source* (nvcc for the kernel, the host compiler for the reference).
- **The tolerance is `0` (exact).** Because both sides run the same integer-only
  decode, a correct run produces **byte-identical** base strings, lengths, and
  checksums for every read (PATTERNS.md §4, the "exact" case). `main.cu` counts how
  many of the `N` reads match all three; `PASS` requires all `N`. There is no
  floating-point tolerance to tune — any mismatch is a real bug, not numerical noise.
- **A second, stronger check (the science, not just CPU==GPU).** The synthetic sample
  *plants known sequences* (PATTERNS.md §6): read 0 is built to decode to `ACGTACGT`,
  read 1 to `AACCGGTT`, etc. The demo's stdout shows the recovered `seq=…`, so a human
  can confirm the decoder recovers the planted truth — including homopolymers
  (`AACCGGTT`, `TTTTGGGG`), the case the blank symbol exists to handle.

## 7. Where this sits in the real world

**The stage we built is real, but it is the *easy* stage.** Production basecallers
(Dorado, Guppy) are dominated by the **neural network** that produces the posteriors:

- **Network.** Older models used a stack of **bidirectional LSTMs**; current "sup"
  models use a **transformer/convolution encoder**. It maps the raw squiggle (after
  normalization and chunking) to the `T × C` posterior matrix — the input we *assume*.
- **CTC loss / training.** The network is trained with the **CTC loss** (the forward
  algorithm summing over all alignments), on labeled reads with known truth sequences.
- **Decoding.** For high accuracy, callers use **CTC beam search** (track the top-`k`
  label prefixes, summing path probabilities) rather than greedy. Greedy is the
  "fast" path and a strict special case; it notably **under-calls homopolymers** and
  ignores near-ties — which Exercise 1 and 3 explore.
- **GPU implementation.** The heavy lifting is **batched GEMMs** (cuBLAS), recurrent/
  attention kernels (**cuDNN**), and inference graphs optimized with **TensorRT**, with
  **CUDA streams** pipelining signal chunks and **NCCL/NVLink** scaling across GPUs.
  ONT reports up to ~30% throughput gains for HAC models on Ampere/Ada/Blackwell.
- **More than bases.** Modern runs also call **modified bases** (5mC/6mA methylation)
  from the same signal via extra classification heads, and tools like **f5c**
  CUDA-accelerate event alignment for downstream analysis.

What we omit, then, is *the model* — its weights, its training, and its GEMM-heavy
inference. What we keep is the conceptually crucial, fully-verifiable bridge from
"network probabilities" to "DNA bases", on the same one-thread-per-read GPU pattern
the real decoder uses.

---

## References

- **Graves, Fernández, Gomez, Schmidhuber (2006), "Connectionist Temporal
  Classification."** ICML. The origin of CTC, the blank symbol, and best-path
  (greedy) vs. forward-algorithm decoding — the math in §2.
- **Oxford Nanopore — Dorado** (https://github.com/nanoporetech/dorado). The modern
  production basecaller; read its decode path to see greedy/beam CTC in CUDA at scale.
- **f5c** (https://github.com/hasindu2008/f5c). CUDA-accelerated event alignment and
  methylation calling — a natural next step beyond basecalling.
- **awesome-nanopore** (https://github.com/GoekeLab/awesome-nanopore). Curated index
  of nanopore datasets and GPU-enabled tools; where to find real R10.4.1 data.
- **GIAB / NIST** (https://www.nist.gov/programs-projects/genome-bottle). Truth sets
  (NA12878/HG002) to benchmark basecalling accuracy against.
