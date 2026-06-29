# THEORY — 1.20 Reaction Yield / Retrosynthesis Scoring

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

> **Scope.** This project is the **reduced-scope teaching version** (CLAUDE.md §13)
> of a research-grade system. It implements and parallelizes the **route-scoring**
> stage of retrosynthesis; the planner and the neural yield model that *produce*
> the routes and features are described in §7 but not implemented.

---

## 1. The science

**Retrosynthesis** is planning a chemical synthesis *backwards*. You start from a
**target molecule** (say a drug candidate) and ask: "what one reaction could make
this from simpler precursors?" Apply that, and you now have one or more simpler
molecules; recurse on each until every leaf is a **purchasable building block**
(a compound you can buy from a catalog like Enamine or Sigma-Aldrich). The tree
of reactions from buyable leaves up to the target is a **synthetic route**.

Two facts make this hard:

1. **The tree explodes.** At each molecule, dozens of reaction templates might
   apply, each yielding different precursors. The number of candidate routes grows
   combinatorially, so a planner (AiZynthFinder, ASKCOS) uses **Monte Carlo Tree
   Search (MCTS)** to explore promisingly rather than exhaustively.
2. **Not all routes are equal.** A route can be *valid on paper* but terrible in
   practice — low-yielding steps, harsh conditions, poor selectivity, or
   precursors nobody sells. So every candidate route must be **scored** for how
   likely it is to actually work, and the best ones surfaced to a chemist.

This project is about that **scoring** step. Scoring is also exactly where the GPU
earns its keep: an MCTS planner generates *millions* of candidate routes, and each
route's score is independent of the others — perfect data parallelism. In
synthesis-aware generative drug design, this score becomes a **synthesizability**
signal that steers the molecule generator toward things you can actually make.

## 2. The math

We score a batch of `N` candidate routes. Route `r` has up to `MAX_STEPS`
reaction steps; step `s` is a feature vector

$$\mathbf{x}_{r,s} \in \mathbb{R}^{F}, \quad F = \texttt{NUM\_FEATURES} = 4,$$

with the four features (all scaled to roughly `[0,1]`):

| slot | symbol | feature | direction |
|---|---|---|---|
| 0 | $x_0$ | `template_prior` — template reliability | higher = better |
| 1 | $x_1$ | `precedent_count` — `log10(1+#precedents)`, normalized | higher = better |
| 2 | $x_2$ | `condition_penalty` — harshness of conditions | **higher = worse** |
| 3 | $x_3$ | `selectivity` — regio/stereo selectivity | higher = better |

**Per-step yield.** A shared logistic model maps a step's features to a success
probability (a "yield" in `(0,1)`):

$$
z_{r,s} = b + \sum_{f=0}^{F-1} w_f\, x_{r,s,f},
\qquad
y_{r,s} = \sigma(z_{r,s}) = \frac{1}{1 + e^{-z_{r,s}}}.
$$

The weights $\mathbf{w}$ and bias $b$ are **shared by every step of every route**
(a global reaction-quality model). The sign of each weight encodes its direction:
`condition_penalty` gets a **negative** weight ($w_2<0$), so harsher conditions
push the yield down. This is a deliberately interpretable surrogate for the
production transformer/GNN (see §7).

**Per-route score.** A route succeeds end-to-end only if **all** its steps
succeed. Treating steps as independent, the route's success probability is the
**product** of step yields, scaled by a building-block **availability** factor
$a_r \in [0,1]$ that rewards routes ending in in-stock precursors:

$$
S_r \;=\; a_r \prod_{s \in \text{steps}(r)} y_{r,s}
\;=\; a_r \prod_{s} \sigma\!\Big(b + \mathbf{w}\cdot \mathbf{x}_{r,s}\Big)
\;\in\; [0,1].
$$

**Inputs:** the `N` routes (features + availability) and the shared `(w,b)`.
**Output:** the `N` scores `S_r`, and the **top-K** routes by score. Higher is
better (more synthesizable).

## 3. The algorithm

```
for each route r in 0..N-1:                 # INDEPENDENT across r  -> parallel
    prob = 1
    for each real step s in route r:        # <= MAX_STEPS, unrolled
        z = b + dot(w, x[r,s])              # F multiply-adds
        prob *= sigmoid(z)                  # one expf + a divide
    S[r] = prob * availability[r]
rank S -> top-K
```

**Complexity.** Scoring all routes is

$$
\Theta(N \cdot \texttt{MAX\_STEPS} \cdot F)
$$

floating-point operations — linear in the batch size. Each route touches only its
own `MAX_STEPS·F` features (a few dozen floats) plus the tiny shared model, so the
**work is regular and the access pattern is simple** (contiguous per-route reads).

- **Serial:** one core walks all `N` routes → `Θ(N·MAX_STEPS·F)` time.
- **Parallel:** `N` independent routes → **work** `Θ(N·MAX_STEPS·F)`, **depth**
  `Θ(MAX_STEPS·F)` (one route, fully unrolled). With `P` threads the time is
  `Θ(N·MAX_STEPS·F / P)`. Since `MAX_STEPS·F` is a small compile-time constant,
  each thread does a fixed, branch-light amount of work — ideal for a GPU.

The ranking (top-K) is `Θ(N log K)` with a partial sort; for the tiny teaching
batch we do it on the host (see Exercise 2 for a GPU top-K).

## 4. The GPU mapping

**One thread per route.** Thread `(blockIdx.x, threadIdx.x)` owns route

```
r = blockIdx.x * blockDim.x + threadIdx.x
```

and, via a **grid-stride loop**, also routes `r + stride`, `r + 2·stride`, … so a
fixed-size grid covers a batch of any size. Block size is **256** threads (8 warps
— a multiple of the 32-lane warp, good occupancy on sm_75…sm_89); the grid is
`ceil(N/256)` blocks, capped at 1024 (the grid-stride loop handles the rest).

```
        candidate routes (global memory, row-major)
        +-----------+-----------+-----------+-----  ... -----+
 feats: | route 0   | route 1   | route 2   |                |   each route =
        | MAX_STEPS  x NUM_FEATURES floats   |                |   ROUTE_STRIDE floats
        +-----------+-----------+-----------+-----  ... -----+
              |           |           |
            thread0     thread1     thread2     ...    (grid-stride for r >= #threads)
              |           |           |
              v           v           v
        out:  S[0]        S[1]        S[2]      ...           (one score per route)

   shared model (w,b)  -->  __constant__ memory  -->  broadcast to every warp
```

**Memory hierarchy and why:**

- **Constant memory** holds the shared logistic model `c_w[F]`, `c_b[1]`. Every
  thread reads the *same* few floats and none writes them, so the constant
  cache's **broadcast** path serves an entire warp from one address in a single
  transaction — far cheaper than `F` global loads per thread. (This is the same
  trick as 1.12's constant-memory query.)
- **Global memory** holds the route features and availability. Route `r`'s block
  is contiguous; with one thread per route, consecutive threads read *different*
  route blocks, so accesses are strided rather than perfectly coalesced. For this
  compute-light kernel that is fine (we are not bandwidth-bound on a few dozen
  floats per thread); a layout that coalesces — store feature `f` of all routes
  together (struct-of-arrays) — is a worthwhile exercise at large `N`.
- **Registers** hold the running product and the dot-product accumulator. The
  inner loops over `MAX_STEPS` and `F` are compile-time bounds, so nvcc **fully
  unrolls** them into straight-line FMAs with no loop overhead and no per-thread
  local memory.
- **No shared memory, no atomics.** Outputs are fully independent (`out[r]`
  written by exactly one thread), so there is nothing to synchronize or reduce.

**No CUDA library is needed** for the scoring kernel — it is plain arithmetic plus
`expf` (a device intrinsic). A production system *would* lean on **cuDNN** for the
transformer attention that predicts the features, and on **CUB/Thrust** for a GPU
top-K; those are noted as the real-world extensions (§7) rather than used here, so
the teaching kernel stays a white box.

## 5. Numerical considerations

- **Precision (FP32).** Yields and the score are single precision. The features
  are `[0,1]` and the product of a handful of `(0,1)` yields stays comfortably in
  range (no overflow; underflow only for very long, very bad routes — see the
  log-domain exercise). FP32 is the natural precision for an ML-style scorer and
  matches what FP16/FP32 inference would use in production.
- **Determinism.** The kernel uses **no atomics and no cross-thread reduction**:
  each `out[r]` is computed by one thread from one route, in a fixed order. So the
  GPU result is **identical run to run** (the demo's stdout is byte-stable), and
  it does **not** depend on thread scheduling. (Contrast 5.01/11.09, which must
  use integer/fixed-point atomics precisely because float atomic sums reorder.)
- **CPU vs GPU: close, not bit-identical.** The CPU reference and the kernel call
  the *same* `route_score()` source (route_score.h). Yet the measured
  `max_abs_err ≈ 6e-8`, not `0`. Two single-precision effects cause this:
  1. **`expf` differs by a few ULPs** between the host C runtime and the device
     intrinsic; that tiny difference propagates through `sigma` and the product.
  2. **FMA contraction:** the GPU fuses `b + w·x` multiply-adds into hardware FMAs
     (one rounding instead of two), which the host compiler may not, so the
     pre-sigmoid `z` differs in the last bit.
  Both are real, expected, single-precision phenomena — and exactly the kind of
  honesty PATTERNS.md §4 asks for. We therefore verify to a **physically
  negligible** tolerance rather than pretending the results are exact.

## 6. How we verify correctness

The trusted baseline is `score_routes_cpu()` in `src/reference_cpu.cpp`: one plain
serial loop that calls the same `route_score()` the kernel calls. `main.cu` runs
both and computes `max_abs_err = max_r |S_r^{CPU} - S_r^{GPU}|`.

- **Tolerance: `1e-6`.** Chosen to sit safely above the observed `~6e-8`
  single-precision `expf`/FMA divergence (§5) and far below any score difference
  that could change the ranking. We *document* the gap rather than hide it.
- **A second, stronger check — a planted answer.** The synthetic sample
  (`make_synthetic.py`) builds **route 0** to be the unambiguous best (short, high
  template priors, low condition penalty, high selectivity, ~fully in stock). The
  demo asserts `route[0]` ranks #1, so we validate not just "CPU == GPU" but "the
  scorer recovers the route we know is best" (PATTERNS.md §6). The Python generator
  prints route 0's score (`0.949570`), matching the C++ output (`0.949571`) to
  rounding — an independent third implementation agreeing.
- **Edge cases.** Padded steps (routes shorter than `MAX_STEPS`) contribute a
  yield of 1 (multiplicative identity) and are skipped via the `STEP_ABSENT`
  sentinel; the loader rejects shape mismatches and out-of-range availability.

Why is this convincing? An independent serial implementation, written for clarity
with no parallelism, agreeing with the parallel GPU implementation to FP precision
— **and** both recovering a known-best route — is strong evidence the GPU plumbing
(indexing, constant memory, grid-stride loop) is correct.

## 7. Where this sits in the real world

A production retrosynthesis system (AiZynthFinder, ASKCOS, IBM RXN) is far more
than this scorer:

- **Templates / reaction prediction.** Where do steps come from? Either a library
  of **reaction templates** extracted from USPTO/Reaxys, applied by subgraph
  matching, or a **template-free** sequence model — the **Molecular Transformer**
  / **Chemformer** — that reads reaction SMILES and *generates* products with
  **cuDNN-backed attention** and **GPU beam-search decoding**. That is the
  catalog's "transformer on augmented SMILES / seq2seq".
- **Planning (MCTS).** The route *tree* is explored by **Monte Carlo Tree Search**:
  selection → expansion → rollout → backpropagation, repeated thousands of times.
  Rollouts are **batched on the GPU** so many candidate expansions are scored per
  launch — and *that batched scoring* is exactly the kernel this project teaches,
  just with a neural yield model in place of our logistic one.
- **Yield / feasibility model.** Real per-step scores come from a **GNN or
  transformer** trained on reaction outcomes (yields, conditions), not four
  hand-features. Swapping our `step_yield()` for a neural net changes *what fills
  the features*, not the GPU mapping: you would still score a batch of routes, one
  per thread (or one per block for a heavier model), reading shared weights from
  device memory.
- **Stock / cost.** The `availability` factor stands in for a real **in-stock**
  lookup against a buyable-compounds database plus cost/greenness terms.

What this teaching version deliberately omits: SMILES parsing, template
application, tree search, a trained model, and FP16 mixed precision. What it keeps
faithfully: the **embarrassingly-parallel batched route scoring**, the
**constant-memory shared model**, the **grid-stride** batch sweep, and an honest
CPU/GPU verification — the reusable GPU lesson at the core of the real system.

---

## References

- Schwaller et al., *Molecular Transformer* (2019) — seq2seq reaction prediction;
  the tokenization + beam-search decoding our §7 references.
  <https://github.com/pschwllr/MolecularTransformer>
- Genheden et al., *AiZynthFinder* (2020) — open MCTS retrosynthesis planner; the
  route scoring + in-stock check that inspired our `availability` factor.
  <https://github.com/MolecularAI/aizynthfinder>
- Coley et al., *ASKCOS* — synthesis-planning platform (templates + scoring +
  conditions). <https://github.com/ASKCOS/ASKCOS>
- Irwin et al., *Chemformer* (2022) — BART-based reaction model (pre-train /
  fine-tune for reaction tasks). <https://github.com/MolecularAI/Chemformer>
- USPTO-50k / USPTO-MIT reaction datasets:
  <https://github.com/connorcoley/rexgen_direct>,
  <https://github.com/wengong-jin/nips17-rexgen>; Open Reaction Database
  <https://open-reaction-database.org>.
- NVIDIA CUDA C++ Programming Guide — constant memory, grid-stride loops, FMA
  contraction (`--fmad`), and `expf` accuracy. <https://docs.nvidia.com/cuda/>
