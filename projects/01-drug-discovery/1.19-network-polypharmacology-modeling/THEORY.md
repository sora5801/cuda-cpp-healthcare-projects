# THEORY — 1.19 Network / Polypharmacology Modeling

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

A classical drug is imagined as a key that fits exactly one lock — one protein
target. Reality is messier: a typical small-molecule drug binds **several**
proteins. Those extra bindings are not a footnote; they are the mechanism behind
**side effects** (an off-target protein gets perturbed), behind **drug
repurposing** (a drug approved for one disease turns out to hit a target relevant
to another), and behind **drug combinations** (two drugs whose target sets
interact synergistically). The study of these many-to-many drug–protein
relationships is **polypharmacology**.

The natural data structure is a **graph**. Nodes are biological entities — drugs,
proteins, diseases, genes — and edges are relationships:

- *protein–protein interaction* (PPI): two proteins physically interact (STRING).
- *drug targets protein* (DTI): a drug binds a protein (DrugBank, STITCH).
- *drug treats disease*, *gene associated with disease*, etc.

Because there are several **types** of node and edge, this is a **heterogeneous
knowledge graph (KG)**. The central computational question of this project is
**link prediction**: given the drug node and the `TARGETS` edge type, which
protein nodes *should* be connected — i.e. which proteins does this drug also
bind, including ones we have not measured yet? Answering that at the scale of a
real KG (tens of thousands of proteins, millions of entities for a full
biomedical graph) is what needs a GPU.

This project implements the **scoring + ranking** half of one well-known link
predictor (TransE) and ships **synthetic, pre-computed embeddings**, because
*training* those embeddings on a million-node graph is the research-grade part
(see §7).

## 2. The math

**Knowledge-graph embedding.** Pick an embedding dimension `d`. Assign every
entity `e` a vector `v_e ∈ ℝ^d` and every relation type `r` a vector `w_r ∈ ℝ^d`.
A *fact* is a triple `(h, r, t)` — head entity, relation, tail entity — for
example `(aspirin, TARGETS, COX-1)`.

**The TransE assumption.** TransE (Bordes et al., 2013) models a relation as a
**translation** in embedding space: a true fact should satisfy

```
    v_h + w_r  ≈  v_t           (h + r lands near t)
```

So the relation `TARGETS` is a single vector you *add* to any drug to move it to
the region of its targets. The **plausibility score** of a candidate tail `t` is
how well this holds — the **negative distance** between the translated head and
the tail:

```
    f(h, r, t) = − ‖ (v_h + w_r) − v_t ‖₂            (larger = more plausible)
```

Symbols, units, ranges:

| Symbol | Meaning | Range / units |
|---|---|---|
| `d` | embedding dimension | small int (here 16; real models 50–200) |
| `v_h` | head (query drug) embedding | `ℝ^d`, unitless latent coords |
| `w_r` | relation (`TARGETS`) embedding | `ℝ^d` |
| `v_t` | candidate tail (protein) embedding | `ℝ^d` |
| `f` | plausibility score | real; `0` is the best (zero distance) |

**Inputs:** one head vector, one relation vector, and `n` candidate tail vectors.
**Output:** `n` scores; we rank them and take the top-K as predicted targets.

**Ranking-equivalent simplification.** The square root in the L2 norm is
order-preserving, so ranking by `f` is identical to ranking by the **negative
squared distance**

```
    s(t) = − Σ_{k=0}^{d−1} ( (v_h[k] + w_r[k]) − v_t[k] )²
```

We rank by `s`. Dropping the `sqrt` saves an op per candidate and — more
importantly for this repo — keeps the arithmetic a plain sum of products, which is
exactly the same on the CPU and the GPU (see §5–6). This is what `transe_score()`
in `src/transe.h` computes.

## 3. The algorithm

```
Input : head h (d floats), relation r (d floats), tails T (n×d floats)
Output: top-K candidate tail indices by score

1. for each candidate j in 0..n-1:                 # INDEPENDENT across j
2.     acc = 0
3.     for k in 0..d-1:                             # the shared per-tail core
4.         diff = (h[k] + r[k]) - T[j][k]
5.         acc += diff*diff
6.     score[j] = -acc
7. rank score[] descending, break ties by lower index
8. report top-K and how many ground-truth targets they recovered
```

**Complexity.**

- Scoring is `Θ(n · d)` multiply-adds. The two loops are a flat double loop with
  no data dependence between candidates.
- **Parallel work / depth.** Total work is the same `Θ(n·d)`, but the *depth*
  (critical path) is just `Θ(d)` — every candidate is computed simultaneously.
  With `n` threads the scoring is `Θ(d)` time, i.e. constant in `n`.
- Ranking the top-K is `Θ(n log K)` with a partial sort; for the tiny demo it is
  done on the host. (At graph scale you would do a GPU top-K with CUB.)

**Arithmetic intensity / access pattern.** Each candidate reads `d` tail floats
from global memory and the `d` shared query floats. The tail reads are the
bandwidth cost (`n·d` floats streamed once); the query is read by *every* thread,
which is why it belongs in constant memory (§4). Intensity is ~2 flops per tail
float loaded — this is a **memory-bound, streaming** kernel, the classic profile
for "score one query against many items".

## 4. The GPU mapping

**Pattern:** *independent jobs + constant-memory query* (PATTERNS.md §1, exemplar
`1.12`). One thread scores one candidate tail.

**Thread-to-data mapping.** Thread `(blockIdx.x, threadIdx.x)` owns candidate
`j = blockIdx.x * blockDim.x + threadIdx.x`, and a **grid-stride loop** lets a
fixed-size grid cover an arbitrarily large `n`:

```
j = block*blockDim + thread,  stride = blockDim * gridDim
for (; j < n; j += stride) score[j] = transe_score(c_head, c_relation, &T[j*d], d);
```

**Launch configuration.** `blockDim = 256` threads (8 warps) — a multiple of the
32-lane warp, enough warps for the scheduler to hide global-memory latency, good
occupancy on `sm_75…sm_89`. `gridDim = ceil(n/256)` capped at 1024 blocks (the
grid-stride loop handles any overflow).

**Memory hierarchy and *why*:**

- **Constant memory** holds `c_head` and `c_relation`. They are read by every
  thread and never written during the launch, so the constant cache **broadcasts**
  one address to a whole warp in a single transaction. If we instead kept them in
  global memory, every thread would re-fetch `2d` floats — wasted bandwidth on
  data that is identical for all threads. (`MAX_DIM = 256` floats = 1 KB each,
  trivially inside the 64 KB constant bank.)
- **Global memory** holds the candidate tails `T` (`n·d` floats). Consecutive
  threads handle consecutive candidates `j`, `j+1`, …; with row-major storage,
  within one dimension `k` the accesses `T[j*d+k]` are strided by `d`, so
  coalescing is imperfect for large `d`. For the teaching kernel this is fine; a
  production kernel would transpose `T` to `[d][n]` (column-major) so threads in a
  warp read *contiguous* floats — an exercise worth doing.
- **Registers** hold the running accumulator `acc` and the loop index; no shared
  memory or atomics are needed because outputs are fully independent.

**Why no CUDA library here.** The scoring is a tiny custom distance kernel — there
is nothing to outsource. The catalog's library mentions (cuSPARSE adjacency
products, FP16 embedding tables, GPU negative sampling) all belong to the
*training* pipeline in §7, not to inference-time scoring. We keep this kernel
hand-written precisely so there is **no black box** (CLAUDE.md §6.1.6).

```
            constant memory (broadcast)        global memory (streamed)
            ┌───────────────┐                  ┌─────────────────────────────┐
            │ c_head  [d]   │                  │ tails  T[n][d] (row-major)  │
            │ c_relation[d] │                  └─────────────────────────────┘
            └──────┬────────┘                                 │
                   │ broadcast to every thread                │ thread j reads row j
                   ▼                                          ▼
   grid:  [ block 0 ][ block 1 ] ... [ block G-1 ]    (256 threads each)
              │t0 t1 ... t255│
               └─ thread j → score[j] = -Σ_k ((h+r-t)[k])²  →  out[n] (global)
```

## 5. Numerical considerations

**Precision.** We use **FP32** throughout — embeddings are inherently low-precision
latent coordinates, and FP32 matches what GPU KG-embedding training uses (often
even FP16 for the tables). `d` is small, so the summation is well-conditioned and
FP32 is ample.

**Determinism.** The per-candidate sum runs the **same loop in the same order** on
host and device, and each candidate writes its own `out[j]` — there are **no
atomics** and **no cross-thread reductions**, so there is no floating-point
reordering and the GPU output is bit-stable from run to run. The demo's stdout is
therefore byte-identical every run (PATTERNS.md §3).

**The one real divergence: FMA.** `nvcc` *contracts* the device-side
`acc + diff*diff` into a single **fused multiply-add** (`fmaf`) by default, which
rounds once instead of twice and so keeps more precision than the host compiler's
separate multiply-then-add. The two results therefore differ by ~`1e-7` per
accumulation — here a measured `max_abs_err ≈ 9.5e-7`. This is **not a bug**; it is
the expected GPU-vs-host difference (PATTERNS.md §4). Two honest ways to handle it:

1. **Verify to a small tolerance** (`1e-5`) and document why — what we do. This
   reflects how real GPU code actually behaves and is the better lesson.
2. **Force bit-identical results** by disabling contraction (`nvcc --fmad=false`,
   or `__fmul_rn`/`__fadd_rn` intrinsics on the device side). Then the tolerance
   could be `0`, at the cost of a slightly slower, less-realistic kernel.

We chose (1) so the learner *sees* FMA in the wild rather than having it hidden.

## 6. How we verify correctness

`src/reference_cpu.cpp` provides `transe_score_cpu()`, an independent serial
scorer. Crucially, **both** the CPU reference and the GPU kernel call the **same**
`transe_score()` from `src/transe.h` (the `__host__ __device__` core idiom,
PATTERNS.md §2), so the only difference between them is the execution substrate —
not the math. `main.cu` runs both, computes `max_abs_err` over all `n` scores, and
asserts it is `≤ 1e-5` (§5 explains the value). Agreement between two independent
implementations that share only the *formula* is strong evidence the GPU plumbing
(indexing, constant-memory upload, grid-stride loop) is correct.

**A second, stronger check: scientific recovery.** The synthetic data
(`scripts/make_synthetic.py`) embeds a **known answer** — a handful of "true
target" proteins constructed so that `h + r` lands on (or extremely near) their
embeddings, with all other proteins random decoys far away (PATTERNS.md §6). The
demo reports `recovery: k / n_true ground-truth targets in top-5`. Getting
`3 / 3` (with the exact-match target scoring `−0.000000`) validates that the
*method* — not just CPU==GPU agreement — does what TransE link prediction claims.

**Edge cases.** The loader throws on a missing/truncated file or a non-positive
`n`/`dim`; the kernel's grid-stride guard `j < n` protects the ragged last block;
the all-decoys-far construction makes the ranking unambiguous (no tie at the top).

## 7. Where this sits in the real world

This project is a **reduced-scope teaching version**. Production polypharmacology
modeling differs in three big ways:

- **Training, not just scoring.** The embeddings here are handed to us. In
  practice you *learn* them by minimizing a **margin-ranking loss** over the
  graph's triples: for each true `(h, r, t)`, corrupt the tail to make a negative
  `(h, r, t')` and push `f(h,r,t)` above `f(h,r,t')` by a margin, via SGD. This is
  where the catalog's GPU ingredients live: **GPU-batched negative sampling**,
  **FP16 embedding tables** for millions of entities, and **cuSPARSE** sparse
  adjacency products for the message-passing GNN variants. **PyTorch Geometric**,
  **DGL**, and **OpenKE** implement these at scale.
- **Richer models.** TransE cannot represent symmetric relations (`h+r≈t` and
  `t+r≈h` can't both hold for `r≠0`) or one-to-many relations cleanly. **RotatE**
  models a relation as a rotation in complex space (`h ∘ r ≈ t`, `|r_k|=1`) and
  fixes much of this; **GraphDTA/DeepDTA** use GNNs/CNNs over molecular graphs and
  protein sequences; **heterogeneous GNNs** pass messages along typed edges.
- **Heterogeneous, multi-relational graphs.** Real KGs mix PPI, DTI, disease, and
  pathway edges; **network diffusion** and **community detection** propagate signal
  across them, and **DeepSynergy**-style models predict drug-*combination* synergy
  (DrugComb). Our single-relation, single-query slice is the smallest piece that
  still teaches the embedding-and-rank idea end to end.

The GPU lesson transfers directly: whether you score TransE distances, RotatE
rotations, or GNN logits, *scoring one query against every candidate* is the
independent-jobs pattern this kernel demonstrates.

---

## References

- **Bordes et al., "Translating Embeddings for Modeling Multi-relational Data"
  (TransE), NeurIPS 2013** — the model implemented here; read for the translation
  assumption and the margin loss.
- **Sun et al., "RotatE", ICLR 2019** — relations as complex rotations; the natural
  next model (Exercise 1).
- **STRING / DrugBank / STITCH / DrugComb** (see `data/README.md`) — the real
  graphs this method consumes; study their schemas and licenses.
- **PyTorch Geometric** (https://github.com/pyg-team/pytorch_geometric) — GPU
  heterogeneous graph learning; the production training stack.
- **DGL** (https://github.com/dmlc/dgl) — GPU graph learning for DTI networks.
- **OpenKE** (https://github.com/thunlp/OpenKE) — focused KG-embedding library;
  the cleanest place to see the TransE training loop this project assumes.
- **DeepPurpose** (https://github.com/kexinhuang12345/DeepPurpose) — DTI prediction
  toolkit (DeepDTA/GraphDTA); the GNN/CNN alternative to KG embeddings.
