# THEORY — 7.2 Drug-Target Interaction Prediction (GNN)

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

This project ships a **reduced-scope teaching version** (CLAUDE.md §13): a
*fixed-weight* (untrained) message-passing graph neural network (GNN) that scores
drug × protein pairs. It teaches the CUDA data-flow of a GNN and pairwise scoring
without the machinery (training loops, protein transformers, Flash Attention)
that would obscure the parallel patterns. §7 describes what a production system
adds.

---

## 1. The science

A **drug** is a small molecule; a **target** is usually a protein. A drug works
by physically **binding** its target (fitting into a pocket, like a key in a
lock) and changing what the protein does. **Drug–Target Interaction (DTI)**
prediction asks, computationally: *given a molecule and a protein, will they bind,
and how strongly (dissociation constant Kd, inhibition constant Ki, or a binary
"interacts / does not")?* Answering this in silico lets a discovery pipeline
**virtually screen** millions of candidate compounds against a target and only
synthesize the promising few — the expensive wet-lab step.

A molecule is naturally a **graph**: atoms are nodes, chemical bonds are edges.
Graphs have *irregular* topology (different molecules have different numbers of
atoms and connectivity), which is exactly what **graph neural networks** are
built for. The core idea: each atom's meaning depends on its neighbourhood
(a carbon in a benzene ring behaves differently from a carbon in a methyl group),
so we let atoms **exchange information along bonds** and build up a vector summary
("embedding") of the whole molecule.

## 2. The math

**Inputs.** A batch of `D` drug graphs. Drug `d` has `n_d` atoms; atom `i` starts
with a feature vector `h_i^{(0)} ∈ R^F` (here `F=8`; in practice one-hot atom
type, degree, charge, aromaticity, …). Bonds define an adjacency; we add a
**self-loop** to every node. We also have `P` protein descriptor vectors
`x_p ∈ R^F`.

**Message passing (one round `t`).** For every node `i`, aggregate the current
features of its neighbours `N(i)` (self-loop included), then transform:

```
  m_i^{(t)}   = Σ_{j ∈ N(i)}  h_j^{(t)}                       (aggregate: sum)
  h_i^{(t+1)} = ReLU( W^{(t)} · m_i^{(t)} + b^{(t)} )         (update: linear + ReLU)
```

`W^{(t)} ∈ R^{F×F}` and `b^{(t)} ∈ R^F` are **shared across all nodes** ("weight
tying") — the network has the same tiny parameter set no matter how big the
graph. We run `T` rounds (here `T=2`, so each atom sees its 2-hop neighbourhood).

**Graph readout (pooling).** Collapse the final node features of drug `d` into a
single embedding by summing:  `z_d = Σ_{i ∈ drug d} h_i^{(T)} ∈ R^F`.

**Protein encoding.**  `y_p = ReLU( W_p · x_p + b_p ) ∈ R^F`  (one linear layer).

**DTI score.** For each drug–protein pair, a scaled dot product through a sigmoid:

```
  score(d,p) = σ( (z_d · y_p) / F ),      σ(u) = 1 / (1 + e^{-u})
```

giving a probability in `(0,1)`. The output is the dense `D × P` score matrix.
(A trained model would learn `W`, `b`, `W_p`, `b_p` from measured affinities; here
they are **fixed and seeded** — see §5, §6.)

## 3. The algorithm

```
for t in 0..T-1:                         # message-passing rounds
    for each node i:                     #   O(deg(i)·F) aggregate + O(F²) update
        m = Σ neighbours' features
        h'[i] = ReLU(W[t]·m + b[t])
for each drug d:  z[d] = Σ its nodes' h  # pooling, O(n_d·F)
for each protein p: y[p] = ReLU(Wp·x[p]+bp)  # O(F²)
for each pair (d,p): score = σ(z[d]·y[p]/F)  # O(F)
```

**Complexity.** Message passing costs `O(T · (E·F + V·F²))` where `V = total
atoms`, `E = total directed edges (incl. self-loops)`. Pooling is `O(V·F)`.
Scoring is `O(D·P·F)` — quadratic in the batch dimensions, which is why **pair
scoring dominates** at virtual-screening scale (`D` = millions of compounds,
`P` = thousands of targets). Serially this is one big loop nest; the parallel
structure below turns each independent unit of work into a thread.

## 4. The GPU mapping

Three of the four stages are the classic **"independent jobs"** / **"gather"**
patterns from `docs/PATTERNS.md §1`; the message-passing rounds use **double
buffering** (ping-pong) exactly like a stencil solver.

- **`message_pass_kernel` — one thread per NODE (a GATHER over edges).** Thread
  `i = blockIdx.x*blockDim.x + threadIdx.x` owns node `i`. It walks its CSR row
  `adj[adj_off[i] .. adj_off[i+1])`, sums neighbour feature vectors into
  registers, then applies the shared linear layer + ReLU. Because **each output
  node is written by exactly one thread, there are no atomics and no races** —
  the irregular graph is handled entirely by the CSR indirection, not by locking.
  Two node-feature buffers are ping-ponged between rounds (`src → dst`, swap).
- **`pool_kernel` — one thread per DRUG.** Sums the drug's node rows in index
  order into the embedding. Deterministic, no atomics.
- **`protein_encode_kernel` — one thread per PROTEIN.** One linear layer + ReLU.
- **`score_kernel` — one thread per PAIR.** The 2-D `D × P` grid is flattened to
  1-D: thread `j` handles `drug = j/P`, `protein = j%P`. `D·P` independent dot
  products — the shape of large-library scoring.

**Memory hierarchy.** The weights (`W`, `b`, `W_p`, `b_p` — only 216 floats) are
**read by every thread but never change during a launch**, so they live in
`__constant__` memory. Constant memory has a broadcast cache: when every thread in
a warp reads the *same* address (which they do — all nodes use the same `W`), the
read is a single transaction. This is the same trick as the Tanimoto query in
flagship 1.12. Node features, adjacency, and scores live in **global memory**;
the per-node feature vector and the aggregation accumulator live in **registers**
(the `#pragma unroll`'d length-`F` loops keep them there).

**Launch config.** 256 threads/block (multiple of the 32-lane warp; 8 warps to
hide latency; good occupancy on sm_75…sm_89). Grids are `ceil(work/256)` for each
stage's work count (nodes / drugs / proteins / pairs).

```
 batched graph (CSR)          message passing (T rounds, ping-pong)
 ┌──────────────┐             featA ──kernel──► featB ──kernel──► featA ...
 │ node_off[D+1]│             (1 thread / node, gathers CSR neighbours)
 │ adj_off[V+1] │                         │
 │ adj[E]       │                         ▼  pool (1 thread / drug)
 │ feat[V*F]    │             z[D*F] drug embeddings
 └──────────────┘                         │
 prot[P*F] ─encode(1 thr/prot)─► y[P*F]   │
                                          ▼
                       score[D*P] = σ(z·y/F)   (1 thread / (drug,protein) pair)
```

**No library needed.** The dense `F×F` layer is small enough to hand-roll in
registers (that *is* the teaching point). A production system replaces these with
**cuBLAS/cuDNN** GEMMs and **cuSPARSE / DGL / PyG** sparse-adjacency SpMM; the
`z_d · y_p` scoring over a whole library is itself a GEMM (`Z Yᵀ`). We write it by
hand here so nothing is a black box (CLAUDE.md §6.1.6).

## 5. Numerical considerations

- **Precision.** FP32 throughout — it is what real GNN inference uses, and the
  numbers here are `O(1)`.
- **Determinism.** Floating-point addition is **not associative**, so a sum's
  result depends on the order the terms are added. We make every reduction (the
  neighbour aggregation, the pooling sum, the dot product) run in a **fixed
  index order** identical on CPU and GPU, and we use **no `atomicAdd`** (each
  output element has a single writer). Result: the GPU output is **bit-stable run
  to run** and stdout is byte-identical, so `demo/run_demo` can diff it
  (PATTERNS.md §3).
- **CPU vs GPU divergence.** The only difference is the GPU's **fused multiply-add
  (FMA)**: `a*b + c` is contracted into one rounding on the device but is two
  roundings on the host. Over these short sums that is ~`1e-6`, well under our
  tolerance (§6).

## 6. How we verify correctness

`src/reference_cpu.cpp` runs the **same forward pass serially**. Because the
per-element math (`gnn_linear_relu`, `gnn_dot`, `gnn_sigmoid`) lives in one
`__host__ __device__` header (`src/gnn.h`, the HD-macro idiom, PATTERNS.md §2),
the CPU and GPU execute *identical arithmetic in identical order*. `main.cu`
compares both the drug embeddings and the full score matrix and asserts
`max |cpu − gpu| ≤ 1e-4`. Observed errors are ~`1e-7` (embeddings) and ~`6e-8`
(scores) — pure FMA rounding, so the tolerance is honest and generous
(PATTERNS.md §4, "same exact operations → small FP tolerance").

A **second, stronger check** validates the *science of the pipeline*, not just
CPU==GPU agreement: the synthetic sample has an **implanted top pair**
(`make_synthetic.py` runs the same fixed model and records the argmax pair as
ground truth). The demo reports the model's top-scoring pair and confirms it
equals that pair (`RECOVERED`) — end-to-end evidence that message passing →
pooling → encoding → scoring is wired up correctly.

## 7. Where this sits in the real world

Production DTI (the catalog "Prior art"):

- **Trained** networks — this version's weights are seeded/untrained, so its
  scores are illustrative of the *mechanics* only, never binding predictions.
  Real models learn `W`, `b`, … from BindingDB/ChEMBL/Davis/KIBA labels via
  backprop; here we only run inference.
- **Richer GNNs** — GAT (attention-weighted neighbours), GIN (injective
  aggregation), and **DMPNN** (messages on directed edges, as in Chemprop) instead
  of plain sum-aggregation.
- **Protein encoders** — sequence transformers (**ESM-2**, ProtTrans) whose
  quadratic self-attention is accelerated by **Flash Attention 2**, replacing our
  single linear layer. The drug↔protein interaction is often **cross-attention**,
  not a bare dot product.
- **Scale & kernels** — **DGL / PyG** batch thousands of graphs and run
  cuSPARSE-backed SpMM; MLP heads run on **cuDNN**; scoring a whole library is a
  batched GEMM on **cuBLAS**. The parallel *shapes* (per-node gather, per-pair
  score) are exactly the ones this project makes explicit.

---

## References

- Gilmer et al., *Neural Message Passing for Quantum Chemistry* (2017) — the MPNN
  framework this project's message-passing step is a minimal instance of.
- Kipf & Welling, *Semi-Supervised Classification with GCNs* (2017) — the
  self-loop + neighbour-sum + linear + nonlinearity layer.
- Huang et al., **DeepPurpose** (<https://github.com/kexinhuang12345/DeepPurpose>)
  — 15 drug/protein encoders and 50+ DTI architectures; study how encoders plug
  into a shared scoring head.
- **TorchDrug** (<https://github.com/DeepGraphLearning/torchdrug>) and
  **DGL-LifeSci** (<https://github.com/awslabs/dgl-lifesci>) — how batched graphs
  and CUDA-backed sparse ops are engineered at scale.
- **DTA-GNN** (<https://github.com/lennylv/DTA-GNN>) — target-specific
  drug–target-affinity dataset construction and GNN training.
