# THEORY — 1.11 QSAR / Property Prediction

A guided deep dive for a reader who knows C++ but is new to CUDA and new to
graph neural networks. We build the *why* behind every line in `src/`.

---

## 1. The science

**QSAR** — *Quantitative Structure–Activity Relationship* — is the decades-old
idea that a molecule's measurable behavior (solubility, toxicity, binding
affinity, ADMET properties) is a *function of its structure*. If we can learn that
function `f(structure) -> property`, we can **screen** millions of hypothetical
molecules in silico and only synthesize the promising ones.

The question is how to feed a *molecule* to a model. A molecule is naturally a
**graph**:

- **atoms** are nodes, each carrying features (element, charge, hybridization, …),
- **bonds** are edges (single/double/aromatic, in-ring or not).

Classical QSAR hand-crafted fixed descriptors (counts, fingerprints) and fed them
to a regressor. The modern approach **learns** the descriptor: a **message-passing
neural network (MPNN)** lets each atom repeatedly exchange information with its
bonded neighbors, so after a few rounds every atom's vector summarizes its local
chemical environment. Pooling those vectors gives a molecule-level fingerprint
that a final layer maps to the property. This is what Chemprop (D-MPNN), DGL-
LifeSci, and DeepChem do at industrial scale.

We implement the *simplest* MPNN — the **Graph Convolutional Network (GCN)** of
Kipf & Welling (2017) — because its single update equation contains every idea
(neighbor aggregation, normalization, learned transform, nonlinearity) that the
fancier variants elaborate.

> **Scope & honesty.** This is a *reduced-scope teaching* version: we run GCN
> **inference** with small, *untrained* weights on a tiny *synthetic* batch. The
> printed "property" is a demonstration number, never a real measurement. Training
> (backprop) and real featurization are described in §7.

---

## 2. The math

### One GCN layer

Let `H ∈ ℝ^{N×F}` be the node-feature matrix (`N` atoms, `F` channels) and `A` the
adjacency matrix. A GCN layer computes

```
H' = σ( Â · H · W )         with   Â = D̃^{-1/2} (A + I) D̃^{-1/2}
```

- `A + I` adds a **self-loop**: every atom keeps its own signal, not just its
  neighbors'.
- `D̃` is the diagonal **degree matrix** of `A + I` (so `D̃_ii = 1 + deg(i)`).
- `D̃^{-1/2}(·)D̃^{-1/2}` is the **symmetric normalization**: it scales edge `(i,j)`
  by `c_ij = 1 / sqrt(D̃_ii · D̃_jj)`. Without it, high-degree atoms would dominate
  and feature magnitudes would explode as you stack layers.
- `W ∈ ℝ^{F×F'}` is the **learned linear map** (the "convolution filter weights").
- `σ` is a nonlinearity (**ReLU**, `max(0,x)`), applied on hidden layers.

Written per node `i` and output channel `o`, with `N(i)` the neighbors of `i`
*including* `i` itself:

```
H'[i,o] = σ(  b[o] + Σ_{j ∈ N(i)}  c_ij · Σ_k H[j,k] · W[k,o]  )
```

This is exactly `gcn_aggregate_then_transform()` in [`src/gcn.h`](src/gcn.h):
the outer sum over neighbors `j`, the inner dot product `H[j,:]·W[:,o]`, scaled by
`c_ij`, plus a bias, then ReLU. (We add a bias `b`, a standard practical extension
to the original equation.)

### The full model (what `gcn_predict_*` computes)

```
H1 = ReLU( Â · H0 · W1 + b1 )        # layer 1:  F_IN(6)  -> F_HID(8)
H2 =       Â · H1 · W2 + b2          # layer 2:  F_HID(8) -> F_OUT(4)   (no ReLU)
g_m = (1/|m|) Σ_{a ∈ m} H2[a, :]     # READOUT: mean-pool a molecule's atoms
y_m = head_w · g_m + head_b          # linear head -> one scalar per molecule
```

Two layers means information flows **2 hops** across the molecule — enough for an
atom to "feel" its neighbors-of-neighbors. Mean pooling is **permutation
invariant** (atom ordering does not change the molecule-level answer), which is
mandatory for a graph-level prediction.

---

## 3. The algorithm

```
load batch  -> CSR (row_ptr, col_idx with self-loops), degrees, features, mol_start
build/seed  -> fixed weights W1,b1,W2,b2,head_w,head_b
for each layer L in {1, 2}:
    for each node i:                       # PARALLELIZED on the GPU
        out[i,:] = bias
        for each neighbor j in N(i):       # CSR slice row_ptr[i]..row_ptr[i+1]
            c = 1/sqrt(deg[i]*deg[j])
            out[i,:] += c * (H[j,:] @ W)
        if L is hidden: out[i,:] = ReLU(out[i,:])
for each molecule m:                       # PARALLELIZED on the GPU
    y[m] = head_w · mean_{a in m} H2[a,:] + head_b
```

### Complexity

Let `E` be the number of edges (including self-loops), `F`/`F'` the layer widths.

- **One layer:** `O(E · F · F')` multiply-adds — each edge touches an `F×F'` map.
- **Whole model:** `O(E · (F_IN·F_HID + F_HID·F_OUT))` ≈ `O(E)` for fixed widths.
- **Serial vs parallel:** the serial CPU does these `E·F·F'` ops one after another;
  the GPU does all `N` nodes of a layer **at once** (one thread per node), so the
  wall-clock time for a layer drops from `O(E·F·F')` to roughly
  `O(max_i deg(i) · F · F')` (the busiest thread) given enough cores.

---

## 4. The GPU mapping

This project is the **per-output gather** pattern (docs/PATTERNS.md §1), the
graph-shaped cousin of CT backprojection's gather.

### Threads → data

- **Layer kernel** (`gcn_layer_kernel`): a 1-D grid of `ceil(N/128)` blocks ×
  128 threads. **Thread `i = blockIdx.x*blockDim.x + threadIdx.x` owns output node
  `i`.** It reads its neighbor slice `col_idx[row_ptr[i] .. row_ptr[i+1])` and
  writes only `out[i, :]`. Two layers = two launches.
- **Readout kernel** (`gcn_readout_kernel`): **thread `m` owns molecule `m`**,
  mean-pools its atoms (range `mol_start[m]..mol_start[m+1]`) and applies the head.

### Why this mapping (and the alternative we rejected)

The "natural" message-passing primitive is a **scatter**: loop over edges and
*add* each message into the destination node. On a GPU many edges target the same
node, so a scatter needs `atomicAdd` — which (for floats) is **non-deterministic**
because float addition is not associative (PATTERNS.md §3). We instead do a
**gather**: each output node *pulls* its neighbors. Now every thread writes a
disjoint row — **no atomics, no races** — and it sums neighbors in the **fixed CSR
order**, identical to the CPU. Determinism falls out for free, which is exactly
what makes the CPU↔GPU check meaningful.

### Memory hierarchy

- **Constant memory** holds all the weights (`c_weights`, ≤ 96 floats). Every
  thread reads the same `W`/`b`; constant memory's per-SM broadcast cache serves
  one address to a whole warp in a single transaction — the textbook use (same
  trick as the query in flagship 1.12). Hand-rolling would mean a global-memory
  load per thread per weight.
- **Global memory** holds `H`, the CSR arrays, and the outputs. The CSR is read
  with `__restrict__` pointers so the compiler can assume no aliasing.
- **Registers/local** hold each thread's output row (`float row[GCN_F_HID]`) while
  it accumulates, so we touch global memory once to write the finished row.

### Why constant width matters

`GCN_F_IN/HID/OUT` are compile-time constants, so the inner loops can be unrolled
and the per-node output row lives in registers. A production net with runtime
widths would tile the `H·W` matmul into shared memory instead (a small GEMM).

---

## 5. Numerical considerations

- **Precision.** All activations are FP32 — plenty for a 2-layer net, and what
  PyTorch defaults to (real training often drops to FP16/BF16 for speed).
- **Self-loop avoids divide-by-zero.** `deg[i]` always includes the self-loop, so
  it is ≥ 1 and `c_ij = 1/sqrt(deg_i·deg_j)` never divides by zero, even for an
  isolated atom.
- **Determinism.** Neighbor sums run in CSR order on both CPU and GPU, and there
  are no atomics, so the *only* possible divergence is the GPU contracting `a*b+c`
  into a single **fused multiply-add (FMA)** where the host used two rounded ops.
  That is a few ulp per layer (we measure ~`6e-8` total).
- **ReLU is branch-free.** `x>0 ? x : 0` compiles to a predicated select, so there
  is no warp divergence at the activation.

---

## 6. How we verify correctness

1. **CPU reference.** `gcn_predict_cpu()` runs the identical pipeline serially,
   calling the **same** `gcn.h` functions the kernels call (the `__host__
   __device__` single-source idiom, PATTERNS.md §2). So "CPU" and "GPU" are the
   same math in two execution models.
2. **Tolerance.** `main.cu` asserts `max_m |pred_cpu[m] - pred_gpu[m]| ≤ 1e-4`.
   The measured error (~`6e-8`, printed on stderr) is far below it — the slack
   only accounts for FMA contraction, and `1e-4` is itself far below the spread of
   the predictions, so the check certifies "same computation" honestly without
   pretending the floats are bit-identical (PATTERNS.md §4).
3. **Interpretable sample.** The five molecules have visibly different topologies
   (chains, rings, a star), so the predictions form a clear, reproducible ranking —
   a sanity check that the message passing actually depends on structure.
4. **Edge cases handled by construction.** Self-loops give every node ≥ 1 neighbor;
   the ragged last thread block is guarded (`if (i >= num_nodes) return;`); the
   loader validates the header, atom-count sum, and edge index ranges.

---

## 7. Where this sits in the real world

- **Training.** Real QSAR learns `W` by backpropagation against measured labels
  (MSE for regression, cross-entropy for classification). We ship *inference only*
  with seeded weights; adding an autograd training loop is the natural next step
  (and an exercise). Frameworks: PyTorch + **PyTorch Geometric** / **DGL**.
- **Better message passing.** Production models use **D-MPNN** (Chemprop — messages
  on *directed bonds*, which reduces tottering), **GAT** (attention-weighted
  neighbors instead of fixed `c_ij`), or **graph transformers** (Uni-Mol, with 3-D
  conformer geometry). All keep the aggregate-then-combine skeleton implemented
  here.
- **Featurization.** Real atom/bond features come from **RDKit** (element,
  degree, formal charge, hybridization, aromaticity, H-count, chirality; bond
  type, conjugation, ring membership). Our 6 features are a toy subset.
- **Scale & batching.** A virtual screen pushes 10⁵–10⁸ molecules through the net.
  Frameworks batch many small graphs into one big disconnected graph (exactly the
  global-CSR layout here) so a single kernel launch processes thousands of
  molecules — that is where the GPU decisively beats the CPU. On our 23-atom batch
  the GPU is launch-bound and *slower* (honest timing, PATTERNS.md §7); the point
  here is the *pattern* and the *exact* CPU↔GPU agreement, not a speed number.
- **Uncertainty.** Real deployments add ensembles or MC-Dropout to flag
  out-of-distribution molecules — important because a confident wrong ADMET
  prediction can waste a synthesis campaign.

---

## References (study, don't copy)

- Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional
  Networks*, ICLR 2017 — the GCN equation in §2.
- Gilmer et al., *Neural Message Passing for Quantum Chemistry*, ICML 2017 — the
  MPNN framework.
- Yang et al. (Chemprop), *Analyzing Learned Molecular Representations…*, 2019.
- See the project [`README.md`](README.md) "Prior art" for the toolkits.
