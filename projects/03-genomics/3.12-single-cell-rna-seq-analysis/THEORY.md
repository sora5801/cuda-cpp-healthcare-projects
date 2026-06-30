# THEORY — 3.12 Single-Cell RNA-seq Analysis

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use. All data here is synthetic._

This project is a **reduced-scope teaching version** (CLAUDE.md §13). The full
scRNA-seq pipeline is research-grade software (Scanpy, rapids-singlecell). We
implement the two steps a learner can fully follow *and verify exactly* —
**library-size normalization** and **exact k-nearest-neighbour (KNN) graph
construction** — because the KNN graph is, by the catalog's own deep dive, the
single most GPU-impactful step. §7 describes the full pipeline this slots into.

---

## 1. The science

A cell expresses its genome selectively: a T cell, a neuron, and a hepatocyte
carry the *same* DNA but transcribe very different sets of genes into mRNA.
**Single-cell RNA-seq (scRNA-seq)** measures, for thousands to millions of
individual cells, how many mRNA molecules of each gene were captured. The output
is a **count matrix** `X` of shape `[N cells × G genes]`, where `X[c][g]` is the
number of unique transcripts (UMIs) of gene `g` seen in cell `c`. Real matrices
are huge (10⁵–10⁷ cells × ~30 000 genes) and ~90% zeros (most genes are off in
most cells).

The biological question is: **which cells are alike?** Grouping cells by their
expression profiles reveals cell *types* and *states* — the foundation of cell
atlases, developmental trajectories, and disease studies. The computational
backbone of that grouping is a **neighbourhood graph**: connect each cell to its
`k` most similar cells. Clustering (Leiden/Louvain) and 2-D visualization
(UMAP/t-SNE) both run *on top of* this KNN graph. Build the graph well and the
rest follows.

Two nuisances must be removed first:

1. **Sequencing depth.** Some cells are simply sampled more deeply than others
   (more total counts), purely a technical artifact. Without correction, "deeply
   sequenced" looks like a biological signal.
2. **Heavy-tailed magnitudes.** A handful of genes dominate the raw counts;
   distances become hostage to them.

The standard fix is **counts-per-target normalization + log1p** (this project's
step 1), then KNN in that normalized space (step 2).

---

## 2. The math

**Inputs.** A raw count matrix `X ∈ ℕ^{N×G}`, a neighbour count `k`, and a
normalization target `S` (`target_sum`, e.g. `S = 10⁴` — "counts per 10k").

**Step 1 — normalization.** For cell `c` define its **library size** (total
counts)

```
T_c = Σ_g X[c][g]
```

Then each entry is rescaled to a common library size `S` and log-compressed:

```
Y[c][g] = log( 1 + (X[c][g] / T_c) · S )          (log1p of counts-per-S)
```

- Dividing by `T_c` removes depth (every cell now sums to `S` before the log).
- `log1p(x) = log(1+x)` is defined at `x=0` (a zero count maps to `0`) and tames
  the heavy tail. This is exactly Scanpy's `normalize_total` + `log1p`, the
  de-facto default. Units of `Y` are "log-normalized expression" (dimensionless).

**Step 2 — the KNN graph.** Treat each normalized cell as a point
`y_c ∈ ℝ^G` (row `c` of `Y`). For a query cell `q`, its neighbours are the `k`
cells minimizing Euclidean distance:

```
d(q, c) = || y_q − y_c ||₂ = sqrt( Σ_g (Y[q][g] − Y[c][g])² ),   c ≠ q
```

The KNN graph stores, per cell, the indices and distances of its `k` nearest
(excluding itself). We rank by the **squared** distance `Σ_g (·)²` — `sqrt` is
monotonic, so the ordering is identical, and skipping it keeps the arithmetic
exact (we take the root only for the human-readable report).

**Output.** `Y ∈ ℝ^{N×G}` (normalized) and two `[N×k]` arrays: neighbour indices
(nearest first) and their distances.

---

## 3. The algorithm

```
normalize(X) -> Y:
  for each cell c:                 # O(N·G)
     T_c = sum of row c
     for each gene g: Y[c][g] = log1p(X[c][g]/T_c · S)

knn(Y) -> graph:
  for each query cell q:           # O(N) queries
     init a length-k "best" list to +inf
     for each candidate cell c != q:    # O(N) candidates
        d2 = squared_distance(y_q, y_c) # O(G)
        insert (d2, c) into the top-k list (insertion sort, O(k))
     emit the sorted top-k
```

**Complexity.**
- Normalization: `O(N·G)` work, `O(1)` extra memory per cell.
- Exact (brute-force) KNN: `O(N² · G)` distance work + `O(N² · k)` for the
  inserts. The `N²` term is the wall: for `N = 10⁶`, `N²` is `10¹²` — this is the
  "O(n²) naive" the catalog deep dive calls out, and exactly where the GPU (and,
  at scale, *approximate* nearest neighbours) earns its keep.

**Parallel structure.** Both steps are **embarrassingly parallel over cells**:
each cell's normalized row depends only on its own counts; each query cell's
neighbour list depends only on reading the (shared, read-only) matrix. There is
no cross-thread dependency — no reduction across threads, no atomics. The work is
`O(N²·G)`; the *depth* (critical path) is just `O(N·G)` per query, run in
parallel across `N` queries.

---

## 4. The GPU mapping

This is the **"score one item vs N, each independent"** pattern
(docs/PATTERNS.md §1, exemplar `1.12` Tanimoto): one thread owns one query and
scans all candidates.

**Kernel 1 — `normalize_kernel` (one thread per cell).**
- Thread `c = blockIdx.x · blockDim.x + threadIdx.x` owns cell `c`.
- It reads its `G` raw counts from global memory, sums them in a **private
  register** (`cell_total`) — a per-thread reduction, no shared memory needed —
  then writes its `G` normalized values back. Pure map; no communication.

**Kernel 2 — `knn_kernel` (one thread per query cell).**
- Thread `q` owns query cell `q`.
- It keeps a length-`k` **top-list in registers/local memory** (`best_d`,
  `best_i`, sized to the compile-time `SC_MAX_K`, so no dynamic allocation inside
  the kernel) and scans all `N` candidate rows of `Y` from global memory,
  inserting better candidates with an `O(k)` insertion sort.
- No shared memory, no atomics: each query's list is independent.

**Launch configuration.** `block = 256` threads (a multiple of the 32-lane warp;
8 warps per block to hide global-memory latency; good occupancy on sm_75–sm_89).
`grid = ceil(N / 256)` blocks. Both kernels use the same geometry because both
have exactly `N` independent units of work.

```
        normalized matrix Y in global memory  ([N x G], row-major)
        +----+----+----+ ... +----+
 cell 0 | .. | .. | .. |     | .. |   <- read by EVERY query thread
 cell 1 | .. | .. | .. |     | .. |
   ...  |    |    |    |     |    |
 cell N | .. | .. | .. |     | .. |
        +----+----+----+ ... +----+

  grid:  block 0          block 1        ...
         [t0 t1 ... t255] [t0 t1 ... t255]
           |   |      |
           q=0 q=1   q=255   each thread q: scan all N rows, keep its own top-k
```

**Memory & bandwidth.** The kernel is **bandwidth-bound**: every query thread
streams the whole `N×G` matrix from global memory, so the dominant cost is `O(N²·G)`
reads. The teaching version reads straight from global memory. The standard
optimization (left as an exercise / described in THEORY of `1.12`) is to **tile**
the candidate matrix into **shared memory**: a block of 256 query threads
cooperatively loads a tile of candidate rows into shared memory once, then all
256 threads reuse it — cutting global traffic by the block size. We keep the
un-tiled version because it is the readable baseline that *motivates* tiling.

**No CUDA library here, on purpose.** This is a hand-rolled kernel so nothing is a
black box: the distance is a plain loop, the top-k is a visible insertion sort.
Production does the opposite (§7): cuML/Faiss provide the KNN as a library call.
We name what they compute but reimplement it didactically (CLAUDE.md §6.1.6).

---

## 5. Numerical considerations

- **Precision.** Counts are small non-negative integers; the matrix is stored in
  **FP32**. The two reductions that matter — the per-cell library-size sum and
  the per-pair squared distance — accumulate in **FP64** (`double`) inside the
  shared `scrna.h` helpers, then cast back to float. Double accumulation keeps
  the result well within FP32's ~7 significant digits.
- **No atomics, no race conditions.** Each thread writes only its own outputs
  (its normalized row; its `k` neighbour slots). There is no shared accumulator,
  so the float-summation-order problem that plagues atomic reductions (see `5.01`,
  `11.09`) simply does not arise here.
- **Determinism.** The CPU and GPU run the *same* `scrna.h` functions in the
  *same* order: the distance loop is `g = 0..G−1` on both sides, and candidates
  are scanned `c = 0..N−1` on both sides. So the floating-point rounding is
  identical and the results are bit-for-bit equal in practice.
- **Tie-breaking (the subtle determinism point).** Two candidates can be
  equidistant from a query. `sc_knn_insert` uses a **strict `<`** test and scans
  candidates in increasing index order, so among ties the **lower cell index**
  always wins and is placed first. This makes the neighbour *order* deterministic
  and identical on CPU and GPU — without it, an `≤` test or a different scan order
  could shuffle ties and the index comparison would spuriously "fail".
- **Empty cells.** A cell with zero total counts would divide by zero;
  `sc_normalize_entry` guards `T_c ≤ 0` and maps the row to the origin.

---

## 6. How we verify correctness

`src/reference_cpu.cpp` is an independent **serial** implementation: a single
readable loop that calls the same `scrna.h` math. `main.cu` runs both and checks:

- **Neighbour indices: exact (0 mismatches required).** The indices are integers
  produced by the same deterministic scan + tie-break, so they must match *every*
  position. This is the strong gate — an off-by-one in the distance, a wrong
  stride, or a botched top-k would change at least one index. (Tolerance class
  "exact integer", docs/PATTERNS.md §4 — same family as `1.12`, `3.01`, `11.09`.)
- **Normalized values & distances: `1e-5`.** These are floats. Because both sides
  run identical math in identical order they agree to the last bit in practice
  (the demo shows `max |norm| diff = 0.000e+00`); the small slack is honest
  headroom for a different host-compiler FMA contraction, not an expected gap.

**A second, stronger check — the science, not just CPU==GPU.** The synthetic
sample embeds **3 known cell types** (marker-gene blocks + random per-cell depth).
A correct pipeline must connect cells to same-type cells, so we report **KNN label
purity** = fraction of the `N·k` graph edges whose endpoints share a type. The
demo reports **100%** — every neighbour edge is within-type. That validates that
normalization actually removed depth and that the graph captures real structure,
not just that two implementations agree (docs/PATTERNS.md §6, like the recovered
cluster sizes in `11.09`).

---

## 7. Where this sits in the real world

Production scRNA-seq runs a longer pipeline; the catalog "Prior art" names the
tools. This project is the **honest, verifiable core** of two of its steps. The
full pipeline and how it differs:

| Step | This teaching version | Production (Scanpy / rapids-singlecell) |
|---|---|---|
| Normalize | counts-per-`S` + `log1p`, dense FP32 | same formula, on a **sparse** matrix (cuPy/cuSPARSE) |
| Feature selection | (omitted) | highly-variable-gene selection to ~2 000 genes |
| Dimensionality reduction | (omitted — KNN runs in full gene space) | **PCA** to ~50 components (cuML) before KNN |
| KNN graph | **exact brute-force**, `O(N²·G)` | **approximate** NN (Faiss-GPU / cuML, HNSW) → ~`O(N log N)` |
| Embedding | (omitted) | **UMAP** force-directed layout (cuML) |
| Clustering | (omitted) | **Leiden/Louvain** on the KNN graph (cuGraph) |
| Differential expression | (omitted) | negative-binomial GLM per gene |

The two big simplifications and their consequences:

1. **Brute-force, not approximate, KNN.** Real tools never compute all `N²`
   distances; an **ANN index** (Faiss/HNSWLIB, GPU) finds *approximate* neighbours
   in roughly `O(N log N)`, trading a tiny recall loss for orders of magnitude
   speed. We do the exact version precisely because it is *verifiable* against a
   CPU baseline — you cannot bit-compare a randomized ANN index. The exact kernel
   here is also the ground truth you would use to *measure* an ANN's recall.
2. **No PCA.** We run KNN in the full gene space; production first projects to
   ~50 PCA components, which denoises and makes the distance step `~600×` cheaper
   per pair. PCA itself is a dense linear-algebra solve — see flagship `2.06`
   (cuSOLVER eigensolve) for how that maps to the GPU.

`rapids-singlecell` reports **>20× speedups up to 20 M cells** precisely by moving
the KNN + UMAP steps onto cuML/cuGraph — the same two steps this project
isolates. The lesson to carry away: the GPU win in scRNA-seq is overwhelmingly
about the **neighbourhood graph**, which is the one piece we built by hand.

---

## References

- **Scanpy** — https://github.com/scverse/scanpy — the canonical CPU pipeline;
  read `normalize_total`, `log1p`, and `pp.neighbors` to see the exact steps we
  reimplement (and everything we omit).
- **rapids-singlecell** — https://github.com/scverse/rapids_singlecell — the
  drop-in GPU backend (cuPy/cuML/cuGraph). Study which steps it moves to the GPU
  and why KNN/UMAP dominate the speedup.
- **NVIDIA RAPIDS single-cell examples** — https://github.com/NVIDIA-Genomics-Research/rapids-single-cell-examples
  — benchmark notebooks up to 1M+ cells; good for seeing the scaling story.
- **ScaleSC** — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12321287/ — a GPU
  scRNA pipeline (20× speed, 20M cells on A100); useful for the memory-management
  ideas at scale.
- **Faiss** — https://github.com/facebookresearch/faiss — the GPU ANN library
  that replaces our brute-force KNN in production; read its HNSW/IVF docs to see
  what "approximate" buys.
- Data portals: Human Cell Atlas (https://www.humancellatlas.org/), 10x Genomics
  (https://www.10xgenomics.com/resources/datasets), CellxGene Census
  (https://cellxgene.cziscience.com/), NCBI GEO (https://www.ncbi.nlm.nih.gov/geo/).
