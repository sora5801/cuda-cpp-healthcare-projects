# 3.12 — Single-Cell RNA-seq Analysis

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.12`
>
> _Educational only — not for clinical use (see CLAUDE.md §8). All data is synthetic._

## Summary

Single-cell RNA-seq (scRNA-seq) measures, for each of thousands to millions of
individual cells, how many mRNA molecules of each gene were captured — a giant
mostly-zero **count matrix**. To find which cells are alike (the basis of cell
atlases and disease studies), the standard pipeline **normalizes** the counts,
then builds a **k-nearest-neighbour (KNN) graph** connecting each cell to its
most similar cells; clustering and 2-D visualization run on top of that graph.
This project is a **reduced-scope teaching version** that implements the two
steps a learner can fully follow and verify exactly — *library-size normalization
+ log1p* and *exact brute-force KNN* — on the GPU, the KNN step being the one the
production deep dive flags as the biggest GPU win. It runs offline on a tiny
synthetic 30-cell sample with 3 embedded cell types and recovers them at 100%
neighbour purity.

## What this computes & why the GPU helps

Single-cell RNA-seq (scRNA-seq) produces count matrices for tens of millions of cells × 30 k genes; downstream analysis involves normalisation, highly variable gene selection, PCA, k-nearest-neighbour graph construction (O(n²) naive, accelerated by approximate nearest neighbours), UMAP / t-SNE embedding, Leiden/Louvain clustering, and differential expression. rapids-singlecell (scverse, 2024) replaces Scanpy's NumPy/SciPy backend with cuPy, cuML, and cuGraph equivalents, achieving >20× speedup for datasets up to 20 M cells. The KNN graph construction and UMAP optimisation are the most GPU-impactful steps, turning hours into minutes.

**The parallel bottleneck:** the **KNN graph**. Comparing every cell to every
other cell is `O(N²·G)` — for a million cells that is `10¹²` distance
evaluations, exactly the "O(n²) naive" wall the deep dive names. Every query
cell's neighbour search is independent, so we give **each query cell its own GPU
thread** (the "score one item vs N, each independent" pattern,
docs/PATTERNS.md §1). Normalization is even simpler — one thread per cell. We
deliberately build the **exact** graph (not the approximate Faiss/HNSW index real
tools use) because it is verifiable bit-for-bit against a CPU baseline; see
[THEORY.md §7](THEORY.md) for the production upgrade path.

## The algorithm in brief

- **Step 1 — normalize.** For each cell, divide its counts by its library size
  (total counts), scale to a fixed `target_sum`, and take `log1p`
  (= counts-per-10k + log, the Scanpy default). Removes sequencing-depth artifact.
- **Step 2 — KNN graph.** For each query cell, scan all cells, compute Euclidean
  distance in the normalized space, and keep the `k` nearest (excluding itself)
  via a length-`k` insertion sort. Deterministic tie-break (lowest index wins).

Production additionally does highly-variable-gene selection, PCA, then
*approximate* KNN, UMAP, and Leiden clustering — see the table in
[THEORY.md §7](THEORY.md). The full science → math → algorithm → GPU-mapping
derivation is in [THEORY.md](THEORY.md).

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/single-cell-rna-seq-analysis.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/single-cell-rna-seq-analysis.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\single-cell-rna-seq-analysis.sln /p:Configuration=Release /p:Platform=x64
```

Both `Release|x64` and `Debug|x64` build with zero warnings. The project links
only `cudart_static.lib` — the kernels are hand-rolled, so there is no extra CUDA
library to install.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/scrna_sample.txt`, prints the KNN
graph and the GPU-vs-CPU agreement check, and prints a timing line (on stderr).

## Data

- **Sample (committed):** `data/sample/scrna_sample.txt` — a tiny **synthetic**
  count matrix (30 cells × 18 genes, 3 cell types) so the demo runs offline with
  zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print pointers (real
  scRNA-seq matrices arrive as `.h5ad`/`.mtx`/10x HDF5 and need Scanpy to export a
  slice into this project's text format).
- **Provenance & license:** see [data/README.md](data/README.md). Regenerate the
  sample with `python scripts/make_synthetic.py`.

Catalog dataset notes: Human Cell Atlas — multi-organ scRNA-seq compendium (https://www.humancellatlas.org/); 10× Genomics public datasets (https://www.10xgenomics.com/resources/datasets); CellxGene Census — 50 M+ cells (https://cellxgene.cziscience.com/); NCBI GEO — thousands of scRNA-seq studies (https://www.ncbi.nlm.nih.gov/geo/).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): the
30-cell KNN graph (each cell's neighbour indices, all same-type), a couple of
normalized values, and `KNN label purity = 100.00%`. The program computes the
result on both the **GPU** (`src/kernels.cu`) and a **CPU reference**
(`src/reference_cpu.cpp`) and asserts they agree:

- neighbour **indices** match **exactly** (0 mismatches) — they are integers from
  the same deterministic scan + tie-break;
- normalized values and distances match within **`1e-5`** (in practice the
  normalized matrices are bit-identical — `max |norm| diff = 0`).

That agreement, plus the 100% label purity recovering the embedded cell types, is
the correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the matrix, runs CPU + GPU, verifies, reports.
2. [`src/scrna.h`](src/scrna.h) — **the shared `__host__ __device__` math**:
   normalize one entry, squared distance, top-k insert, per-cell KNN. Read this to
   understand why CPU and GPU agree exactly.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the two kernels (normalize, KNN) and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline + loader.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

rapids-singlecell (https://github.com/scverse/rapids_singlecell) — drop-in GPU Scanpy replacement, cuPy/cuML/cuGraph; NVIDIA RAPIDS single-cell examples (https://github.com/NVIDIA-Genomics-Research/rapids-single-cell-examples) — benchmark notebooks up to 1 M+ cells; ScaleSC (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12321287/) — GPU scRNA pipeline, 20× speed, 20 M cells on A100; Scanpy (https://github.com/scverse/scanpy) — CPU reference with GPU-aware backends.

What to learn from each: **Scanpy** for the exact normalize/neighbors steps we
reimplement; **rapids-singlecell** for which steps move to the GPU and why KNN
dominates; **Faiss** (https://github.com/facebookresearch/faiss) for the
approximate-NN index that replaces our brute force at scale. Study these to learn
the production approach; **do not copy code wholesale** — reimplement didactically
and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**One thread per independent unit of work**, twice: one thread per *cell* for
normalization (a pure map, private register reduction for the library size), and
one thread per *query cell* for KNN (scan all candidates, keep a private length-`k`
top list — the "score one item vs N, each independent" pattern of flagship `1.12`).
No shared memory, no atomics, no library calls — a clean, readable baseline. The
natural next optimization (shared-memory tiling of the candidate matrix) is left
as an exercise.

> The catalog's production pattern — cuPy sparse GEMM; cuML PCA/UMAP/KNN; cuGraph
> Leiden/Louvain; Faiss-GPU HNSW for ANN; cuDF; multi-GPU Dask — is described in
> [THEORY.md §7](THEORY.md). This teaching version isolates the KNN step and does
> it exactly, by hand.

## Exercises

1. **Scale it.** Generate a bigger synthetic set
   (`python scripts/make_synthetic.py --cells 4000 --genes 48 --k 10`) and watch
   the GPU kernel time grow far slower than the CPU's — the `O(N²)` story made
   visible. At what `N` does the GPU overtake the CPU on your card?
2. **Shared-memory tiling.** Rewrite `knn_kernel` so each block cooperatively
   loads tiles of the candidate matrix into `__shared__` memory and all threads in
   the block reuse them. Compare the kernel time to the un-tiled version (this is
   the single biggest win available here; see THEORY §4).
3. **Add PCA.** Reduce the `G` genes to ~8 principal components before KNN (use
   flagship `2.06`'s cuSOLVER eigensolver as a guide). Confirm label purity stays
   high while the distance step gets cheaper.
4. **Cosine instead of Euclidean.** Swap `sc_sqdist` for `1 − cosine similarity`
   in `scrna.h` (a common scRNA-seq choice) and see how the neighbour graph
   changes. Keep the CPU/GPU parity intact.
5. **Symmetrize the graph.** Turn the directed KNN into an undirected adjacency
   (edge `q–c` if either is a neighbour of the other) and report the degree
   distribution — the input a Leiden clusterer would consume.

## Limitations & honesty

- **Reduced scope (CLAUDE.md §13).** This builds two steps of a much larger
  pipeline. It omits highly-variable-gene selection, PCA, UMAP, Leiden clustering,
  and differential expression — all described in [THEORY.md §7](THEORY.md).
- **Exact, not approximate, KNN.** Real tools use an approximate NN index
  (Faiss/HNSW) for `~O(N log N)`; we compute all `N²` distances on purpose,
  because the exact result is what you can verify bit-for-bit (and what you would
  use to *measure* an ANN's recall). It does not scale to millions of cells as-is.
- **Dense + small panels.** The matrix is dense FP32 and capped at
  `SC_MAX_GENES = 64` genes / `SC_MAX_K = 16` neighbours so device threads can use
  fixed-size stack arrays. Real matrices are sparse and ~30 000 genes wide.
- **Synthetic data.** The committed sample is generated, labeled synthetic
  everywhere, and carries **no clinical meaning**. Nothing here may inform a real
  medical decision.
- **Timing is a teaching artifact.** For 30 cells the GPU is *slower* than the CPU
  (launch overhead dominates a tiny `O(N²)`); the GPU's edge appears only at real
  scale. The reported ms is never a benchmark claim (CLAUDE.md §12).
