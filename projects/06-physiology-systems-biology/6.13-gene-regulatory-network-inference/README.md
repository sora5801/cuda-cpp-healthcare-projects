# 6.13 — Gene Regulatory Network Inference

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.13`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A **gene regulatory network (GRN)** is a wiring diagram of which genes control
which: an edge `A–B` means gene `A`'s activity statistically depends on gene
`B`'s (a transcription factor driving a target, say). This project infers that
graph from an expression matrix using the classic **ARACNE** recipe:
score every gene *pair* by their **mutual information (MI)**, then apply the
**Data Processing Inequality (DPI)** to strip out *indirect* edges. Every pair is
independent, so the O(G²) MI computation maps cleanly onto the GPU — one gene
pair per thread — exactly the "many independent scoring jobs" pattern of the
Tanimoto flagship (`1.12`). On a tiny synthetic dataset with a **planted**
network, you get to watch the algorithm both recover the true edges and correctly
discard the spurious ones.

## What this computes & why the GPU helps

Infers the directed causal graph of transcription-factor→gene interactions from
single-cell RNA-seq (scRNA-seq) data. State-of-the-art methods use mutual
information, GENIE3 random forests, or neural-ODE formulations. Computing pairwise
mutual information across 20 000 genes needs O(N²) comparisons — a
~200-million-pair problem, and each pair is independent.

**The parallel bottleneck:** the pairwise MI matrix. For `G` genes there are
`G·(G−1)/2` unordered pairs, and each pair scans all `S` samples to build a joint
histogram — an `O(G² · S)` computation that utterly dominates the runtime and is
*embarrassingly parallel* (no data dependence between pairs). We give **one pair
its own GPU thread**; each thread privately tallies its `B×B` histogram (integer
counting → deterministic) and evaluates the MI. A second `O(G³)` kernel then does
the DPI prune, one candidate edge per thread. The discretization is integer, so
the GPU and CPU agree to machine precision.

## The algorithm in brief

- **Discretize** each gene's expression into `B = 8` equal-width bins (per-gene range).
- **Mutual information** for each pair `(i,j)`: build the `B×B` joint histogram
  over the `S` samples, then `I = Σ p(a,b)·ln[p(a,b)/(p(a)p(b))]` in nats.
- **Data Processing Inequality**: in any triangle `i–k–j`, if `I(i,j)` is the
  strictly-weakest edge, `(i,j)` is most likely *indirect* (mediated by `k`) and
  is pruned. An MI significance floor removes chance correlations first.
- The catalog also lists GENIE3 random forests, PANDA message-passing, neural-ODE
  dynamics, scVI VAEs, LASSO/elastic-net, and Granger causality — we implement the
  **ARACNE MI + DPI** path as the most self-contained, exactly-verifiable teaching
  version; [THEORY.md](THEORY.md) §"real world" contrasts the others.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/gene-regulatory-network-inference.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/gene-regulatory-network-inference.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\gene-regulatory-network-inference.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the result, shows the
GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/` — a tiny, offline input so the demo runs
  with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (documented, idempotent).
- **Provenance & license:** see [data/README.md](data/README.md).

The committed sample is **synthetic** with a **planted** ground-truth network
(`TF→A→B`, `TF→C`, `D→E`, plus noise genes) so the demo has a verifiable answer.
Catalog dataset notes: Gene Expression Omnibus (GEO) — scRNA-seq datasets
(<https://www.ncbi.nlm.nih.gov/geo/>); ENCODE TF-binding ChIP-seq
(<https://www.encodeproject.org>); BEELINE benchmark GRN datasets
(<https://github.com/Murali-group/BEELINE>); Human Cell Atlas
(<https://www.humancellatlas.org>).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): after
MI + DPI, exactly **four direct edges** — `TF–A`, `TF–C`, `D–E`, `A–B` — the true
network, with the spurious `TF–B`, `A–C`, `B–C` correlations pruned. The program
computes MI on both the **GPU** (`src/kernels.cu`) and a **CPU reference**
(`src/reference_cpu.cpp`) and asserts (a) the MI matrices agree within `1e-9` and
(b) the pruned edge sets are **bit-identical**. That double check is the
correctness guarantee; the stderr `[verify]` line typically shows
`max_abs_err ≈ 2.2e-16` (machine precision).

## Code tour

Read in this order:

1. [`src/grn.h`](src/grn.h) — the shared `__host__ __device__` core: the binning
   rule and the one true `mi_from_joint()` formula both sides call.
2. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`.cpp`](src/reference_cpu.cpp) —
   the data model, loader, and trusted serial baseline (discretize → MI → DPI).
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
5. [`src/kernels.cu`](src/kernels.cu) — the three kernels (discretize, MI, DPI)
   and the host wrapper.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

BEELINE GRN benchmark (https://github.com/Murali-group/BEELINE) — benchmarking framework for GRN inference methods; scVI (https://github.com/scverse/scvi-tools) — deep generative models for scRNA-seq on GPU via PyTorch; torchdiffeq (https://github.com/rtqichen/torchdiffeq) — GPU neural ODE for dynamics inference; Scanpy (https://github.com/scverse/scanpy) — scRNA-seq analysis with GPU-accelerated backends (rapids-singlecell).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent per-pair scoring** (PATTERNS.md §1, exemplified by the Tanimoto
flagship `1.12`): the `G·(G−1)/2` gene pairs are independent, so we map **one pair
→ one thread**, flatten the strict upper triangle to a 1-D pair index, and use a
grid-stride loop to cover any pair count. Each thread builds its `B×B` joint
histogram in **private local memory** (integer counting → order-independent →
deterministic, no atomics), then calls the shared `mi_from_joint()`. The DPI prune
is a second one-thread-per-edge kernel over the finished MI matrix. Discretization
is itself a one-thread-per-gene kernel.

> The catalog also floats *cuBLAS for a correlation matrix* and *Thrust for
> per-gene ranks*. Those help a *correlation/rank-MI* variant; the histogram-MI
> path here needs neither, keeping the arithmetic exactly integer-deterministic
> and CPU↔GPU bit-comparable. Only `cudart` is linked. THEORY.md §"GPU mapping"
> discusses when the cuBLAS tiled-GEMM route wins (dense correlation at scale).

## Exercises

1. **Adaptive binning.** Replace equal-width bins with equal-*frequency*
   (quantile) bins so each bin holds ≈`S/B` samples. Does it sharpen the MI of the
   true edges? (Hint: sort each gene once; watch that CPU and GPU still match.)
2. **Miller–Madow correction.** The plug-in MI is biased upward on small samples.
   Add the `(B_xy − B_x − B_y + 1)/(2S)` bias correction in `grn.h` and see the
   noise-gene MI shrink toward zero.
3. **Bootstrap edge confidence.** Resample the `S` cells with replacement `R`
   times (one RNG per thread, per `5.01`) and report the fraction of bootstraps in
   which each edge survives — a poor-man's ARACNE p-value.
4. **Shared-memory tiling.** For large `G`, cache a tile of the discretized matrix
   in shared memory so a block of pairs reuses it (the catalog's "one-tile-per-
   gene-pair block" idea). Measure the bandwidth saving.
5. **Directionality.** MI is symmetric, so this recovers an *undirected* skeleton.
   Add a time-lagged (Granger-style) MI `I(A_t ; B_{t−1})` to orient edges.

## Limitations & honesty

- **Reduced-scope teaching version.** This is the ARACNE MI+DPI path only — not
  GENIE3, PANDA, neural-ODE, or scVI (all named in the catalog and sketched in
  THEORY §"real world"). It infers an **undirected** skeleton; real GRN inference
  adds direction, sign, and confidence.
- **Synthetic data.** The sample is generated with a known linear-Gaussian model
  (`scripts/make_synthetic.py`) purely so the answer is checkable. Real scRNA-seq
  is sparse, zero-inflated, and confounded by cell type and the cell cycle — MI
  estimation on it is far harder and the "true" network is unknown.
- **Fixed equal-width binning** (`B=8`) is the simplest estimator and is biased on
  small samples; production tools use adaptive kernels or B-splines (Exercises 1–2).
- **O(G³) DPI** is fine for tens–hundreds of genes; genome-scale (20k genes) needs
  sparsification and tiling. **Not for any clinical or diagnostic use.**
