# 3.9 — Phylogenetic Likelihood / Tree Inference

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.9`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Given a DNA alignment and a few candidate evolutionary **trees**, this project
computes the **log-likelihood** of each tree — the log-probability that the
observed sequences arose by mutation along that tree's branches — and reports the
**maximum-likelihood** tree. The engine is **Felsenstein's pruning recursion**, the
exact calculation at the heart of RAxML, IQ-TREE, and MrBayes. The likelihoods of
different alignment **sites** (columns) are independent, so the GPU gives each site
its own thread: one teaching kernel evaluates a whole column's pruning recursion,
and the per-site results are summed into a per-tree total. On the committed
synthetic sample the program recovers the true tree the data were simulated under,
and the GPU total matches the CPU reference **bit-for-bit**.

## What this computes & why the GPU helps

Maximum-likelihood phylogenetic inference evaluates the Felsenstein pruning recursion—computing site likelihood at each internal node by multiplying branch transition probability matrices (4×4 or 20×20 per site, per node) up the tree—for millions of alignment columns and hundreds of tree search moves (NNI, SPR). For large trees (thousands of taxa, genome-scale alignments), the log-likelihood computation is the bottleneck and is embarrassingly parallel across alignment sites. Bayesian phylogenetics (MrBayes) runs thousands of MCMC steps each requiring full-tree likelihood evaluation; GPU acceleration reported 63× speedup vs. serial CPU by assigning each site to a thread. RAxML-NG and IQ-TREE GPU are active development targets.

**The parallel bottleneck:** evaluating one tree's likelihood means running the
pruning recursion **once per alignment site** and summing the results. With genome-
scale alignments (10^5–10^7 columns) and a tree search that re-evaluates the
likelihood for hundreds of rearrangements (NNI/SPR) or thousands of MCMC steps,
that **per-site recursion** is the overwhelming cost. It is also **embarrassingly
parallel across sites** — each column's likelihood is independent — so the GPU maps
**one site per thread** (the same mapping BeagleLib/MrBayes use, which reported
~63× over serial CPU). This project teaches that mapping on a small, verifiable
problem.

## The algorithm in brief

What this teaching version **implements**:

- **Felsenstein's pruning recursion** — a post-order sweep of the tree that builds
  each node's conditional-likelihood vector (CLV) from its children, giving the
  per-site likelihood in O(n_taxa) work.
- **K2P substitution model in closed form** — the 4×4 transition-probability matrix
  P(t) = exp(Qt) for the Kimura-2-parameter model has an analytic formula (no
  numerical matrix exponential), distinguishing transitions from transversions via
  a transition/transversion ratio `kappa` (kappa = 1 reduces to Jukes–Cantor).
- **Maximum-likelihood tree selection** — score a small set of candidate
  topologies (the true tree plus the wrong resolutions an **NNI** move explores)
  and pick the best.

What the **full** field adds (described in [THEORY.md](THEORY.md) "Where this sits
in the real world", not implemented here): general GTR / amino-acid (WAG, LG)
models via real matrix exponentiation, full **NNI/SPR** tree *search*, rate
heterogeneity (Γ categories), Bayesian **MCMC** (MrBayes), and bootstrap support.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/phylogenetic-likelihood-tree-inference.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/phylogenetic-likelihood-tree-inference.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\phylogenetic-likelihood-tree-inference.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: TreeBASE — curated phylogenetic alignments and trees (https://www.treebase.org/); SILVA rRNA database — large rRNA alignment for phylogenetics (https://www.arb-silva.de/); NCBI CDD — conserved domain alignments (https://www.ncbi.nlm.nih.gov/Structure/cdd/cdd.shtml); OpenTreeOfLife — aggregated phylogenetic data (https://opentreeoflife.github.io/).

## Expected output

Success looks like `demo/expected_output.txt`. The program computes the result on
both the **GPU** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`)
and asserts they agree within the documented tolerance — that agreement is the
correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
2. [`src/felsenstein.h`](src/felsenstein.h) — **the science**: the shared
   `__host__ __device__` K2P model + pruning recursion (`site_log_likelihood`).
   This single file runs on both the CPU and the GPU, so they compute identically.
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`reference_cpu.cpp`](src/reference_cpu.cpp)
   — the data model, the text loader, and the trusted serial driver.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-site idea.
5. [`src/kernels.cu`](src/kernels.cu) — the kernel (tree in constant memory,
   fixed-point atomic reduction) and the host wrapper.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

IQ-TREE 2 (https://iqtree.github.io/) — state-of-the-art ML tree inference (GPU extension in development); RAxML-NG (https://github.com/amkozlov/raxml-ng) — fast ML inference with GPU acceleration hooks; MrBayes (https://github.com/NBISweden/MrBayes) — Bayesian inference with CUDA-accelerated site likelihood; BeagleLib (https://github.com/beagle-dev/beagle-lib) — GPU-accelerated phylogenetic likelihood library used by MrBayes/BEAST.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent jobs + constant-memory broadcast + deterministic atomic reduction.**
Each alignment **site** is an independent job (one GPU thread), so this is the
same "score N independent items" pattern as flagship `1.12` (Tanimoto). Three
teaching points:

- The **tree** (its post-ordered nodes) is read by every thread but never written,
  so it lives in **`__constant__` memory** — the constant cache broadcasts a node
  to a whole warp in one transaction.
- The alignment is stored **column-major**, so thread *j* reads its site's column
  as a contiguous run → coalesced loads.
- The sum of per-site log-likelihoods is reduced with an **integer `atomicAdd`** on
  a fixed-point value (not a float atomic), so the total is **deterministic** and
  matches the CPU exactly (see [PATTERNS.md](../../../docs/PATTERNS.md) §3).

Production libraries (BeagleLib, used by MrBayes/BEAST) extend this to 4×4/20×20
matrix–vector products per site *per node*, use cuBLAS for general matrix
exponentiation, and partition large trees across multiple GPUs.

## Exercises

1. **Bigger problem.** Regenerate a longer alignment
   (`python scripts/make_synthetic.py --n-sites 50000 --seed 7`) and watch the
   GPU/CPU time ratio shift as the per-site work grows. Where does the GPU start
   to win?
2. **Jukes–Cantor vs. K2P.** Set `kappa = 1` in the sample header and re-run. The
   model degenerates to Jukes–Cantor (all substitutions equally likely); confirm
   the closed-form probabilities in `felsenstein.h` collapse accordingly.
3. **Add a fourth tree.** Hand-write another topology (a third NNI neighbour or a
   deliberately bad "comb" tree) and confirm ML still prefers the true one. How
   much does the log-likelihood drop?
4. **Per-thread CLV → shared memory.** For a fixed small `n_taxa`, move the CLV
   scratch from global memory into per-block shared memory (THEORY §GPU mapping
   discusses the trade-off). Measure the effect.
5. **Scale a tree-search loop.** Wrap the scorer in a loop that proposes random
   NNI moves and keeps improvements — a tiny hill-climbing ML *search*, the next
   step toward what RAxML/IQ-TREE do.

## Limitations & honesty

This is a deliberately **reduced-scope teaching version** (CLAUDE.md §13):

- **Model.** Only the **K2P** DNA model (4 states) with a single `kappa`. No
  general **GTR**, no amino-acid models (WAG/LG), no among-site rate variation
  (Γ categories), no invariant sites. Real tools fit these by numerical matrix
  exponentiation (eigendecomposition of Q); we use K2P's closed form on purpose so
  the math is transparent.
- **Tree *scoring*, not tree *search*.** We evaluate a fixed handful of candidate
  trees; we do **not** implement the NNI/SPR search loop or Bayesian MCMC that
  generate candidates. Those are described in THEORY and left as exercises.
- **Branch lengths are given, not optimised.** Production ML re-optimises branch
  lengths per topology; here they are fixed inputs.
- **Synthetic data.** The committed sample is simulated DNA with a known answer,
  labeled synthetic everywhere. It demonstrates the method; it is **not** real
  sequence data and supports **no** biological or clinical conclusion (CLAUDE.md §8).
- **Determinism over speed.** The CLV scratch lives in global memory and the
  per-launch `cudaDeviceSynchronize` in `CUDA_CHECK_LAST` is kept for legibility;
  both would change in a throughput build.
