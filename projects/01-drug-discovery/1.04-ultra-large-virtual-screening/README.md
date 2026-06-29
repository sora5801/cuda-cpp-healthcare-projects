# 1.4 — Ultra-Large Virtual Screening

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.4`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A virtual-screening campaign takes **one drug target** and a **giant library of
candidate molecules** and ranks the library to find the few that are worth
synthesizing and testing. Modern make-on-demand libraries are enormous (Enamine
REAL: >6 billion compounds), so the screen must be cheap per molecule and
massively parallel. This project builds the **reduced-scope teaching core** of
such a campaign: for each ligand we run a fast **drug-likeness filter cascade**
(Lipinski's Rule of Five + Veber rules) and then a deterministic **surrogate
docking score** (pharmacophore-feature overlap + property complementarity), and
we return the **top-K hits**. Every ligand is independent, so we put **one GPU
thread on each ligand** — the textbook embarrassingly-parallel pattern that is
the literal engine of billion-compound campaigns.

## What this computes & why the GPU helps

Modern make-on-demand chemical libraries (Enamine REAL: >6 billion compounds,
ZINC: ~2 billion) make exhaustive docking computationally prohibitive on CPUs.
GPU-accelerated screening batches thousands of ligands at once on a single GPU,
and ML/cheap **surrogate** filters cut the number of expensive full-docking
evaluations dramatically (HASTEN/REINVENT reach ~90% recall of the true top-1000
after scoring ~1% of the library). The Summit campaign against COVID-19 docked
>1 billion compounds with AutoDock-GPU.

**The parallel bottleneck** is the *per-ligand scoring* loop: the same function
(filter + score) is evaluated independently on every one of N ligands, and N is
in the billions. That is `O(N)` independent work with no cross-ligand
dependencies — a perfect fit for the GPU. We parallelize across the **ligand
dimension N** (one thread per ligand, a grid-stride loop for libraries larger
than the grid), keep the shared **target in constant memory**, and accumulate
nothing across threads (so no atomics, no races). This project teaches that
mapping with a transparent surrogate score; full physics-based docking is
described honestly in [THEORY.md](THEORY.md) §"Where this sits in the real world".

## The algorithm in brief

- **Filter cascade (stage 1):** reject any ligand that violates **Lipinski's
  Rule of Five** (MW ≤ 500, logP ≤ 5, HBD ≤ 5, HBA ≤ 10) or the **Veber** rules
  (rotatable bonds ≤ 10, PSA ≤ 140 Å²). Cheap; saves the scoring budget.
- **Surrogate dock score (stage 2):** for survivors, `score = BASE +
  feature_weight · popcount(ligand.feat & target.feat) − property-mismatch
  penalties`. The `feat & target` + `popcount` is the **same bit-overlap motif**
  as Tanimoto search (project 1.12).
- **Top-K:** report the highest-scoring survivors (ties broken by lower index for
  determinism).

The full catalog algorithm list (GPU-batched LGA/BFGS docking, Bayesian active
learning, GNN surrogates, pharmacophore/shape pre-filters, ADMET cascades) is
discussed in THEORY; this teaching version implements the **filter cascade +
surrogate scoring + top-K** slice end-to-end. See
[THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/ultra-large-virtual-screening.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/ultra-large-virtual-screening.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\ultra-large-virtual-screening.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/ligands_sample.txt` (1 target vs
64 synthetic ligands), prints the **survivor count** and the **top-5 hits**, the
**GPU-vs-CPU agreement** check, and a timing line.

## Data

- **Sample (committed):** `data/sample/ligands_sample.txt` — 1 target + 64
  **synthetic** ligands (4 engineered "designed binders", 8 deliberate
  filter-fails, the rest decoys). Deterministic; scores are chemically
  meaningless.
- **Full dataset:** generate 2-D descriptors + pharmacophore features from real
  libraries with **RDKit** — `scripts/download_data.ps1` / `.sh` print the recipe.
- **Library-scale synthetic set:** `python scripts/make_synthetic.py --n 1000000`.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: Enamine REAL (>6B synthesizable compounds,
<https://enamine.net>); ZINC20 (<https://zinc20.docking.org>); ChEMBL
(<https://www.ebi.ac.uk/chembl/>); ExCAPE-DB
(<https://solr.ideaconsult.net/search/excape/>).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program scores all `N` ligands on the **GPU** (`src/kernels.cu`) and on a **CPU
reference** (`src/reference_cpu.cpp`) and asserts they agree. Because both call
the *same* `__host__ __device__` `score_ligand()` (in `src/screen_core.h`) and
the score is **integer fixed-point**, the two agree **exactly** —
`mismatches = 0`, tolerance **zero** (the strongest possible check). The four
engineered binders (feature mask `0x0000A5B3`, on-target size/logP) top the
ranking, which validates the science as well as CPU==GPU.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the library, runs CPU + GPU, verifies, prints survivors + top-K.
2. [`src/screen_core.h`](src/screen_core.h) — **the shared `__host__ __device__` core**: `Ligand`/`Target` structs, the filter cascade, and the surrogate score (the one true math both sides run).
3. [`src/reference_cpu.h`](src/reference_cpu.h) — the data model (`LigandLibrary`) + loader/reference prototypes.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the constant-memory / grid-stride idea.
5. [`src/kernels.cu`](src/kernels.cu) — the kernel (one thread per ligand) + host wrapper.
6. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline + the text loader.
7. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **AutoDock-GPU** (<https://github.com/ccsb-scripps/AutoDock-GPU>) — the core CUDA docking engine used in the Summit billion-compound campaigns; study its LGA/BFGS launch and per-ligand batching (the *real* stage-2 this project's surrogate stands in for).
- **Uni-Dock** (<https://github.com/dptech-corp/Uni-Dock>) — high-throughput GPU docking with batch input; learn how thousands of ligands are kept co-resident.
- **DiffDock** (<https://github.com/gcorso/DiffDock>) — a diffusion model for blind docking; the ML-surrogate frontier.
- **gpusimilarity** (<https://github.com/schrodinger/gpusimilarity>) — GPU fingerprint similarity for rapid pre-screening; the closest analogue of this project's bit-overlap term and of project 1.12.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Independent jobs (one thread per ligand) · **constant memory** for the shared
target (broadcast warp-wide) · **grid-stride loop** over the library · a shared
`__host__ __device__` per-ligand core so CPU and GPU run identical math ·
**integer/fixed-point** scoring for exact, deterministic results · top-K on the
host (the production path keeps the reduction on-device with `cub::DeviceRadixSort`
or `thrust::sort_by_key`). Catalog also lists texture memory for docking-grid
lookups and warp-parallel GA evaluation — those belong to the full docking
engine described in THEORY.

## Exercises

1. **Top-K on the GPU.** Replace the host `partial_sort` with a Thrust
   `sort_by_key` or `cub::DeviceRadixSort` over the device scores. Does it help
   at `n = 64`? At `n = 1,000,000` (`make_synthetic.py --n 1000000`)?
2. **Relax the cascade.** Real Lipinski filtering allows **one** rule violation.
   Change `passes_filter_cascade` to count violations and reject only when `≥ 2`.
   How many more ligands survive?
3. **Active learning.** Score a random 1% of the library, train a tiny linear
   surrogate on `(descriptors → score)`, predict the rest, and measure the recall
   of the true top-K — the HASTEN/REINVENT idea, in miniature.
4. **Warp-per-ligand.** For a much heavier scoring function, assign one *warp*
   per ligand and reduce with `__shfl_down_sync`. When does that beat
   one-thread-per-ligand?
5. **Float vs. fixed-point.** Re-implement the surrogate score in `float` and
   watch the CPU/GPU agreement degrade from exact to `~1e-6` — a concrete lesson
   in why we chose integer arithmetic (PATTERNS.md §3–4).

## Limitations & honesty

- **This is NOT real docking.** Stage 2 is a transparent, cheap **surrogate**
  (feature overlap + property fit), not a physics-based pose search. Real
  campaigns run a genetic-algorithm + local-search docking engine (AutoDock-GPU's
  LGA/BFGS) per ligand; that is genuinely research-grade and out of scope for one
  teaching kernel (CLAUDE.md §13). THEORY explains the gap.
- The committed sample is **synthetic** and labelled as such everywhere; scores
  carry **no chemical meaning**.
- Top-K is computed on the host (fine here — the bottleneck, scoring, is on the
  GPU). At true library scale you would keep the reduction on-device.
- We load the whole library into device memory; production tools stream/shard
  libraries that exceed GPU memory and may pre-filter with shape/pharmacophore
  screens before any scoring. No ADMET/toxicity model is included.
