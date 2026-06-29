# 1.20 — Reaction Yield / Retrosynthesis Scoring

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.20`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

> **Reduced-scope teaching version (CLAUDE.md §13).** The research-grade project
> is a transformer/GNN over reaction SMILES wrapped in a Monte-Carlo-tree-search
> planner. That is a deep-learning system, not a single CUDA kernel. This project
> isolates and teaches **the one massively parallel step inside it** — *scoring a
> whole batch of candidate routes at once* — with a transparent, hand-written
> yield model instead of a black-box neural net. The full pipeline is described
> in [THEORY.md](THEORY.md) → "Where this sits in the real world".

## Summary

A retrosynthesis planner breaks a target molecule down into purchasable building
blocks through a sequence of known reactions; each such sequence is a **route**.
Because a planner explores a tree of possibilities, it produces *huge numbers* of
candidate routes and must **score** each one — essentially "how likely is this
route to actually work in the lab?" This project computes that score for a whole
batch of routes **in parallel on the GPU**: each route gets its own thread, its
per-step yields are multiplied together and weighted by how available its
building blocks are, and the best routes are ranked. The per-route math is shared
verbatim between the CPU reference and the GPU kernel, so the two agree and the
demo is a genuine correctness check.

## What this computes & why the GPU helps

A route of `k` reaction steps succeeds end-to-end only if **every** step
succeeds. Under a teaching independence assumption, its success probability is
the **product of the per-step yields**, times a building-block **availability**
bonus that rewards routes ending in cheap, in-stock starting materials
(AiZynthFinder applies exactly this kind of stock reward). Each per-step yield is
predicted from a few features of the reaction (`template_prior`,
`precedent_count`, `condition_penalty`, `selectivity`) by a small logistic model
— our transparent stand-in for the production transformer/GNN.

**The parallel bottleneck:** scoring routes. A planner emits **millions** of
candidate routes, and each route's score depends only on its own steps — there
are no dependencies between routes. That is embarrassingly parallel: assign one
GPU thread per route and the whole batch is scored in one launch. The shared
logistic model lives in **constant memory** (every thread reads the same handful
of weights), and a **grid-stride loop** lets one modest grid cover a batch of any
size. This is the same "independent jobs · constant-memory query" pattern as
[1.12 Tanimoto](../1.12-molecular-fingerprint-similarity-search) and
[12.01 spectral search](../../12-omics-data-processing/12.01-mass-spectrometry-proteomics-search).

## The algorithm in brief

- **Per-step yield** — `yield = sigmoid(w · x + b)` over the 4 step features
  (a logistic model; in production a transformer/GNN).
- **Per-route score** — `score = (∏ step_yields) × availability` (chain of
  independent yields × stock bonus).
- **Batched scoring** — one GPU thread scores one route; grid-stride loop covers
  the whole batch (the parallel step inside an MCTS planner's rollout/scoring).
- **Rank** — deterministic top-K (ties broken by lower index).

The catalog also lists transformer attention, sequence-to-sequence reaction
prediction, MCTS planning, and graph-to-graph transformation; those build the
*routes and features*, and are summarized in [THEORY.md](THEORY.md). This project
implements the **scoring** stage end-to-end.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/reaction-yield-retrosynthesis-scoring.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/reaction-yield-retrosynthesis-scoring.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\reaction-yield-retrosynthesis-scoring.sln /p:Configuration=Release /p:Platform=x64
```

The project links only `cudart_static.lib` (the CUDA runtime); no extra CUDA
library is needed because the kernel is plain arithmetic plus `expf`.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/routes_sample.txt`, prints the
top-5 routes, shows the GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/routes_sample.txt` — 24 **synthetic**
  candidate routes so the demo runs with zero downloads. Route 0 is engineered to
  be the best, so its #1 ranking is a built-in sanity check.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print the recipe (atom-
  mapped reactions → planner → per-step features) and the source links.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: USPTO-50k — 50k atom-mapped reactions (https://github.com/connorcoley/rexgen_direct); Reaxys/CAS reaction databases (commercial); Open Reaction Database (ORD) — open-access reaction data (https://open-reaction-database.org); USPTO-MIT — 479k reactions (https://github.com/wengong-jin/nips17-rexgen).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): the
top-5 most synthesizable routes (with `route[0]` first, by construction) and a
`RESULT: PASS` line. The program scores every route on both the **GPU**
(`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) — both
calling the same `route_score()` in `src/route_score.h` — and asserts they agree
within `1e-6`. They are not bit-identical: a few-times-`1e-8` difference comes
from single-precision `expf`/FMA rounding diverging between host and device
(`THEORY.md` → "Numerical considerations").

## Code tour

Read in this order:

1. [`src/route_score.h`](src/route_score.h) — **the one true scoring formula**
   (`__host__ __device__`, shared by CPU and GPU). Start here.
2. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports top-K.
3. [`src/reference_cpu.h`](src/reference_cpu.h) — the `RouteSet` data model + loader prototype.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the file loader and the trusted serial baseline.
5. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
6. [`src/kernels.cu`](src/kernels.cu) — the kernel (one thread per route) + host wrapper + constant memory.
7. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **Molecular Transformer** (<https://github.com/pschwllr/MolecularTransformer>) —
  GPU transformer for reaction-product prediction; study how reaction SMILES are
  tokenized and how beam search decodes products.
- **AiZynthFinder** (<https://github.com/MolecularAI/aizynthfinder>) — MCTS-based
  retrosynthesis planner; study its route scoring and the **in-stock** check that
  inspired our `availability` factor.
- **ASKCOS** (<https://github.com/ASKCOS/ASKCOS>) — full synthesis-planning
  platform; study how template application + scoring fit together in a pipeline.
- **Chemformer** (<https://github.com/MolecularAI/Chemformer>) — BART-based
  reaction model; study pre-training/fine-tuning for reaction tasks.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent jobs · constant-memory shared model · grid-stride loop.** One GPU
thread scores one candidate route; the shared logistic weights sit in constant
memory (broadcast warp-wide by the constant cache); a grid-stride loop lets a
fixed-size grid cover a batch of any size. The catalog's full pattern note
(cuDNN attention, FP16, batched beam search, parallel MCTS rollouts with batched
transformer scoring) describes the *production* system — this project implements
the **batched-scoring** core of it without a neural-net dependency.

## Exercises

1. **Log-domain scoring.** Replace the product of yields with a **sum of
   `log(yield)`** (and keep a running max for numerical stability). Why is the log
   form preferred when routes get long? Does the ranking change?
2. **Top-K on the GPU.** Right now `main.cu` ranks on the host. Implement a GPU
   top-K (e.g. a per-block reduction into a small heap, or use CUB's
   `DeviceRadixSort`) and compare timings as `n` grows.
3. **Bigger batch, real timing.** Generate `--n 1000000` with
   `make_synthetic.py` and watch the GPU/CPU timing gap open up (the tiny sample
   is launch-bound; see [THEORY.md](THEORY.md) → "honest timing").
4. **Richer model.** Add features (e.g. a per-step `cost` or `green-chemistry`
   score) and extend `NUM_FEATURES`; confirm CPU and GPU still agree.
5. **FP64 variant.** Switch the scoring to `double` and observe the max error
   shrink toward `0` — evidence that the residual error really is FP rounding.

## Limitations & honesty

- **Reduced scope.** This is **not** a retrosynthesis planner. It does not parse
  SMILES, apply reaction templates, or search a tree. It scores a **given** batch
  of routes — the parallel step inside a planner — with a hand-written logistic
  yield model standing in for the production transformer/GNN.
- **Synthetic data.** The committed routes and the model weights are synthetic and
  illustrative (route 0 is planted as the winner). **No chemical conclusion may be
  drawn**; nothing here is validated against real yields.
- **Independence assumption.** Multiplying per-step yields assumes steps succeed
  independently, which real chemistry violates (shared intermediates, telescoped
  steps). Real planners learn route-level scores; we keep the product because it
  is transparent and teaches the GPU mapping cleanly.
- **Not bit-exact.** CPU and GPU agree to ~`1e-8`, not exactly, due to
  single-precision `expf`/FMA differences (documented, verified to `1e-6`).
- **Not for clinical or laboratory decisions.** Educational study material only.
