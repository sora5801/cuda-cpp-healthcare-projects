# 1.17 — Markov State Models from MD

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.17`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A **Markov State Model (MSM)** turns a long molecular-dynamics (MD) trajectory —
a time series of molecular conformations — into a small, interpretable model of
the molecule's *kinetics*. This project builds a reduced-scope MSM end to end on
the GPU: it **clusters** MD frames into a handful of "microstates" with k-means,
**counts** the transitions between microstates at a chosen lag time, **estimates**
the transition probability matrix, and **extracts** the equilibrium populations
and the slowest implied timescale (the molecule's slowest conformational
process). It is a clean, self-checking example of the most important MSM steps,
built on the GPU pattern *parallel-assign + atomic-integer-reduce*.

## What this computes & why the GPU helps

Markov State Models discretize MD conformational space into metastable states and
estimate transition probabilities from long or many-short trajectories. Building
an MSM requires: (1) featurization of (often millions of) MD frames,
(2) dimensionality reduction (tICA/PCA), (3) clustering into microstates
(k-means), and (4) transition-matrix estimation. Steps 1–3 are
GPU-acceleratable. The payoff is extraction of thermodynamics and kinetics
(populations, slow timescales, binding pathways) from aggregated µs–ms of GPU MD.

**The parallel bottleneck:** with millions of frames, the two dominant steps are
both embarrassingly parallel and are exactly what this project runs on the GPU:

- **k-means assignment** — one thread per frame finds its nearest microstate
  centroid (Euclidean). Independent per frame → a 1-D grid over frames.
- **transition counting** — one thread per time index `t` scatters the pair
  `(state(t) → state(t+τ))` into a K×K count matrix via an **integer** atomicAdd.

Both reductions use **integer / fixed-point** accumulation so they are
*deterministic* and match the CPU reference bit-for-bit (see THEORY §Numerics).

## The algorithm in brief

- **Featurize** (here: provided as the input feature matrix; in production: tICA /
  PCA on internal coordinates).
- **Cluster** frames into K microstates with Lloyd's **k-means** (farthest-first
  seeding for determinism).
- **Count transitions** `C[i][j]` = number of times the trajectory is in
  microstate `i` at time `t` and microstate `j` at time `t+τ` (lag time τ).
- **Estimate** the transition matrix `T` by row-normalizing `C` (maximum
  likelihood): `T[i][j] = C[i][j] / Σ_j C[i][j]`.
- **Analyze**: the stationary distribution `π` (`πT = π`) gives equilibrium
  populations; the second eigenvalue `λ₂` gives the slowest implied timescale
  `t₂ = −τ / ln(λ₂)`.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation, including PCCA+, the Chapman-Kolmogorov test, and VAMP (the
production-grade extensions this teaching version omits).

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/markov-state-models-from-md.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/markov-state-models-from-md.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\markov-state-models-from-md.sln /p:Configuration=Release /p:Platform=x64
```

This project links only the CUDA runtime (`cudart`); it uses **no** extra CUDA
library, so the kernels (k-means assign, atomic accumulate, transition count) are
written and explained by hand — nothing is a black box (CLAUDE.md §6.1.6).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/trajectory_sample.txt`, prints
the MSM (populations, transition matrix, slowest timescale), shows the
GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/trajectory_sample.txt` — a tiny synthetic
  featurized trajectory (12,000 frames × 3 features) so the demo runs offline.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print pointers to real MD
  sets; `scripts/make_synthetic.py` regenerates / enlarges the synthetic one.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: mdCATH — 5 µs MD trajectories for 272 proteins
(<https://huggingface.co/datasets/compsciencelab/mdcath>); Fast-folder benchmark
trajectories (chignolin, Trp-cage, Villin — Piana/Shaw); GPCRmd
(<https://gpcrmd.org>); D. E. Shaw millisecond trajectories (via RCSB deposition).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program builds the MSM on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and asserts they agree: the microstate
labels and the integer transition-count matrix match **exactly**, and the
centroids / transition matrix match to machine precision. The recovered
transition matrix closely matches the known ground-truth matrix that generated
the synthetic trajectory (see `data/README.md`) — that is the science check, not
just a CPU==GPU check.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the trajectory, runs CPU + GPU, verifies, reports.
2. [`src/msm.h`](src/msm.h) — the shared `__host__ __device__` core: distance,
   nearest-centroid, and fixed-point quantization (one source of truth for CPU+GPU).
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the assign / accumulate / count kernels + the driver.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial pipeline and the
   shared host helpers (centroid update, transition matrix, π, timescale).
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, I/O helpers.

## Prior art & further reading

- **PyEMMA** (<https://github.com/markovmodel/PyEMMA>) — the classic MSM toolkit:
  featurization, tICA, clustering, MSM estimation, Chapman-Kolmogorov tests.
- **MSMBuilder** (<https://github.com/msmbuilder/msmbuilder>) — statistical models
  for biomolecular dynamics; study its transition-matrix estimators.
- **deeptime** (<https://github.com/deeptime-ml/deeptime>) — modern MSM/VAMP and
  VAMPnets; the variational view that supersedes hand-picked features.
- **cuML / RAPIDS** (<https://github.com/rapidsai/cuml>) — GPU k-means and PCA; the
  production way to do the clustering step at scale.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Parallel assign + atomic-integer reduction** (see `docs/PATTERNS.md`, the
"clustering / centroid accumulation" row, exemplified by flagship 11.09): one
thread per frame for k-means assignment; integer/fixed-point atomicAdd for the
centroid sums and the K×K transition counts, which makes the reductions
deterministic and exactly CPU-matching. The spectral analysis (stationary
distribution, slowest timescale) is a tiny K×K host step shared by both paths.
Catalog note (for scale-up): cuML k-means and cuBLAS-based tICA covariance are the
library route for production featurization/clustering.

## Exercises

1. **Lag-time scan (implied-timescale plot).** Re-run with several lag times
   (`--lag` in `make_synthetic.py`, or load with different τ) and plot `t₂` vs τ.
   A good MSM shows `t₂` *plateauing* once τ exceeds the fast relaxation — the
   standard way to choose a lag.
2. **Chapman-Kolmogorov test.** Check that `T(2τ) ≈ T(τ)²` (Markovianity). Add a
   kernel that counts transitions at `2τ` and compare to the squared matrix.
3. **More microstates → PCCA+.** Increase `K` (over-discretize), then coarse-grain
   back to 3 macrostates by lumping microstates with similar kinetics (PCCA+).
4. **FP64 vs FP32 features.** Switch the feature storage to `double` and observe
   that labels/counts are unchanged (the clustering is robust) — a lesson in when
   precision matters.
5. **Bigger trajectory.** `python scripts/make_synthetic.py --frames 200000` and
   watch the GPU's relative edge grow as the launch overhead is amortized.

## Limitations & honesty

- **Reduced-scope teaching version.** Real MSM construction includes featurization
  and **tICA** (this project takes the features as given), **mini-batch** k-means,
  **Bayesian / reversible** transition-matrix estimators, **PCCA+** coarse-graining,
  **Chapman-Kolmogorov** validation, and **VAMP** model selection. THEORY.md
  describes these; the code implements the core estimator only.
- **Eigen-analysis is intentionally simple.** π and `λ₂` use power iteration with
  deflation (exact enough for the small, well-conditioned `T` here); production
  code uses a general (and reversibility-aware) eigensolver.
- **The data is synthetic.** The committed trajectory is a hidden-Markov toy with
  a *known* transition matrix, chosen so the result is interpretable and
  verifiable. It is not real MD and carries no physical or clinical meaning.
- **Timing is a teaching artifact, not a benchmark.** On this tiny set the GPU is
  launch-bound; its advantage appears only with millions of frames.
