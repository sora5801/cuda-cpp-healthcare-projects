# 11.09 — Flow Cytometry & High-Content Screening Analysis

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biotechnology%2C%20Bioprocess%20%26%20Synthetic%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 11: Biotechnology, Bioprocess & Synthetic Biology · Catalog ID `11.09`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Cluster flow-cytometry **events** (cells, each measured on several markers) into
populations with GPU **k-means**. k-means alternates two GPU steps: **assign**
every event to its nearest centroid (one thread per event), then **update** each
centroid as the mean of its members (a scatter-**reduction** via `atomicAdd`).
Tenth distinct GPU pattern in the flagships: **parallel assignment + atomic
reduction** — made deterministic with fixed-point integer accumulation.

## What this computes & why the GPU helps

Modern sorters emit ~10⁵ cells/second at 20–50 parameters; clustering millions of
events (immunophenotyping, rare-population detection) is the bottleneck. Both
k-means steps are data-parallel: assignment is independent per event, and the
centroid update is a reduction where many events accumulate into the same K
centroids. RAPIDS cuML turns 30-minute analyses into seconds this way.

**The parallelized work** is per-event assignment + the atomic centroid
accumulation, iterated; the tiny centroid divide is a host step.

## The algorithm in brief

- **Init:** deterministic **farthest-first** seeding (the greedy core of k-means++).
- **Assign:** `label[i] = argmin_k ‖x_i − c_k‖²` (one thread per event).
- **Accumulate:** `atomicAdd` each event's fixed-point coordinates into its
  cluster's sum + count.
- **Update:** `c_k = sum_k / count_k`. Repeat.

See [THEORY.md](THEORY.md) for Lloyd's algorithm, the atomic-reduction determinism trick, and init.

## Build

Requires **Visual Studio 2026** (v145) + **CUDA 13.3** ([docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/flow-cytometry-high-content-screening-analysis.sln`.
2. **`Release|x64`** → **Build** → `build/x64/Release/flow-cytometry-high-content-screening-analysis.exe`.

CLI: `msbuild build\flow-cytometry-high-content-screening-analysis.sln /p:Configuration=Release /p:Platform=x64`

## Run the demo

```powershell
./demo/run_demo.ps1
```

Clusters the committed events on CPU + GPU and verifies labels + centroids match.

## Data

- **Sample (committed):** `data/sample/cytometry_sample.txt` — 20k events, 5 markers, 5 populations.
- **Real data:** FCS files via FlowKit / FlowRepository / RAPIDS — see
  `scripts/download_data.ps1` and [data/README.md](data/README.md).
- Bigger synthetic set: `python scripts/make_synthetic.py --scale 50`.

## Expected output

`demo/expected_output.txt` holds the deterministic clusters (size + centroid each)
and inertia. The GPU (`src/kernels.cu`) and CPU (`src/reference_cpu.cpp`) share the
distance + **fixed-point** accumulation (`src/kmeans.h`) and the same host update,
so labels and centroids are **identical** (`0 mismatches`, `centroid diff 0`). The
demo recovers all 5 synthetic populations with the correct sizes.

## Code tour

1. [`src/main.cu`](src/main.cu) — load, CPU + GPU k-means, verify labels+centroids, print.
2. [`src/kmeans.h`](src/kmeans.h) — **distance, nearest-centroid, and the fixed-point accumulation** (host + device).
3. [`src/kernels.cuh`](src/kernels.cuh) — the assign + accumulate kernel interface.
4. [`src/kernels.cu`](src/kernels.cu) — assign + **atomic** accumulate + the Lloyd loop.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — farthest-first init + the serial reference.

## Prior art & further reading

- **RAPIDS cuML** (<https://github.com/rapidsai/cuml>) — GPU k-means / HDBSCAN / UMAP for cytometry.
- **FlowKit** (<https://github.com/whitews/FlowKit>) — FCS file processing (upstream of clustering).
- **CellProfiler** (<https://github.com/CellProfiler/CellProfiler>) / **CellPose** — HCS imaging + segmentation.
- PhenoGraph, FlowSOM — cytometry-specific clustering methods.

Study these for production analysis; reimplement the pattern didactically (CLAUDE.md §2).

## CUDA pattern used here

**Parallel assignment** (one thread per event, argmin) + **atomic scatter-reduction**
for centroid sums · **fixed-point integers** so the atomics commute → deterministic
and CPU-matching · deterministic farthest-first init.

## Exercises

1. **Naive init pitfall.** Switch init to evenly-spaced indices and watch k-means
   fall into a local minimum (it splits one population and merges two — higher
   inertia). This is *why* k-means++ exists.
2. **Convergence test.** Stop when labels stop changing (or inertia plateaus)
   instead of a fixed iteration count.
3. **Shared-memory centroids.** Cache the K·D centroids in shared memory in the
   assign kernel to cut global reads.
4. **Float vs fixed-point.** Replace the fixed-point accumulation with float
   `atomicAdd` and observe that the centroids vary run-to-run (lost determinism).
5. **GPU-HDBSCAN / FlowSOM.** Swap k-means for a density-based method better suited
   to non-spherical cytometry clusters.

## Limitations & honesty

- **Spherical k-means** with fixed iterations; real cytometry uses
  density/graph methods (HDBSCAN, PhenoGraph, FlowSOM) for non-convex populations.
- The centroid divide runs on the host (the parallelized work — assign + accumulate
  — is on the GPU); fixed-point quantizes coordinates to ~6 digits.
- Data is synthetic, well-separated, and grouped so the demo recovers the truth.
