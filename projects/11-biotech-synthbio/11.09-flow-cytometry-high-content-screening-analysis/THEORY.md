# THEORY — 11.09 Flow Cytometry & High-Content Screening Analysis

> For a reader who knows C++ but is new to CUDA and to clustering. See
> [README.md](README.md) for the tour and build. _Educational only._

## 1. The science

A flow cytometer streams cells past lasers and records, per cell, a vector of
**marker** intensities (scatter + fluorescent antibodies like CD3/CD4/CD8). The
analysis task is **gating**: grouping the millions of recorded "events" into cell
**populations** (T-helper, cytotoxic, B cells, ...). Automated clustering replaces
manual gating; the simplest representative is **k-means**, which partitions events
into `K` clusters by minimizing within-cluster spread.

## 2. The math

Given events `x_1..x_N ∈ ℝ^D` and `K`, k-means minimizes the **inertia**

```
J = Σ_i ‖ x_i − c_{a(i)} ‖²
```

where `a(i)` is event `i`'s cluster and `c_k` is the centroid of cluster `k`.
**Lloyd's algorithm** alternates the two steps that each minimize `J` holding the
other fixed:

```
assign:  a(i) = argmin_k ‖ x_i − c_k ‖²
update:  c_k  = mean of { x_i : a(i) = k }
```

It converges to a **local** minimum (sensitive to the initial centroids — §5).

## 3. The algorithm

```
init centroids (farthest-first)
repeat:
    for each event i:   a(i) = nearest centroid          # ASSIGN  (parallel)
    for each cluster k: c_k = mean of its events         # UPDATE  (reduction)
```

**Complexity.** Each iteration is `O(N·K·D)` for assignment and `O(N·D)` for the
reduction — dominated by assignment for large `K`. Both are parallel over the `N`
events.

## 4. The GPU mapping

**Assign** maps cleanly: one thread per event computes `K` distances and takes the
argmin (`assign_kernel`). **Update** is the interesting part — a **scatter
reduction**: every event must add its coordinates to *its* cluster's running sum,
and many events share a cluster, so the writes collide. We use `atomicAdd`
(`accumulate_kernel`); the tiny divide `sum/count` is a host step shared with the
CPU reference.

**Determinism via fixed-point (the key trick).** Floating-point `atomicAdd` is
**not associative**, so the order in which thousands of threads add — which is
nondeterministic on a GPU — would change the centroid in the last bits, making the
result irreproducible *and* different from the CPU. We instead normalize
coordinates to `[0,1]` and accumulate them as **fixed-point integers**
(`atomicAdd` on `unsigned long long`). Integer addition **commutes**, so the sum is
identical regardless of thread order and matches the CPU exactly. (This is the same
lesson as project 5.01's integer dose tally.) Exercise 4 swaps in float atomics to
*see* the determinism break.

**Why labels still match across CPU/GPU.** The argmin uses `km_sqdist`, which has
the usual FMA caveat — but the populations are well separated, so the nearest
centroid wins by a wide margin and no assignment flips. With identical labels and
identical (integer) sums, the centroids are bit-identical, so the next iteration's
assignment is identical too — the whole run stays in lock-step.

## 5. Numerical considerations & initialization

- **Determinism:** guaranteed by fixed-point atomics + a deterministic init.
- **Initialization matters.** k-means only finds a *local* optimum, and a bad seed
  gives a bad clustering. With **naive evenly-spaced** init on this grouped data,
  two seeds land in one population and none in another, so k-means splits one
  cluster and merges two (inertia ≈ 637). We instead use **farthest-first** seeding
  (the greedy heart of **k-means++**): start anywhere, then repeatedly pick the
  event farthest from all chosen centers. That seeds one centroid per well-separated
  population, and the demo recovers all five cleanly (inertia ≈ 160). Exercise 1
  reproduces the bad case on purpose — a vivid lesson in why initialization is not
  a detail.

## 6. How we verify correctness

`main.cu` runs k-means on CPU and GPU and checks that **every label** and **every
centroid** match (`0 mismatches`, `centroid diff = 0`) — exact, thanks to the
fixed-point reduction. Beyond CPU/GPU parity, the result is biologically sensible:
the five synthetic populations are recovered with the right sizes and centroids at
their true marker patterns, and the inertia is the low-objective solution.

## 7. Where this sits in the real world

Production cytometry pipelines (RAPIDS cuML, FlowSOM, PhenoGraph, GPU-HDBSCAN) use
density- and graph-based clustering that handles **non-spherical** populations,
plus GPU **UMAP/t-SNE** for visualization and deep classifiers for rare phenotypes.
k-means is the didactic entry point; the **parallel-assign + atomic-reduce** GPU
structure you learn here recurs in all of them (and in GPU EM, mean-shift, and
mini-batch variants).

## References

- Lloyd (1982), *Least Squares Quantization in PCM* — the algorithm.
- Arthur & Vassilvitskii (2007), *k-means++: The Advantages of Careful Seeding*.
- Van Gassen et al. (2015), *FlowSOM* — cytometry clustering.
- NVIDIA CUDA C++ Programming Guide — atomics; RAPIDS cuML docs — GPU k-means.
