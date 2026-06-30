# THEORY — 2.17 Allosteric Network Analysis

> The deep dive. We go science → math → algorithm (with complexity) → GPU mapping →
> numerics → verification → where this sits in the real world. Read alongside the
> code: `src/dcc_core.h` (the shared per-pair math), `src/reference_cpu.cpp` (the
> baseline + network analysis), and `src/kernels.cu` (the GPU DCC kernel).
>
> _Educational only — not for clinical use._

---

## The science

**Allostery** is action at a distance inside a protein: a molecule binding at one
site (the *allosteric site*) changes the protein's behavior at a *different*,
often distant, site (the *active site*) — without the two ligands ever touching.
It is one of biology's master switches: hemoglobin's cooperative oxygen binding,
the on/off control of countless enzymes, and a large fraction of modern drug
targets (allosteric drugs can be more specific than active-site competitors, and
can reach "undruggable" proteins through cryptic pockets).

How does a perturbation at site A reach site B? Through **correlated motion**. A
protein is not a rigid statue; it jiggles. Groups of residues move together as
semi-rigid domains, hinges flex, and these collective motions can carry a
mechanical/entropic signal across the structure. If we watch the protein move (a
**molecular-dynamics, MD, trajectory**) we can ask, for every pair of residues:
*do they move together?* Residues that consistently move in concert form a
**communication network**, and the strongest-coupled route from the allosteric
site to the active site is the candidate **allosteric pathway**. Finding that
pathway (and the **bottleneck residue** that gates it) is the goal here.

This project implements the core, GPU-accelerated numerical engine of that
analysis — the residue–residue correlation matrix and the network shortest path —
on a small **synthetic** trajectory engineered to contain a known pathway, so the
result is interpretable and the GPU can be verified against a CPU reference.

---

## The math

We have a trajectory of `N` residues (represented by their Cα atoms) over `T`
frames. Let `r_i(t) ∈ ℝ³` be the position of residue `i` in frame `t`, and let

$$\langle r_i \rangle = \frac{1}{T}\sum_{t=0}^{T-1} r_i(t)$$

be its **mean (equilibrium) position**. The **displacement from the mean** is
`Δr_i(t) = r_i(t) − ⟨r_i⟩`.

### Dynamical Cross-Correlation (DCC)

The (normalized) **DCC matrix** entry is the time-averaged, variance-normalized dot
product of the two residues' displacement vectors:

$$
C_{ij} \;=\; \frac{\big\langle \Delta r_i \cdot \Delta r_j \big\rangle}
{\sqrt{\big\langle \Delta r_i \cdot \Delta r_i \big\rangle \;
       \big\langle \Delta r_j \cdot \Delta r_j \big\rangle}}
\;=\;
\frac{\sum_t \Delta r_i(t)\cdot \Delta r_j(t)}
     {\sqrt{\big(\sum_t |\Delta r_i(t)|^2\big)\big(\sum_t |\Delta r_j(t)|^2\big)}}.
$$

This is a **Pearson correlation coefficient** generalized to 3-D vectors, so
`C_{ij} ∈ [−1, +1]`:

- `C_{ij} ≈ +1` — residues move **together** (in phase): correlated.
- `C_{ij} ≈ −1` — residues move **oppositely** (anti-phase): anti-correlated.
- `C_{ij} ≈ 0`  — motions unrelated.

By construction `C_{ii} = 1` and `C` is symmetric. The `1/T` factors in numerator
and denominator cancel, so we never form them (see `dcc_pair()` in `dcc_core.h`).

### From correlations to a communication network

Strong correlation = strong potential to transmit a signal. Following Bio3D /
WORDOM / Sethi *et al.* (2009), convert each correlation into an edge **weight**
that behaves like a distance ("how expensive is it to send a signal across this
edge?"):

$$ w_{ij} = -\log |C_{ij}|. $$

`|C|→1` gives `w→0` (a cheap, strong link); `|C|→0` gives `w→∞` (an expensive,
weak link). We only place an edge where residues are in **spatial contact** (mean
Cα–Cα distance ≤ cutoff, or backbone neighbors), so a signal must physically hop
residue-to-residue rather than teleport.

### The allosteric pathway

The **optimal communication pathway** between the allosteric site `a` and the
active site `b` is the **shortest path** in this weighted graph:

$$ \text{path}(a,b) = \arg\min_{\text{paths } a\to b} \sum_{(i,j)\in\text{path}} w_{ij}. $$

Its total cost is `dist(a,b)`. The **bottleneck** is the edge on that path with the
largest `w` (smallest `|C|`) — the weakest link, the synthetic stand-in for a real
allosteric **hotspot** residue.

```
 allosteric site                                         active site
      (a)                                                    (b)
       o---o---o---o == o---o---o---o---o == o---o---o---o---o
       2   3   4  ... \                    /  ... 24  25  26  27
                       \  bottleneck hop  /
                        (hinge boundary 9-10, weak |C|)
   strong |C| edges (cheap)   weak |C| edge (gate)   strong |C| edges (cheap)
```

---

## The algorithm (and complexity)

| Step | What | Serial complexity |
|---|---|---|
| 1 | Per-residue means `⟨r_i⟩` | `O(N·T)` |
| 2 | **DCC matrix** `C` (every entry an `O(T)` average) | **`O(N²·T)`** ← bottleneck |
| 3 | Contact graph from the mean structure | `O(N²)` |
| 4 | All-pairs shortest paths (Floyd–Warshall) | `O(N³)` |
| 5 | Reconstruct the `a→b` pathway | `O(N)` |

For real proteins `N` is hundreds–thousands and `T` is `10⁴–10⁶` frames, so **step
2 dominates** — it is `O(N²·T)` and is exactly the step we move to the GPU. Steps
3–5 are the cheap, deterministic network payoff computed once on the CPU from the
verified matrix. (For very large `N`, step 4 can also be parallelized — a blocked
GPU Floyd–Warshall — noted as an exercise.)

---

## The algorithm → GPU mapping

The DCC matrix is **embarrassingly parallel**: every entry `C_{ij}` is an
independent `O(T)` reduction that reads the (shared, read-only) trajectory and
writes one output cell. No entry depends on another. This is PATTERNS.md's
"independent jobs" idiom (as in `1.12` Tanimoto), lifted from a 1-D array of jobs
to a 2-D matrix of jobs.

**Thread → data map.** We launch a **2-D grid of 2-D blocks** of `16×16 = 256`
threads. Thread

```
col = blockIdx.x*blockDim.x + threadIdx.x   (residue j, the matrix column)
row = blockIdx.y*blockDim.y + threadIdx.y   (residue i, the matrix row)
```

computes exactly `C[row][col] = dcc_pair(coords, mean, row, col, T, N)` and stores
one float. The grid is `(⌈N/16⌉, ⌈N/16⌉)` so it tiles the whole `N×N` matrix; the
ragged right/bottom tiles guard `row≥N || col≥N`.

**Memory hierarchy.**

- `coords` (`T·N·3` floats) and `mean` (`N·3` doubles) live in **global memory**,
  read by all threads.
- Each thread keeps its three running covariance sums in **registers** (`double`).
- The output `C` (`N·N` floats) is written once per thread — coalesced within a
  warp because adjacent `col` values map to adjacent memory.
- **No shared memory, no atomics, no synchronization** — each thread owns a
  distinct output cell, so there is nothing to coordinate. That is what makes this
  a clean, race-free teaching kernel.

**Why no atomics here (contrast with k-means/Monte-Carlo).** Those patterns have
*many* threads writing the *same* accumulator, which needs atomics (and integer
fixed-point for determinism). Here the writes are disjoint, so the result is
deterministic for free.

**An optimization we deliberately skip (mentioned for the curious).** Each residue's
`T`-long track is re-read by every thread in its row and column. A shared-memory
tiling — cooperatively staging strips of the trajectory into `__shared__` memory and
reusing them across a `16×16` tile — would cut global-memory traffic markedly,
exactly like the matrix-multiply tiling classic. We keep the naive version because
it teaches the thread→entry mapping with zero distraction; the tiled version is an
exercise.

**Occupancy / bandwidth.** With `O(T)` flops per output and a single float store,
the kernel is compute-light and bandwidth-friendly; `256` threads/block gives the
scheduler 8 warps to hide the global-load latency of the inner sum.

---

## Numerical considerations

- **Accumulate in `double`.** The trajectory is `float`, but the covariance sums
  over `T` frames are formed in `double` (in `dcc_pair`). Summing `T` products in
  `float` would lose precision; `double` accumulation is the cheap, standard guard.
  The final `C_{ij}` is cast to `float` only for storage/comparison.
- **Determinism.** Each thread sums its `T` terms in the **same order** the CPU
  loop does, with the same `double` arithmetic. There is no cross-thread reduction,
  so there is no floating-point reordering. The GPU and CPU therefore produce
  **bit-identical** matrices — see verification below.
- **Divide-by-zero guard.** A residue that never moves has zero variance; `dcc_pair`
  returns `1.0` on the diagonal and `0.0` off-diagonal instead of `0/0`.
- **`-log|C|` guard.** `comm_weight` clamps `|C|` to `(10⁻⁶, 1]` so the logarithm is
  finite and the edge weight is always `≥ 0` (never negative — Floyd–Warshall
  assumes non-negative edges).
- **Floyd–Warshall sentinel.** "Unreachable" is `1e30`, a large *finite* value, so
  `INF + INF` cannot overflow to a NaN that would corrupt a relaxation test.

---

## How we verify correctness

Two layers:

1. **GPU vs CPU, exact.** `main.cu` computes the DCC matrix on both paths and
   asserts the worst entry-wise difference is **exactly `0.0`**. This is the
   strongest, most honest tolerance and it is justified: both paths call the *same*
   `dcc_pair()` from `dcc_core.h`, in `double`, in the same order (PATTERNS.md §4:
   "exact when the same exact operations run on both sides"). If the kernel had a
   bug — a bad index, a missing guard, a precision slip — this would catch it.
2. **Recovering the planted science.** The synthetic data embeds a known pathway:
   two functional sites at opposite ends, three breathing domains coupled by one
   collective mode, and a partly-decoupled hinge. The demo must recover (a) a
   communication path spanning the chain end-to-end, and (b) a **bottleneck hop at
   the hinge boundary** (`9-10` in the committed sample). This validates the
   *analysis*, not just CPU==GPU agreement.

Edge cases covered: ragged grid tiles (`N` not a multiple of 16), the diagonal
(`C_{ii}=1`), zero-variance residues, and unreachable site pairs (reported as
"NONE").

---

## Where this sits in the real world

Production allosteric-network tools do considerably more than this teaching engine:

- **ProDy / Bio3D** compute DCC (and the related **mutual-information** and **linear
  mutual information** measures, which capture *nonlinear* coupling that Pearson DCC
  misses), build the residue network, and run shortest-path / **betweenness** and
  **community detection** (Girvan–Newman, Louvain) to find functional modules. Our
  catalog entry lists exactly these (DCC, MI, LRT perturbation scanning, Girvan–
  Newman/Louvain, WORDOM shortest-path communication).
- **Perturbation Response Scanning (PRS)** and **Linear Response Theory** push a
  force at one residue and measure the displacement everywhere — a complementary,
  *causal* view of allostery this project does not implement.
- **The trajectory matters.** Real analyses need **long, well-equilibrated MD**
  (often µs-scale, GPU-accelerated in OpenMM/GROMACS/AMBER) and careful **alignment**
  (removing global translation/rotation, e.g. by RMSD superposition) before
  computing fluctuations — we skip alignment because the synthetic data has no global
  drift by construction.
- **Scaling.** At production `N` and `T`, the `O(N²T)` DCC is genuinely the
  bottleneck and is computed with GPU GEMM-style kernels or **cuBLAS** outer
  products on the centered displacement matrix; **RAPIDS cuGraph** provides GPU
  Louvain/betweenness for the network step (our catalog's "CUDA pattern"). This
  project's custom kernel teaches the *mechanics* of that GEMM; the libraries are the
  industrial version.

This is a **reduced-scope teaching version**: correct DCC + a clean shortest-path
network on synthetic data, with the fuller method (MI, PRS, community detection,
real trajectories) described here and left as exercises.
