# THEORY — 1.17 Markov State Models from MD

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

Proteins and other biomolecules are not static crystal structures; they *move*.
A molecular-dynamics (MD) simulation integrates Newton's equations for every atom
in tiny femtosecond steps, producing a **trajectory**: a movie of the molecule
wandering across its energy landscape. The biologically interesting events —
folding, a domain hinge motion, a ligand binding or unbinding — are **rare**:
the molecule spends most of its time jiggling inside a deep energy basin (a
*metastable state*) and only occasionally hops over a barrier to another basin.

Two problems make raw trajectories hard to use:

1. **They are enormous.** Capturing even one slow event needs micro- to
   milliseconds of simulated time, i.e. millions to billions of frames.
2. **The signal is buried.** The slow, important motion is hidden under fast,
   uninteresting thermal vibrations.

A **Markov State Model (MSM)** is the standard cure. The idea is to *coarse-grain*
the continuous, high-dimensional trajectory into a small **discrete-state Markov
chain**: a handful of metastable "microstates" plus the probabilities of hopping
between them in a fixed time interval. From that tiny model you read off the
quantities chemists actually want:

- **Thermodynamics** — the equilibrium population of each state (which
  conformations are favored, and by how much).
- **Kinetics** — the rates and **timescales** of the slow processes (how long
  folding takes; on/off rates `k_on`, `k_off` for binding).
- **Mechanism** — the pathways and intermediate states between, say, unfolded and
  folded.

This project builds the *core* of an MSM — discretize, count, estimate, analyze —
and runs the two heavy steps on the GPU.

## 2. The math

**Inputs.** A featurized trajectory: `N` time-ordered frames, each a feature
vector `x_t ∈ ℝ^D` (here `D = 3`, values in `[0,1]`). A number of microstates `K`
and a lag time `τ` (in frames).

**Step 1 — Discretize (k-means).** Partition the `N` points into `K` clusters by
minimizing the within-cluster sum of squares (inertia)

  J = Σ_t ‖ x_t − c_{s(t)} ‖²,  where  s(t) = argmin_k ‖ x_t − c_k ‖²

and `c_k` is the centroid of cluster `k`. Lloyd's algorithm alternates **assign**
(`s(t)` for all `t`) and **update** (`c_k ← mean of its assigned points`) until
convergence. The discrete label sequence `s(0), s(1), …, s(N−1)` is the
trajectory's path through microstate space.

**Step 2 — Count transitions.** At lag `τ`, form the `K×K` **count matrix**

  C[i][j] = #{ t : s(t) = i and s(t+τ) = j },  0 ≤ t < N−τ.

`C` is the *sufficient statistic* of a Markov chain: it records how often each
hop `i → j` occurs over the interval `τ`.

**Step 3 — Estimate the transition matrix.** The maximum-likelihood estimate of
the row-stochastic transition probability matrix `T` is the row-normalized count
matrix

  T[i][j] = C[i][j] / Σ_j C[i][j] ,  so  Σ_j T[i][j] = 1.

`T[i][j]` is the probability that, given the system is in microstate `i` now, it
is in microstate `j` after time `τ`. (A row with no observed outgoing transitions
is set to a self-loop, keeping `T` stochastic.)

**Step 4 — Analyze the spectrum.** A row-stochastic `T` always has top eigenvalue
`λ₁ = 1`. Its eigenvalues are real (for a reversible chain) and ordered
`1 = λ₁ > λ₂ ≥ λ₃ ≥ …`.

- The **stationary distribution** `π` is the left eigenvector for `λ₁ = 1`
  (`π T = π`, `Σ π_i = 1`): the equilibrium population of each microstate.
- The **slowest implied timescale** comes from the *second* eigenvalue:

    t₂ = −τ / ln(λ₂).

  `λ₂` near 1 means a very slow relaxation (a high barrier); `t₂` is that slow
  process's characteristic time. This single number — the slowest motion of the
  molecule — is the headline result an MSM exists to produce.

## 3. The algorithm

```
load trajectory (N frames, D features, K, lag)
init_centroids        (farthest-first / k-means++ seed)
repeat ITERS times:                       # Lloyd's k-means
    assign:    s(t) = nearest centroid for every frame t        # O(N·K·D)
    update:    c_k  = mean of frames with s(t)=k                # O(N·D)
count_transitions:   C[s(t)][s(t+lag)] += 1   for all t          # O(N)
build T:             row-normalize C                            # O(K²)
stationary_distribution(T)  by power iteration                  # O(K²·iters)
slowest_timescale(T)        by deflated power iteration         # O(K²·iters)
```

**Complexity.** The dominant cost is the k-means **assign** step,
`O(ITERS · N · K · D)`. For real MSMs `N` is in the millions and this is the
runtime, which is why it is the step we parallelize. **update** and
**count** are `O(N·D)` and `O(N)` — also linear in `N`, also parallel. Everything
after the counting (`T`, `π`, `t₂`) is `O(K²)` with tiny `K`, so it is negligible
and stays on the host.

**Data-access pattern.** *assign* reads each frame's `D` features once and the
`K·D` centroids many times (centroids are small and cache-resident). *count*
reads the label sequence twice (`s(t)` and `s(t+τ)`) and scatters into the small
`K×K` matrix — a reduction with heavy collisions, hence atomics.

## 4. The GPU mapping

Two kernels do the parallel work; both are **one thread per data item over a 1-D
grid** with `blockDim.x = 256` and `gridDim.x = ⌈N/256⌉`.

**(A) `assign_kernel` — one thread per frame.**
Thread `i = blockIdx.x*blockDim.x + threadIdx.x` owns frame `i`. It reads
`x[i*D .. i*D+D)` from global memory, loops over the `K` centroids calling the
shared `km_nearest()` (the *same* routine the CPU uses), and writes `labels[i]`.
No atomics, no shared memory — assignments are independent. The centroids array
(`K·D` floats) is reused by every thread and lives hot in L2.

**(B) `accumulate_kernel` — one thread per frame, atomic reduce.**
Each frame adds its `D` **fixed-point** coordinates to its microstate's running
sum and bumps that state's count:

```
for d in 0..D:  atomicAdd(&sum[k*D + d], km_to_fixed(x[i*D+d]))
atomicAdd(&count[k], 1)
```

Many frames share a microstate → the adds collide → `atomicAdd`. Accumulating
**integers** (fixed-point) instead of floats makes the adds commute, so the result
is order-independent (deterministic) and equals the CPU's plain `+=` exactly. The
host then divides `sum/count` to get the new centroids (the same `update_centroids`
both paths call).

**(C) `count_transitions_kernel` — one thread per time index `t`.**
Thread `t` (with `t+τ < N`) reads `labels[t]` and `labels[t+τ]` and does
`atomicAdd(&C[from*K + to], 1)`. Integer atomics again → the GPU `C` is identical
to the CPU `C`, frame for frame.

```
   frames x[0..N)                centroids c[0..K)            counts C[K x K]
   ┌─┬─┬─┬─ ... ─┬─┐             ┌──┬──┬──┐                  ┌──┬──┬──┐
   │ │ │ │       │ │             │c0│c1│c2│                  │  │  │  │   row=from
   └┬┴┬┴┬┴─ ... ─┴┬┘             └──┴──┴──┘                  ├──┼──┼──┤   col=to
    │ │ │         │   assign        ▲ reused                 │  │  │  │
    ▼ ▼ ▼         ▼   ───────►  km_nearest()  ──► labels ──► atomicAdd(C[from][to])
  thread per frame (grid of 256-thread blocks)              (thread per time t)
```

**No CUDA library is used.** The catalog's scale-up route is **cuML k-means**
(GPU clustering) and a **cuBLAS** covariance for tICA. We hand-write the kernels
here so the assignment and the atomic reduction are fully visible
(CLAUDE.md §6.1.6). What cuML adds is mini-batching, k-means++ on the GPU, and
multi-GPU scaling — orchestration around the same two primitives shown above.

## 5. Numerical considerations

- **Precision.** Features are FP32 (plenty for clustering); distances accumulate
  in **double** inside `km_sqdist` so host and device round identically. Centroid
  sums use **fixed-point `unsigned long long`** (scale `2²⁰`): a sum over millions
  of `[0,1]` coordinates stays far below `2⁶⁴`, so no overflow.
- **Determinism (the core lesson).** A float `atomicAdd` is *not associative* — the
  thread-scheduling order changes the rounding, so the sum is irreproducible
  (PATTERNS.md §3). We sidestep this everywhere on the hot path: the centroid sums
  are fixed-point integers and the transition counts are plain integers. **Integer
  addition commutes**, so both reductions are bit-for-bit reproducible across runs
  *and* identical to the serial CPU result. Running the demo twice yields
  byte-identical stdout.
- **Ties.** `km_nearest` replaces the best centroid only on a *strict* `<`, so ties
  always go to the lowest index — the same rule on CPU and GPU, so no frame can
  land in different microstates on the two paths.
- **Eigen-analysis.** `π` and `λ₂` use power iteration (with deflation against the
  dominant constant eigenvector for `λ₂`). This is exact enough for the small,
  well-conditioned `T` we build; it is the one floating-point step that *could*
  differ between paths, but both run it on an *identical* integer-derived `T`, so
  they agree to ~1e-12.

## 6. How we verify correctness

`src/reference_cpu.cpp` is an independent, plainly serial implementation of the
whole pipeline. `main.cu` runs it and the GPU version on the same input and checks
**four** things:

1. **Labels** — every frame's microstate must match exactly (integer compare).
2. **Transition counts** — the entire `K×K` integer matrix must match exactly.
3. **Centroids** — `max|Δ|` ≤ `1e-4` (they are derived from identical fixed-point
   sums, so this slack only covers the float cast of the mean).
4. **Transition matrix `T`** — `max|Δ|` ≤ `1e-12` (derived from identical counts).

Why two *exact* checks? Because the parallel steps use only integer/fixed-point
atomics, exact agreement is achievable and is the strongest possible evidence: a
race or an indexing bug would change at least one count. We also print a second,
*scientific* check: the recovered `T` is compared (by eye, in `demo/README.md`)
against the **known** transition matrix that generated the synthetic trajectory —
they match to ~0.01 (sampling noise), validating the *science*, not just CPU==GPU.

Edge cases handled: ragged last block (guarded with `if (i >= N) return`), empty
microstates (keep their old centroid), and a never-left state (self-loop row in
`T`).

## 7. Where this sits in the real world

Production MSM construction (PyEMMA, deeptime, MSMBuilder) adds, in order:

- **Featurization + tICA.** Real frames are thousands of atomic coordinates;
  tICA (time-lagged ICA) finds the *slowest* linear collective variables to
  cluster in — far better than raw coordinates. We take the features as given.
- **Better clustering.** Mini-batch k-means / regular-space clustering over
  millions of frames (cuML on the GPU). We do plain Lloyd's k-means.
- **Reversible / Bayesian estimators.** Real `T` is estimated under detailed
  balance (reversibility) and with error bars (Bayesian sampling), not just the
  raw row-normalized MLE. We do the MLE.
- **Validation.** The **Chapman-Kolmogorov test** (`T(nτ) ≈ T(τ)ⁿ`) and the
  **implied-timescale plot** (`t_i` vs `τ` plateau) decide whether the model is
  truly Markovian and how to pick `τ`. We compute one `t₂` at one `τ`
  (Exercise 1 builds the plot).
- **Coarse-graining + VAMP.** **PCCA+** lumps many microstates into a few
  interpretable macrostates; the **variational approach (VAMP/VAMPnets)** replaces
  hand-built features with learned ones and scores models objectively.

These are the chapters this reduced-scope version deliberately omits; the GPU
primitives it *does* show — parallel assignment and deterministic atomic
reductions — are exactly the ones those production tools accelerate.

---

## References

- **PyEMMA** — Scherer et al., *J. Chem. Theory Comput.* 2015. The reference MSM
  workflow; read its `coordinates` (tICA) and `msm` modules.
  <https://github.com/markovmodel/PyEMMA>
- **deeptime** — Hoffmann et al., *Mach. Learn.: Sci. Technol.* 2022. Modern MSM +
  VAMPnets; the variational/learned-feature view.
  <https://github.com/deeptime-ml/deeptime>
- **MSMBuilder** — Harrigan et al., *Biophys. J.* 2017. Estimators and
  coarse-graining for biomolecular MSMs. <https://github.com/msmbuilder/msmbuilder>
- **MSM theory** — Prinz et al., *J. Chem. Phys.* 134:174105 (2011), "Markov models
  of molecular kinetics: Generation and validation" — the canonical derivation of
  the count→matrix→spectrum pipeline and the Chapman-Kolmogorov test.
- **PCCA+** — Röblitz & Weber, *Adv. Data Anal. Classif.* 2013 — spectral
  coarse-graining of microstates into macrostates.
- **cuML** — RAPIDS GPU k-means/PCA used to scale the clustering step.
  <https://github.com/rapidsai/cuml>
