# THEORY — 2.35 Electron Paramagnetic Resonance (EPR/DEER) Constrained Modeling

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See
> [README.md](README.md) for the quick tour and build steps.
>
> _Educational only — not for clinical use. The data is synthetic._

---

## 1. The science

Many of the most interesting proteins — membrane transporters (ABC transporters,
LeuT-fold symporters), GPCRs, intrinsically disordered regions — are **flexible**.
They do not sit in one rigid shape; they sample an **ensemble** of conformations,
and the populations of those conformations encode the mechanism (an importer
gating between inward- and outward-open states, for example). X-ray
crystallography gives you one (or a few) static snapshots, often of a protein
yanked out of its membrane and locked by a crystal lattice. That is exactly the
wrong tool for a floppy machine.

**EPR** (Electron Paramagnetic Resonance) spectroscopy offers a complementary
view. You engineer two cysteine residues into the protein and attach a small
**spin label** to each — most commonly **MTSSL**, a nitroxide whose unpaired
electron is the thing EPR detects. A pulsed-EPR experiment called **DEER** (Double
Electron–Electron Resonance, a.k.a. **PELDOR**) measures the **dipolar coupling**
between the two electron spins. Because that coupling scales as 1/r³ with the
spin–spin separation r, inverting the DEER time trace yields a **distance
distribution** `P(r)` — a probability density over the distance between the two
labels, typically in the **1.5–8 nm** range. Crucially, DEER works on the protein
**in a near-native, often membrane, environment**, and the *width* of `P(r)`
reports the conformational heterogeneity directly.

So now you have an experimental `P(r)` and a computational **ensemble** of model
structures (e.g. from molecular dynamics). The scientific question of this
project: **which conformations, and in what populations, are consistent with the
DEER data?** Answering it lets you refine — or outright build — models of proteins
that crystallography cannot reach.

Two wrinkles make this non-trivial and are the heart of the computation:

1. **The label is not a point.** MTSSL is a ~6-atom flexible tether; its nitroxide
   samples a *cloud* of positions (a **rotamer** distribution) relative to the
   backbone. So even a single rigid protein frame produces a *spread* of spin–spin
   distances. You must **convolve** the two rotamer clouds to back-calculate that
   frame's `P_m(r)`.
2. **The ensemble may be wrong.** An MD force field rarely reproduces experimental
   populations out of the box. Rather than trust it blindly, you **reweight** the
   ensemble: adjust the population of each frame to match the data — but only as
   much as the data demands, using a **maximum-entropy** prior so you do not
   overfit (the BioEn / EROS philosophy).

This project implements the two GPU-relevant, well-defined stages — **rotamer-
convolution back-calculation** and **maximum-entropy reweighting** — on a small
synthetic ensemble with a known answer. (Running the upstream MD that *generates*
the ensemble is a separate, much larger problem; see §7.)

## 2. The math

**Symbols.**

| symbol | meaning | units / range |
|---|---|---|
| `M` | number of ensemble members (MD frames) | integer |
| `R` | spin-label rotamers modelled per site per frame (`ROTAMERS_PER_SITE`) | integer (24 here) |
| `A_{m,i}, B_{m,j}` | 3-D positions of rotamer `i`/`j` of label A/B in frame `m` | nm |
| `r` | a spin–spin distance | nm |
| `NBINS, r_b` | number of `P(r)` histogram bins; centre of bin `b` | 50; nm |
| `P_m(r)` | frame `m`'s back-calculated distance distribution | prob. (sums to 1) |
| `P_exp(r)` | the experimental target distribution | prob. (sums to 1) |
| `w_m` | reweighted population of frame `m` | `w_m ≥ 0`, `Σ w_m = 1` |
| `P_w(r)` | ensemble model `Σ_m w_m P_m(r)` | prob. |
| `θ` | maximum-entropy confidence parameter (`THETA`) | > 0 |

**Stage 1 — back-calculation (rotamer convolution).** For frame `m`, every pair
of rotamers `(i, j)` contributes a distance `r_{ij} = ‖A_{m,i} − B_{m,j}‖`. We
histogram all `R²` distances onto the fixed `r`-axis and normalize:

```
P_m(b) = (1/Nm) · #{ (i,j) : distance_to_bin(‖A_{m,i} − B_{m,j}‖) = b }
```

where `Nm` is the number of in-window pairs (pairs outside `[r_min, r_max)` are
dropped, mimicking DEER's finite distance window). This is a discrete
**convolution** of the two rotamer clouds — exactly what tools like MMM compute,
just with a small equally-weighted library instead of a Boltzmann-weighted one.

**Stage 2 — maximum-entropy reweighting.** The ensemble model distribution is the
population-weighted mixture `P_w(b) = Σ_m w_m P_m(b)`. We want weights that fit the
data without straying needlessly from the prior populations `w⁰_m = 1/M`. Define
the objective (a Bayesian / EROS form):

```
L(w) = χ²(w) + θ · S_KL(w)
χ²(w)   = Σ_b ( P_w(b) − P_exp(b) )²              # data misfit (we use uniform bin weights)
S_KL(w) = Σ_m w_m · ln( w_m / w⁰_m )             # relative entropy to the prior (≥ 0)
```

`θ` is the knob: large `θ` keeps `w` near uniform (trust the simulation), small
`θ` lets the data dominate (trust the experiment). This is the same trade-off
BioEn exposes through its confidence parameter.

To keep `w` a valid probability vector during optimization we parametrize it by
**unconstrained log-weights** `g_m` through a **softmax**:

```
w_m = exp(g_m) / Σ_k exp(g_k)        ⇒  w_m > 0 and Σ_m w_m = 1 automatically.
```

We minimize `L` over `g` by gradient descent. The gradient is closed-form (§3),
so no autodiff is needed — and seeing it is the teaching point.

## 3. The algorithm

```
load ensemble (M frames × 2 rotamer clouds) and target P_exp        O(M·R)
STAGE 1 — back-calculation:
  for each frame m:                                                  ── parallel over m
      P_m  = histogram of the R² pairwise distances ‖A_i − B_j‖      O(R²) per frame
  total: O(M · R²)
STAGE 2 — reweighting (gradient descent on log-weights g, start g=0):
  repeat REWEIGHT_ITERS times:
      w      = softmax(g)                                            O(M)
      P_w(b) = Σ_m w_m P_m(b)                                        O(M · NBINS)
      G_k    = Σ_b 2(P_w(b) − P_exp(b))·P_m(b)  +  θ(ln(w_k/w⁰)+1)   O(M · NBINS)   # dL/dw_k
      grad_j = w_j ( G_j − Σ_k w_k G_k )                             O(M)           # softmax Jacobian
      g     -= LR · grad                                             O(M)
  total: O(ITERS · M · NBINS)
```

**Where the cost is.** Stage 1 is `O(M·R²)` and is **arithmetic-heavy** (each
distance is 3 subtractions, 3 multiplies, a sqrt). With a realistic library of
`R ≈ 200` rotamers and `M ≈ 10⁴`–`10⁵` frames, this is the dominant cost and is
*embarrassingly parallel across frames*. Stage 2 is `O(ITERS·M·NBINS)` but with
tiny constants (`NBINS = 50`, the per-step work is a couple of dot products); it
is cheap and sequential in the iteration index. **This asymmetry decides the GPU
mapping** (§4): GPU-parallelize stage 1; run stage 2 as shared host code.

**The gradient, derived.** With `m_b = P_w(b)` and `χ² = Σ_b (m_b − t_b)²`:
`∂χ²/∂w_k = Σ_b 2(m_b − t_b)·∂m_b/∂w_k = Σ_b 2(m_b − t_b)·P_k(b)`. For the entropy,
`∂/∂w_k [ Σ w ln(w/w⁰) ] = ln(w_k/w⁰) + 1`. Combine, then push the weight-space
gradient `G` through the softmax Jacobian `∂w_j/∂g_i = w_j(δ_ij − w_i)` to get the
log-space gradient `∂L/∂g_j = w_j(G_j − Σ_k w_k G_k)` (a mean-subtracted, weight-
scaled form). That last line is exactly `reweight_cpu`'s step-3 loop.

## 4. The GPU mapping

The pattern (PATTERNS.md §1, "the same expensive job for many members", as in
flagships `9.02`/`13.02` ensemble RK4 and `11.09` per-event work):

> **Stage 1: one MD frame per GPU thread.** Thread
> `m = blockIdx.x·blockDim.x + threadIdx.x` owns frame `m`. It reads the `m`-th
> slice of the rotamer arrays (`siteA + m·R`, `siteB + m·R`) and writes the `m`-th
> row of the `[M × NBINS]` histogram matrix via the shared
> `deer_member_histogram()`. **No two threads touch the same output row**, so
> there are **no atomics and no races** — pure data parallelism.

```
ensemble frames:   m=0      m=1      m=2            ...        m=M-1
                  ┌──────┐ ┌──────┐ ┌──────┐                  ┌──────┐
rotamer clouds →  │A0 B0 │ │A1 B1 │ │A2 B2 │      ...         │A.. B.│   (R points each)
                  └──┬───┘ └──┬───┘ └──┬───┘                  └──┬───┘
GPU thread     →   t0       t1       t2          ...           t(M-1)
                  R² conv  R² conv  R² conv                    R² conv
                    │        │        │                          │
hist rows [M×NBINS] ▼        ▼        ▼                          ▼
                  row 0    row 1    row 2         ...          row M-1     (disjoint → no atomics)
```

**Launch config.** `block = 256` threads (a solid occupancy default on
sm_75–sm_89); `grid = ceil(M / 256)`. Each thread does an `O(R²)` compute-bound
loop, so the kernel is **arithmetic-bound, not bandwidth-bound** — the exact block
size barely matters, and 256 keeps register pressure low. The rotamer arrays are
read coalesced per frame from **global memory**; the per-frame histogram lives in
**registers/local memory** during accumulation then is written once to global.
There is no shared memory because frames do not cooperate — the parallelism is
across frames, not within one.

**Why stage 2 is NOT on the GPU.** The reweighting touches an `M`-vector and a
`NBINS`-vector per step; on `M = 64`, `NBINS = 50` that is a few thousand flops
per iteration — a *rounding error* next to a single stage-1 launch. Shipping it to
the GPU would add kernel-launch latency and a non-deterministic float reduction
for *no* speed-up. So we run it as **shared host code** (`reweight_cpu`), fed
either the CPU or the GPU histograms; because those histograms are identical
(§5–6), both pipelines converge to the same weights. This "GPU the heavy
embarrassingly-parallel stage, keep the cheap glue on the host" split is the
realistic engineering choice and is itself a lesson. *(Scaling note: for `M ~ 10⁵`
the per-step `M·NBINS` work does grow enough to justify a GPU dot-product /
`cublasDgemv` for the mixture — left as an exercise.)*

**No CUDA library is linked.** The one heavy kernel is a hand-rolled histogram;
there is no FFT, eigensolve, or GEMM to delegate, so we add no cuBLAS/cuFFT/etc.
(unlike sibling `2.06`/`2.20`). The `.vcxproj` links only `cudart_static.lib`.

## 5. Numerical considerations

- **Precision: FP64 everywhere.** DEER distances are a few nm and the histogram
  counts are exact integers, but the reweighting accumulates thousands of small
  gradient steps; `double` keeps the CPU and GPU in lockstep and the descent
  stable. At this problem size FP64 is essentially free.
- **No atomics, deterministic by construction.** Each thread owns a disjoint
  histogram row, so stage 1 needs no `atomicAdd` and reorders nothing. The
  histogram value is `(integer count) × (1 / integer pair-count)` — the *same*
  exact operations on host and device — so the two match **bit-for-bit** (the demo
  reports `max |P_m cpu−gpu| = 0.0`). This is the PATTERNS.md §4 "exact when the
  same exact operations run on both sides" case, sidestepping the float-atomics
  determinism trap of `5.01`/`11.09` entirely.
- **Softmax stability.** `softmax_weights` subtracts the max log-weight before
  `exp` (the log-sum-exp trick) so no `exp` overflows even when descent drives some
  `g_m` large; the subtracted constant cancels in the ratio.
- **Reproducible stdout.** All run-varying numbers (timings) go to **stderr**; the
  deterministic result goes to **stdout**, which the demo diffs. Fixed iteration
  count + fixed learning rate + fixed synthetic seed ⇒ byte-identical output.

## 6. How we verify correctness

Two independent checks, layered:

1. **GPU ≡ CPU (implementation check).** `reference_cpu.cpp` back-calculates the
   histograms with a plain serial loop; `kernels.cu` does it one-frame-per-thread.
   Both call the *same* `deer_member_histogram()` from `deer.h`, so we expect — and
   get — agreement to `0.0` (tolerance `HIST_TOL = 1e-12`, pure double slack). Then
   we feed *both* histogram sets through the shared `reweight_cpu` and check the
   recovered weights agree (`WEIGHT_TOL = 1e-9`). An independent serial twin
   matching the parallel version is strong evidence neither has a logic bug.
2. **Recovering the known answer (science check).** The synthetic sample embeds a
   ground truth (PATTERNS.md §6): 16 of 64 frames are "true" matches at the
   target distance, the rest are decoys. A correct reweighting must move population
   onto the true frames. The demo reports `true-frame population: 0.2500 → 0.9895`
   and the `P(r)` peak snapping to the target's `3.45 nm` — the method found the
   right conformations without being told which they were. This validates the
   *science*, not just CPU==GPU agreement.

Edge cases handled: frames with no in-window pairs contribute an all-zero `P_m`
(dropped from the mixture); the loader validates the header against the compiled
`ROTAMERS_PER_SITE`/`NBINS` and rejects a zero-sum target.

## 7. Where this sits in the real world

This is a deliberately **reduced-scope teaching version** of a 🔴 frontier method.
Production EPR/DEER ensemble modelling differs in several ways:

- **The ensemble comes from real MD**, often with **soft DEER restraints applied
  during the simulation** (OpenMM, GROMACS), not a fixed pre-computed set. That
  restrained MD — thousands of replicas integrated for nanoseconds — is the part
  that truly needs a GPU MD engine; here we take the ensemble as given.
- **Rotamer libraries are Boltzmann-weighted and large** (~200 states with energies
  and clash screening against the local structure), as in **MMM** or
  **DEER-PREdict**. We use 24 equal-weight rotamers for clarity.
- **The DEER forward model is richer**: real pipelines fit the *time-domain* dipolar
  signal (with a background function), not just `P(r)`, and weight each bin by its
  experimental uncertainty in `χ²`. We compare `P(r)` directly with uniform
  weights.
- **Reweighting is Bayesian and scale-selected**: **BioEn** scans the confidence
  parameter `θ` and picks it by an L-curve / evidence criterion, and can reweight
  millions of frames. We fix `θ` and a small `M`.

What the teaching version preserves faithfully: the **rotamer-convolution back-
calculation**, the **maximum-entropy objective** (`χ² + θ·S_KL`), the **softmax
log-weight optimization**, and the **GPU mapping** (frame-parallel back-calc +
cheap host reweighting) that a production tool also uses.

---

## References

- **MMM — Multiscale Modeling of Macromolecules** (ETH Zürich):
  <https://www.epr.ethz.ch/software/mmm.html> — the canonical MTSSL rotamer
  libraries and DEER back-calculation; study how a real rotamer convolution is
  weighted and clash-screened.
- **BioEn / EnsembleFit**: <https://github.com/bio-phys/BioEN> — the reference
  Bayesian maximum-entropy reweighting; read it for the `θ` selection and the
  proper Bayesian derivation our `χ² + θ·S_KL` approximates.
- **DEER-PREdict** (Lindorff-Larsen lab; verify URL) — DEER/PRE prediction from MD
  ensembles; the practical forward model for `P(r)` from trajectories.
- **OpenMM** <https://github.com/openmm/openmm> — where soft DEER distance
  restraints would be applied *during* the GPU MD that generates the ensemble.
- **Jeschke, G. (2012), "DEER Distance Measurements on Proteins"**, *Annu. Rev.
  Phys. Chem.* — the standard primer on the experiment and `P(r)` extraction.
- **SASBDB** <https://www.sasbdb.org/> and **PDB** <https://www.rcsb.org/> — sources
  of EPR-constrained / EPR-refined structural models.
