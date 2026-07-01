# THEORY — 4.23 Arterial Spin Labeling & Perfusion Imaging

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

**Perfusion** is the rate at which arterial blood is delivered to a tissue's
capillary bed — for the brain, **cerebral blood flow (CBF)**, quoted in
mL of blood per 100 g of tissue per minute. Healthy grey matter runs ~60,
white matter ~20; strokes, tumors, and dementia change it. Measuring CBF usually
needs an injected tracer (contrast MRI, PET). **Arterial Spin Labeling (ASL)**
does it with **no contrast agent at all** — the tracer is the subject's own blood
water, magnetically tagged.

How the tag works:

1. **Label.** A radiofrequency pulse train inverts the spins of water protons in
   arterial blood in the neck (pseudo-continuous ASL, **pCASL**, tags for a
   duration τ ≈ 1.8 s). The tagged blood is now magnetically distinct.
2. **Wait (the post-labeling delay, PLD).** The labeled blood flows up and into
   the brain tissue. The transit takes the **arterial transit time (ATT)**, ~0.5–1.5 s,
   which varies by region and with vascular disease.
3. **Image (label).** Acquire an image; tagged blood has slightly perturbed the
   tissue signal.
4. **Control + subtract.** Acquire a second image with no labeling and subtract:
   the difference **ΔM = control − label** cancels the huge static tissue signal
   and leaves only the perfusion contribution — but ΔM is only **0.5–1%** of the
   raw signal, so many label/control pairs are averaged for SNR.

In **multi-delay ASL** we repeat at several PLDs, so each voxel gets an inflow
**curve** ΔM(PLD). The curve's *amplitude* encodes CBF and its *onset/shape*
encodes ATT — so from the curve we can recover both. Turning ΔM(PLD) into
(CBF, ATT) at every voxel is a **kinetic-model fit**, and that is exactly what
this project computes.

## 2. The math

**The forward model (Buxton general kinetic model, pCASL).** For a voxel with flow
`f` (here in mL/100g/min; converted to per-second `f_s = f/6000` inside the model)
and transit time `Δt` (ATT, s), the perfusion-weighted difference signal at
post-labeling delay `t` (PLD, s) is, for a single well-mixed compartment with
blood-T1 decay:

Let `q = 2 α M₀ (f_s/λ) T₁ᵦ`, where

| Symbol | Meaning | Units | Value used |
|---|---|---|---|
| `α` | labeling (inversion) efficiency | — | 0.85 |
| `M₀` | equilibrium blood magnetization | MR units | 1.0 |
| `λ` | blood–brain water partition coefficient | mL/g | 0.90 |
| `T₁ᵦ` | longitudinal relaxation time of blood | s | 1.65 |
| `τ` | labeling (bolus) duration | s | 1.80 |

Then

```
             ⎧ 0                                                        , t < Δt        (A)
ΔM(t;f,Δt) = ⎨ q·e^(−Δt/T₁ᵦ)·(1 − e^(−(t−Δt)/T₁ᵦ))                      , Δt ≤ t < Δt+τ (B)
             ⎩ q·e^(−Δt/T₁ᵦ)·e^(−(t−Δt−τ)/T₁ᵦ)·(1 − e^(−τ/T₁ᵦ))        , t ≥ Δt+τ      (C)
```

- **(A)** the labeled bolus has not yet arrived → no signal.
- **(B)** the bolus is arriving; more of it has flowed in as `t` grows (the
  `1 − e^(…)` build-up).
- **(C)** the whole bolus has been delivered; the magnetization now only relaxes
  with the blood T1.

`ΔM` is **linear in `f`** (through `q`) and **nonlinear in `Δt`**.

**The inverse problem (the fit).** Given the measured curve `y_j = ΔM_measured(PLD_j)`
at delays `PLD_0…PLD_{n−1}`, estimate `(f, Δt)` by nonlinear least squares:

```
minimize   S(f,Δt) = Σ_j ( ΔM(PLD_j; f,Δt) − y_j )²
over       f ≥ 0 ,  0 ≤ Δt ≤ 5 s
```

This is a 2-parameter fit per voxel, solved with the analytic Jacobian
`J_j = [ ∂ΔM_j/∂f , ∂ΔM_j/∂Δt ]`. `∂ΔM/∂f = ΔM/f` (linearity); `∂ΔM/∂Δt` is the
regime-B expression derived in `src/asl.h` (zero in A and, to first order, in C).

## 3. The algorithm

Per voxel we run **Levenberg-Marquardt (LM)**, the standard robust nonlinear
least-squares method that blends Gauss-Newton and gradient descent:

1. Evaluate residuals `r_j = model_j − y_j` and Jacobian rows `J_j`.
2. Form the 2×2 normal system `A = JᵀJ` and gradient `g = Jᵀr`.
3. Solve the **damped** system `(A + μ·diag(A)) d = −g` in closed form (2×2 inverse)
   for the step `d = (Δf, ΔΔt)`.
4. **Trial + accept/reject:** if the step lowers `S`, accept it and *shrink* μ
   (trust Gauss-Newton more); else reject and *grow* μ (take a smaller, safer step).
5. Repeat until the gradient ≈ 0 or `S` stops decreasing (or an iteration cap).

**Why Marquardt diagonal scaling** (μ·diag(A), not μ·I): `f` (~tens) and `Δt` (~1)
have Jacobian columns differing by ~1000×, so `A` is badly scaled. `μ·I` would swamp
the tiny CBF-curvature entry and cripple the CBF step; `μ·diag(A)` is scale-invariant.
(This project's earlier naive fixed-μ Gauss-Newton *failed* to converge for exactly
this reason — a real lesson, see §5.)

**Complexity.** Per voxel: `O(iters · n_plds)`, with `iters` ≈ 3–6 here and a couple
of trial evaluations per iteration; a handful of PLDs. Trivial per voxel — but a
whole brain is `V ≈ 10⁵–10⁶` voxels, so the serial cost is `O(V · iters · n_plds)`.
**Parallel work** is the same total; **parallel depth** is just one voxel's fit,
because the `V` voxels are independent. Arithmetic intensity is high (all the work is
in registers; the only global traffic is reading `n_plds` signals and writing one
result), so this is compute-bound, not bandwidth-bound — ideal for the GPU.

## 4. The GPU mapping

**Thread-to-data mapping.** One thread fits one voxel:
`v = blockIdx.x * blockDim.x + threadIdx.x`. Thread `v` reads voxel `v`'s signal row,
runs the entire LM loop in registers, and writes one `AslFit`. No inter-thread
communication, no shared-memory cooperation, no atomics — pure embarrassingly-parallel
independence (docs/PATTERNS.md rows 1 & 8).

**Launch configuration.** `block = 128` threads, `grid = ceil(V / 128)`. 128 is a
good default here: the fit is register-heavy (LM state + a few doubles), so a smaller
block keeps register pressure and occupancy balanced on sm_75…sm_89; we do not need a
huge block to hide memory latency because there is very little global memory traffic.

**Memory hierarchy.**
- **Constant memory** holds the **PLD schedule** (`__constant__ double c_pld[]`). Every
  thread reads the same `pld[j]` at step `j`; the constant cache **broadcasts** one
  value to all 32 lanes of a warp in a single transaction — the textbook use case
  (the same trick project 1.12 uses for its query fingerprint).
- **Registers** hold the whole per-voxel LM state (`f`, `att`, `A`, `g`, `μ`, …). This
  is where the fit lives and why it is fast.
- **Global memory** is touched only twice per voxel: read the `n_plds` measured
  signals, write one `AslFit`. Both are coalesced-friendly (contiguous rows).
- **Shared memory / atomics:** *none needed*. There is nothing to reduce across
  threads — a deliberate contrast to the atomic-reduction projects (5.01, 11.09).

```
                 constant memory: c_pld[0..n_plds-1]  (broadcast to every warp)
                          |  (read by all threads)
   grid ─┬─ block 0 ─┬─ thread 0  → voxel 0   ─ LM in registers ─→ fits[0]
         │           ├─ thread 1  → voxel 1   ─ LM in registers ─→ fits[1]
         │           └─ …         → …                              …
         ├─ block 1 ─── …         → voxel 128 …
         └─ …
   global memory:  signal[V*n_plds]  (in, one row per voxel) ,  fits[V]  (out)
```

**On the "cuBLAS for matrix products" catalog note (no black box).** The catalog
suggests cuBLAS for kinetic-model matrix products. For *this* 2-parameter model the
normal equations are a **2×2** system, so a batched cuBLAS/cuSOLVER call would cost
more (kernel-launch and setup overhead) than the closed-form inverse we hand-write —
and the hand-written version teaches the linear algebra explicitly. Where cuBLAS/
cuSOLVER *would* pay off: a **higher-parameter** model (macrovascular component,
dispersion, multi-compartment → 4–8+ unknowns), where each voxel's normal system is a
dense `p×p` solve; then `cusolverDnDpotrf`/`Dpotrs` (batched Cholesky) per voxel is the
right tool. See project 2.06 (`cuSOLVER Dsyevd`) and 1.08 (batched `Dsyevj`) for how a
batched dense solve is wired in.

## 5. Numerical considerations

- **Precision: FP64 (double) throughout.** The fit involves differences of nearly
  equal exponentials and a 2×2 solve whose conditioning is poor (the ~1000× column
  scale mismatch). Double precision keeps the normal-equation determinant meaningful.
  sm_75 runs FP64 slowly, but correctness-you-can-trust beats speed here (a teaching
  choice; a production fitter might use mixed precision with care).
- **Conditioning & the damping choice.** As §3 explains, the CBF/ATT scale mismatch
  makes plain Gauss-Newton with `μ·I` damping *fail*. **Marquardt scaling** (`μ·diag(A)`)
  is scale-invariant and converges in a few iterations. We also floor the diagonal so a
  zero-curvature direction (e.g. all PLDs past `Δt+τ`, where `∂ΔM/∂Δt ≈ 0`) still gets a
  finite, invertible damping term rather than a singular system.
- **Determinism.** Each voxel is fit by **one thread**, so there is **no cross-thread
  reduction and no atomics** — nothing reorders a floating-point sum. Within a thread,
  the residual/Jacobian accumulation over `j` is a fixed ascending loop, the 2×2 solve
  is closed-form, and the accept/reject test is a plain comparison. Therefore the GPU
  result is **bit-reproducible** run to run, and matches the CPU reference to a few ULPs
  (the only divergence is host-vs-device FMA contraction, ~1e-14 over this short
  computation). This is the "integer-free but still deterministic" case in
  docs/PATTERNS.md §3 — determinism comes from *no shared reduction*, not from fixed-point.

## 6. How we verify correctness

Two independent checks, both in `src/main.cu`:

1. **GPU == CPU (implementation check).** The CPU reference (`fit_cpu`) and the GPU
   kernel both call the **identical** `__host__ __device__` `asl_fit_voxel()` in
   `src/asl.h` (the HD-macro idiom, docs/PATTERNS.md §2). So their per-voxel `(CBF, ATT)`
   must agree to round-off. **Tolerance 1e-9** (observed ~7e-15). Agreement between an
   obviously-correct serial loop and the parallel kernel is strong evidence the GPU
   bookkeeping (indexing, constant memory, guards) is right.
2. **Fit recovers ground truth (science check).** The synthetic sample's curves are
   **noise-free** Buxton evaluations of known `(CBF_true, ATT_true)` per voxel (see
   `scripts/make_synthetic.py`, which mirrors `asl.h` exactly). A converged LM fit must
   return those truths. **Tolerance 1e-4** (observed ~1e-8) — loose enough to absorb the
   residual of a finite iteration cap, tight enough to prove the optimizer actually
   inverts the model. This is the "embed a known answer" design (docs/PATTERNS.md §6):
   it validates the *algorithm*, not just CPU==GPU agreement.

**Edge cases exercised by the sample:** short vs. long transit times (bolus fully
arrived in some voxels, still arriving in others → all three Buxton regimes appear),
and a ~4× CBF range, so the fit is tested across the model's regimes.

## 7. Where this sits in the real world

Production ASL analysis (**oxford_asl / FSL BASIL**) shares this project's spine — a
per-voxel kinetic-model fit — but adds substantial machinery this teaching version
omits:

- **Bayesian, not just least-squares.** BASIL uses **variational Bayes** with physio­
  logical **priors** on CBF and ATT. On noisy real data, priors regularize the fit and
  give uncertainty estimates; our plain LM would overfit noise. (Exercise 1 adds noise
  so you can see this.) The per-voxel Bayesian update is still embarrassingly parallel —
  BASIL's GPU port parallelizes across voxels exactly as we do.
- **Richer forward model.** Real fits add a **macrovascular (arterial) component**,
  **bolus dispersion**, and sometimes multiple compartments — raising the parameter count
  so a dense `p×p` per-voxel solve (cuSOLVER batched Cholesky) becomes worthwhile (§4).
- **T1 partial-volume correction** (grey/white/CSF fractions per voxel) and full
  preprocessing (motion correction, registration, calibration to M₀) — the domain of
  **ExploreASL**.
- **Upstream reconstruction.** 3-D multi-delay ASL with **compressed sensing** needs a
  per-timepoint **NUFFT** k-space reconstruction (the cuFFT/NUFFT bottleneck shared with
  CS-MRI) — handled by **BART** / **SigPy**, and out of scope here (we start from ΔM curves).

So this project is the **deterministic kinetic-fit core** that a full ASL pipeline wraps
with priors, a richer model, correction steps, and reconstruction.

---

## References

- **Buxton et al. (1998), "A general kinetic model for quantitative perfusion imaging
  with arterial spin labeling," MRM** — the forward model in §2.
- **Alsop et al. (2015), "Recommended implementation of ASL," MRM** — the consensus
  pCASL parameters (α, τ, T1, λ) used as fixed constants here.
- **FSL BASIL / oxford_asl** (<https://fsl.fmrib.ox.ac.uk/fsl/docs/physiological/basil.html>)
  — the production Bayesian fit; study its kinetic-model definitions and priors.
- **Levenberg (1944) / Marquardt (1963)** — the damped least-squares method in §3;
  Marquardt's diagonal scaling is the key to the scale-mismatch robustness (§5).
- **ExploreASL** (<https://github.com/ExploreASL/ExploreASL>) — end-to-end ASL pipeline
  (preprocessing, PVC, QC) to see what surrounds the fit.
- **BART** (<https://github.com/mrirecon/bart>), **SigPy**
  (<https://github.com/mikgroup/sigpy>) — GPU compressed-sensing reconstruction (the
  upstream stage).
