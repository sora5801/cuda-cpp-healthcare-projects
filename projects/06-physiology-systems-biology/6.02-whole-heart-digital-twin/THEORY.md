# THEORY — 6.2 Whole-Heart Digital Twin

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to cardiac modeling. See
> [README.md](README.md) for the quick tour and build steps.
>
> **Reduced-scope teaching version.** A production whole-heart twin is a 3-D
> finite-element PDE model on a patient mesh (research-grade). Per CLAUDE.md §13
> we ship the *simplest correct* model that still teaches every ingredient — a
> spatially-lumped (0-D) closed-loop heart — and describe the full approach in
> §7. _Educational only — not for clinical use._

---

## 1. The science

A **cardiac digital twin** is a computational model of a specific patient's heart,
calibrated so that its simulated behaviour matches that patient's clinical
measurements. Once calibrated, the twin can be *interrogated* — "what happens if
we pace here?", "how much has contractility dropped?" — in ways you cannot do on
a living patient. Building one couples four physical subsystems:

1. **Electrophysiology (EP).** Cardiac muscle cells are excitable: a stimulus
   triggers a fast depolarization (the **action potential**) that propagates as a
   wave and then slowly recovers. This is what an ECG measures.
2. **Active mechanics.** The electrical activation triggers **contraction**: the
   muscle stiffens and shortens, squeezing blood out of the ventricle. Mechanically
   the ventricle behaves like a chamber whose stiffness (**elastance**) rises and
   falls each beat.
3. **Circulation.** The ejected blood flows into the aorta and the systemic
   arteries, which act as an elastic reservoir with resistance — the classic
   **Windkessel** ("air chamber") load that sets arterial pressure.
4. **Inference.** The patient-specific parameters (contractility, resistances,
   conduction) are not measured directly; they are *fit* by running the forward
   model many times and adjusting until outputs match data.

This project models all four in **0-D** (one cell, one chamber, one arterial
compartment). The clinical readouts it produces — **stroke volume** (mL of blood
ejected per beat), **ejection fraction** (fraction of the filled volume ejected),
and the **pressure–volume loop** — are exactly the quantities a real twin is
calibrated against, so the *shape* of the problem is faithful even though the
spatial detail is dropped.

## 2. The math

The state is `y = (v, w, V, P)` and the model is a coupled ODE system
`dy/dt = f(t, y)` integrated in **milliseconds**.

**Electrophysiology — FitzHugh-Nagumo (FHN)**, a 2-variable reduction of the
Hodgkin-Huxley excitable cell:

```
dv/dt = c1 · v · (v − a) · (1 − v) − c2 · w + I_stim(t)
dw/dt = b · (v − d · w)
```

- `v` — fast (membrane-potential-like) variable, dimensionless, ~[−0.2, 1.1].
- `w` — slow recovery variable, dimensionless.
- `a` — excitation threshold; `b, c1, c2, d` — shape/time-scale constants.
- `I_stim(t)` — a periodic pacing pulse of amplitude `stim_amp` and width
  `stim_dur`, repeating every `bcl_ms` (basic cycle length; 800 ms ≈ 75 bpm).

**Mechanics — time-varying elastance.** The muscle activation `A(v) ∈ [0,1]`
(a clamped copy of `v`) ramps the ventricular elastance between its diastolic and
systolic values, and the ventricular pressure follows the end-systolic
pressure–volume relation:

```
E(t) = E_min + (E_max − E_min) · A(v)          [mmHg/mL]
P_lv = max(0, E(t) · (V − V0))                 [mmHg]
```

- `E_min` — relaxed (filling) elastance; `E_max` — **contractility** (the swept
  knob); `V0` — unstressed volume; `V` — ventricular volume [mL].

**Valves and volume balance.** Diode-like valves open only under a forward
pressure gradient (resistances converted from mmHg·s/mL to per-ms):

```
q_in  = (P_venous − P_lv)/(R_mitral·1000)  if P_venous > P_lv, else 0   [mL/ms]
q_out = (P_lv − P)/(R_aortic·1000)         if P_lv > P,        else 0   [mL/ms]
dV/dt = q_in − q_out
```

**Circulation — 3-element Windkessel.** The aorta is charged by ejected flow and
drains through the peripheral resistance into the compliant arterial compartment:

```
dP/dt = ( q_out − P/(Rp·1000) ) / C_art        [mmHg/ms]
```

- `Rp` — peripheral resistance; `C_art` — arterial compliance; the characteristic
  resistance `Rc` sets the fast ejection pressure kick.

**Outputs (per heart).** Over the final, converged beat we record EDV = max `V`,
ESV = min `V`, **SV = EDV − ESV**, **EF = SV/EDV**, and the peak `P_lv`, `P`.

**The inference objective.** Given a target stroke volume `SV*`, find the
contractility that minimises `|SV(E_max) − SV*|`. In this teaching version the
"optimizer" is an exhaustive scan of the ensemble (1-D grid search); §7 explains
what a real twin uses instead.

## 3. The algorithm

For each ensemble member (each contractility value):

1. Build its `HeartParams` (baseline physiology + its own `E_max`).
2. Set a physiological initial condition `(v, w, V, P)`.
3. **Integrate** `beats` cardiac cycles with classical **4th-order Runge-Kutta**
   (RK4): four derivative evaluations per step, combined for O(dt⁴) local error.
   RK4 is accurate and stable for these smooth ODEs.
4. Over the **final beat only** (transient washed out), track the volume/pressure
   extremes → one `TwinResult`.

Then a single pass over the `n` results picks the member closest to the target SV.

**Complexity.** One member costs `steps = beats · (bcl_ms/dt_ms)` RK4 steps
(here 6 · 8000 = 48 000), each O(1). The ensemble is `O(n · steps)` — and every
member is **independent**, so the *depth* (critical path) is just `steps` while
the *work* is `n · steps`. That gap is precisely what a parallel machine exploits.

## 4. The GPU mapping

This is the **ensemble-ODE pattern** (PATTERNS.md §1, exemplified by flagships
`9.02` SEIR and `13.02` PBPK): the same ODE solved for many parameter sets, one
GPU **thread per parameter set / trajectory**.

- **Thread-to-data map:** `idx = blockIdx.x · blockDim.x + threadIdx.x` owns
  ensemble member `idx` (one virtual heart). A guard `if (idx >= n) return;`
  handles the ragged last block.
- **What each thread does:** derives its `E_max`, then runs the *entire* multi-beat
  RK4 loop for its heart and writes one `TwinResult`. The time loop is sequential
  *within* a thread but fully parallel *across* threads.
- **Launch config:** `block = 128`, `grid = ceil(n/128)`. 128 (not 256) because
  RK4 over a 4-state system with four stages keeps many live registers; a smaller
  block relieves register pressure so occupancy is not capped, while still giving
  the scheduler 4 warps/block to hide latency.
- **Memory hierarchy:** the heavy state `(v, w, V, P)` and all intermediates live
  in **registers/local memory** for the whole run — there is no input array to
  stream and **no shared memory or atomics** needed. The only global-memory
  traffic is the tiny `TwinResult` each thread writes at the very end. This is a
  *compute-bound, register-resident* kernel, the opposite of a bandwidth-bound
  stencil.
- **No CUDA library is used here.** The catalog lists cuSPARSE/cuSOLVER/cuBLAS
  because the *full* FEM twin solves a large sparse linear system each timestep
  (see §7). Our 0-D model has no spatial coupling, hence no matrix to invert, so a
  hand-written kernel is both sufficient and more transparent. We flag this
  explicitly so the library omission is a *decision*, not an oversight.

```
grid of blocks (128 threads each)
┌──────── block 0 ────────┐ ┌──────── block 1 ────────┐
│ t0  t1  t2 ...   t127   │ │ t128 ...          t255  │  ...
│ ↓   ↓   ↓         ↓     │ │  ↓                 ↓    │
│ heart heart ... heart   │ │ heart ...        heart  │   each thread:
│  0     1        127     │ │  128              255   │   full RK4 over `beats`
└─────────────────────────┘ └─────────────────────────┘   → one TwinResult
```

## 5. Numerical considerations

- **Precision: FP64 (double) throughout.** The RK4 combination and the elastance
  products need the dynamic range; single precision would drift visibly over
  48 000 steps. (The catalog mentions FP16 forward passes — that is a
  performance trick for the *large FEM* twin, not appropriate for a 4-state ODE.)
- **Determinism.** Every operation is plain double arithmetic in a **fixed order**;
  there are **no atomics and no parallel reductions**, so a member's result is
  bit-identical run to run and across the Release/Debug builds. stdout is
  therefore reproducible and safe to diff (PATTERNS.md §3). All run-varying data
  (timings) go to stderr.
- **CPU/GPU parity.** The physics (`heart_deriv`, `rk4_step`, `simulate_heart`)
  lives in the single `__host__ __device__` header `src/heart.h`, so the CPU
  reference and the kernel execute the *same* instructions. The only possible
  divergence is the GPU's fused-multiply-add (FMA) reassociating `a*b+c`, which
  shifts results by a few ULPs — see §6.
- **Stability.** `dt = 0.1 ms` is comfortably inside the stability region for the
  FHN + elastance dynamics; the valve `if` branches are continuous in value (flow
  → 0 at the threshold) so they do not introduce stiffness.

## 6. How we verify correctness

`main.cu` runs the ensemble on **both** the CPU (`integrate_cpu`, a plain serial
loop) and the GPU (`integrate_gpu`, one thread per member), then compares **every**
clinical output of **every** member and reports the single worst absolute
difference.

- **Tolerance: `1e-9`** (mL / mmHg). Because both paths run the identical shared
  double-precision RK4, they agree to a few ULPs; we observe a worst diff of
  ~`5.7e-14`. The `1e-9` bound absorbs the GPU's FMA reassociation (~`1e-12`/step)
  with margin, yet is *billions* of times smaller than any physiological
  significance — so `PASS` is meaningful, not a rubber stamp (PATTERNS.md §4,
  "exact/near-exact" tier).
- **Physical sanity as a second check.** The demo output tells a *biologically
  correct story*: as contractility `E_max` rises, ESV falls, so **SV and EF
  increase and peak pressures rise** — the Frank-Starling / elastance response.
  EF spans ~39–63% and SV ~55–90 mL, physiologically plausible ranges. A model
  that passed GPU==CPU but produced nonsense physics would still be wrong; this
  cross-check guards that.
- **The twin-fit** recovers a sensible answer: for target SV = 70 mL it selects
  the member with SV = 69.3 mL (error 0.74 mL), the closest available grid point.

Agreement between an *independent* serial implementation and the parallel one is
strong evidence the parallelization introduced no bug (race, indexing, guard).

## 7. Where this sits in the real world

Production cardiac-twin frameworks differ from this teaching model in scale, not
in spirit:

- **Spatial EP.** openCARP and TorchCor solve the **monodomain/bidomain**
  reaction–diffusion PDE on a **patient tetrahedral mesh** (millions of nodes),
  with realistic ionic cell models (ten Tusscher, O'Hara-Rudy) — not FHN. Each
  timestep couples neighbours via a **sparse stiffness matrix**, so the diffusion
  solve is a large sparse linear system → **cuSPARSE + cuSOLVER/CG**; the ionic
  ODEs are a **batched per-node kernel** (that batch is the direct analogue of our
  per-member kernel). Our 0-D model drops the diffusion term entirely, which is
  why we need no sparse solver.
- **Mechanics.** simcardems (FEniCS) solves **nonlinear finite elasticity** with an
  **active-stress/active-strain** law on the same mesh — a Newton iteration with a
  sparse tangent each step. We replace this with a single time-varying-elastance
  chamber.
- **Fibers.** Real twins assign myocardial fiber orientation via the **rule-based
  Bayer-Blake-Plank (BBP)** Laplace solves or DTI. A 0-D model has no space, so no
  fibers.
- **Inference.** Instead of a grid search over one parameter, real twins use the
  **ensemble Kalman filter** or **adjoint/gradient-based** optimization through the
  forward model (TorchCor makes the whole solver differentiable) to fit *dozens* of
  parameters — running **thousands to millions of forward solves**. That ensemble
  is exactly why the GPU's per-member parallelism matters at scale, and it is the
  concept this project isolates and teaches.

To grow this project toward the real thing: replace the single chamber with a few
coupled compartments (both ventricles + atria), replace FHN with a proper ionic
model, then add a 1-D cable (diffusion) — at which point a sparse solver enters and
the cuSPARSE/cuSOLVER story becomes real.

---

## References

- **openCARP** — https://git.opencarp.org/openCARP/openCARP — the EP engine used
  in many published twins; study its monodomain solver and ionic model library.
- **simcardems** — https://github.com/ComputationalPhysiology/simcardems — how EP
  and finite-element mechanics are coupled in FEniCS.
- **TorchCor** — https://github.com/sagebei/torchcor — a differentiable, GPU
  PyTorch cardiac EP FEM; the model for gradient-based twin fitting.
- **Awesome-Cardiac-Digital-Twins** — https://github.com/lileitech/Awesome-Cardiac-Digital-Twins
  — curated index of datasets, methods, and papers.
- **FitzHugh (1961) / Nagumo (1962)** — the excitable-cell reduction used here.
- **Suga & Sagawa (1970s)** — the time-varying-elastance ventricle model.
- **Westerhof et al. (2009)** — "The arterial Windkessel", the 3-element load.
