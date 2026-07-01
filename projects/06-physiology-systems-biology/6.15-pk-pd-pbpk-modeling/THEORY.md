# THEORY — 6.15 PK/PD & PBPK Modeling

> For a reader who knows C++ but is new to CUDA and to pharmacometrics. See
> [README.md](README.md) for the tour and build. _Educational only._

## 1. The science

When a drug is taken, two coupled stories unfold. **Pharmacokinetics (PK)** —
"what the body does to the drug" — is how the drug's concentration in plasma rises
(absorption from the gut) and falls (clearance by liver/kidney). **Pharmacodynamics
(PD)** — "what the drug does to the body" — is how that concentration moves a
biological response: a blood pressure, a clotting factor, a cholesterol level, an
enzyme activity. A **PK/PD model** links the two: the PK concentration curve is the
*input* that drives the PD response curve.

Because patients differ (body size, organ function, genetics), drug developers run
**virtual population** studies: simulate thousands of in-silico patients, each with
sampled physiology, to predict the *spread* of drug exposure and effect — who is
under-dosed, who is at risk of too much effect. That population of independent,
coupled ODE solves is the compute load of this project. **Physiologically-based**
PK (PBPK, project `13.02`) is the same idea with many tissue compartments; here we
keep the PK minimal and add the PD coupling that PBPK-only projects omit.

Our PD choice is the **indirect-response (turnover) model** (Dayneka, Garg & Jusko
1993), the workhorse of mechanistic PD. A biomarker `R` is *produced* at a constant
zero-order rate `kin` and *removed* at a first-order rate `kout`; left alone it sits
at the steady-state baseline `R0 = kin/kout`. Our drug **inhibits the removal** of
`R`, so while drug is present `R` climbs above baseline and then relaxes back as the
drug clears — a response that *lags* the concentration, which is exactly the
hallmark of real PD that a direct "effect ∝ concentration" model misses.

## 2. The math

Three state variables (amounts in mg, response in arbitrary units):

```
PK  (one-compartment, first-order oral absorption)
  dA_gut/dt = −ka · A_gut                              # gut depot empties
  dA_cen/dt =  ka · A_gut − CL · Cc                    # absorption in, clearance out
       Cc   =  A_cen / Vc                              # plasma concentration (mg/L)

PD  (indirect response: inhibition of loss)
  dR/dt     =  kin − kout · (1 − I(Cc)) · R            # production − (slowed) loss
       I(Cc) = Imax · Cc / (IC50 + Cc)                 # Emax/Hill inhibition, 0..Imax
```

Symbols: `ka` absorption rate (1/h); `CL` clearance (L/h); `Vc` central volume (L);
`kin` production rate (units/h); `kout` loss rate (1/h); `Imax ∈ [0,1]` maximum
fractional inhibition; `IC50` concentration for half-maximal inhibition (mg/L).
Initial conditions: `A_gut(0) = dose`, `A_cen(0) = 0`, `R(0) = R0 = kin/kout`.

**Outputs per patient.** PK exposure: `Cmax` (peak `Cc`), `Tmax` (its time), `AUC`
(area under `Cc(t)`, trapezoidal). PD effect: `Rmax`, its time `Tresp`, and the
peak fractional response `effect = (Rmax − R0)/R0`.

**A model-checking identity.** For complete absorption, integrating the central
equation gives **AUC = dose/CL**, independent of `ka` and `Vc`. The population mean
AUC must reproduce this — a check on the science, not just on CPU==GPU agreement.

## 3. The algorithm

```
for each virtual patient p:                          # INDEPENDENT -> parallel
    seed RNG from p
    sample ka, CL, Vc, IC50 ~ lognormal(median, CV)  # between-subject variability
    A_gut = dose;  A_cen = 0;  R = kin/kout           # initial conditions
    for each of `steps` time steps:                  # the coupled solve
        RK4 advance (A_gut, A_cen, R) by dt
        update Cmax/Tmax; accumulate AUC (trapezoid); update Rmax/Tresp
    effect = (Rmax − R0)/R0
```

Each patient is `O(steps)` RK4 steps of a tiny 3-ODE system; the population
multiplies by `n_patients`. Total serial work `O(n_patients · steps)`. The
**parallel depth** is just `O(steps)` (one patient's loop) because patients are
mutually independent — perfect data-parallelism, no communication, no reduction
during integration.

**RK4** evaluates the derivative four times per step (start, two midpoints, end)
and takes the weighted average `(k1 + 2k2 + 2k3 + k4)/6`; it is 4th-order accurate
(local error `~dt^5`) and stateless, ideal for a register-only per-thread loop.

## 4. The GPU mapping

**Decomposition — one thread per patient.** Thread `p = blockIdx.x·blockDim.x +
threadIdx.x` seeds a reproducible RNG from its index, samples its physiology, then
runs the *entire* coupled-PK/PD RK4 loop in **registers** and writes one
`PatientResult`. There is no global-memory traffic during integration and no
inter-thread communication — the kernel is **compute-bound**, the GPU's sweet spot.

```
grid  = ceil(n_patients / 128) blocks
block = 128 threads
  block 0            block 1            ...
 ┌───────────────┐  ┌───────────────┐
 │ t0 t1 ... t127│  │ t0 ...        │   each thread t_i integrates patient i
 └──┬────────────┘  └──┬────────────┘   fully in registers; writes results[i]
    │ pkpd_integrate()  │
    ▼                    ▼
 results[0..127]     results[128..255]  (device array -> copied back to host)
```

**Block size 128.** The RK4 loop keeps many `double`s live (three states plus
twelve `k` slopes), so the per-thread register footprint is non-trivial; 128
threads/block (4 warps) balances latency-hiding against register pressure so
occupancy stays healthy on sm_75–sm_89. A larger block can spill registers to slow
local memory; tune per GPU (see BUILD_GUIDE).

**Sampling on-device, no cuRAND.** Each patient's parameters are drawn log-normally
(`median·exp(CV·z)`, `z` standard-normal via Box-Muller) from a shared
`__host__ __device__` splitmix64 RNG in `pkpd.h`. Using a *shared deterministic*
RNG (rather than cuRAND) is the key trick (PATTERNS.md §2): the CPU reproduces the
identical population, so the two results match to round-off and verification is
**exact**, not statistical. Hand-rolling this is ~10 lines; the catalog's
cuRAND/SUNDIALS route trades that exactness for library convenience at scale
(§7).

**Memory hierarchy.** `PkPdParams` is a small POD passed **by value** into the
kernel (each thread reads its own copy from registers/constant, not global memory).
The only global-memory writes are the final `results[p]` — one struct per patient.
No shared memory, no atomics, no texture: the arithmetic *is* the workload.

## 5. Numerical considerations

- **Precision — double throughout.** Concentrations span orders of magnitude over
  hundreds of steps and `AUC` accumulates many small trapezoids; `double` keeps the
  CPU and GPU in lock-step and the metrics accurate. FP32 would drift between the
  two and blur the exact check.
- **Determinism.** No reductions or atomics during integration → each patient's
  result is independent of thread scheduling and reproducible bit-for-bit run to
  run. The population summary statistics are computed afterward on the host in a
  fixed order. STDOUT is therefore byte-identical every run (PATTERNS.md §3);
  timings go to STDERR.
- **The GPU FMA caveat.** Even in double precision, the GPU may fuse a
  multiply-add that the host compiler does not, so results can differ in the last
  bit or two, growing to ~`1e-12` over ~10³ steps. This is real and expected; we
  verify to `1e-6` (physically negligible for these units) and report the actual
  worst diff on STDERR (PATTERNS.md §4).
- **Stability.** RK4 is stable for these non-stiff PK/PD dynamics at `dt = 0.05 h`.
  Real PBPK can be **stiff** (fast tissue equilibria) and needs implicit solvers —
  Exercise 5 / §7.

## 6. How we verify correctness

`main.cu` integrates the population on CPU (`reference_cpu.cpp`, a plain serial
loop) and on GPU (`kernels.cu`, one thread per patient), then compares **every**
PK metric (Cmax/Tmax/AUC) and **every** PD metric (Rmax/Tresp/effect) of **every**
patient. The single worst absolute difference is the headline number; it must be
≤ `1e-6` (we observe ~`1e-12`). Because the two implementations are genuinely
independent code paths (serial host loop vs. massively-parallel kernel) that only
share the pure-math `pkpd.h`, their agreement is strong evidence the integration is
right. Beyond CPU==GPU, the result is pharmacologically sensible: plasma peaks after
oral absorption, the biomarker rises above baseline while drug is present, and the
**population mean AUC ≈ dose/CL** — the parameter-independent PK identity.

## 7. Where this sits in the real world

Production PK/PD and PBPK tools (PK-Sim/MoBi, Simcyp, mrgsolve, Pumas-AI, NVIDIA
nvQSP) go far beyond this teaching model: whole-body PBPK with ~15 physiological
compartments (liver, kidney, lung, fat, muscle, gut…), literature tissue
volumes/blood flows, compound-specific partition coefficients and saturable
Michaelis-Menten metabolism, richer PD (all four Jusko indirect-response variants,
transit-compartment absorption, tolerance/feedback), and **population fitting**:
nonlinear mixed-effects (NLME) estimation, empirical Bayes estimates, and full
Bayesian posterior sampling (HMC/NUTS). Those systems are often stiff (implicit
Rosenbrock/RODAS solvers, e.g. SUNDIALS batch CVODE on GPU) and use cuRAND for
Monte-Carlo sampling with *statistical* (not exact) comparison across runs. The
one-thread-per-subject ensemble integration you learn here is precisely the
parallel backbone those platforms accelerate — the model gets bigger, the parallel
pattern does not change.

---

## References

- Dayneka, Garg & Jusko (1993), *Comparison of four basic models of indirect
  pharmacodynamic responses* — the turnover PD model used here.
- Rowland & Tozer, *Clinical Pharmacokinetics and Pharmacodynamics* — PK/PD fundamentals.
- Jones & Rowland-Yeo (2013), *Basic concepts in PBPK modeling* — the multi-compartment extension.
- Press et al., *Numerical Recipes* — Runge-Kutta integration.
- **Open Systems Pharmacology / PK-Sim**, **mrgsolve**, **Pumas-AI**, **NVIDIA nvQSP**
  — production PK/PD/PBPK/QSP platforms; study their solver and population machinery.
