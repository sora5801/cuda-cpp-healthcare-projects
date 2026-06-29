# THEORY — 13.02 PBPK at Scale

> For a reader who knows C++ but is new to CUDA and to pharmacokinetics. See
> [README.md](README.md) for the tour and build. _Educational only._

## 1. The science

After a drug is taken, its concentration rises and falls as it is absorbed,
distributed into tissues, metabolized, and excreted. **Pharmacokinetics (PK)**
quantifies this; **physiologically-based** PK (PBPK) builds the model from actual
physiology — organs as compartments connected by blood flows, sized by tissue
volumes. Because patients differ (body size, organ function, genetics), drug
developers run **virtual population** studies: simulate thousands of in-silico
patients to predict the spread of exposures and the chance of toxicity or
under-dosing. That population of independent ODE solves is the compute load here.

## 2. The math

Our teaching reduction has three compartments — a gut **depot**, the **central**
(plasma) compartment (volume `Vc`), and a **peripheral** tissue compartment
(volume `Vp`) — with amounts `A_gut, A_cen, A_per` (mg):

```
dA_gut/dt = -ka·A_gut                                   # first-order absorption
dA_cen/dt =  ka·A_gut − CL·Cc − Q·(Cc − Cp)            # absorption − clearance − distribution
dA_per/dt =  Q·(Cc − Cp)                               # tissue distribution
        Cc = A_cen/Vc,  Cp = A_per/Vp
```

`ka` absorption rate, `CL` clearance, `Q` inter-compartment flow. The exposure
metrics are **Cmax** (peak `Cc`), **Tmax** (its time), and **AUC** (area under the
`Cc`–time curve). A key check: for complete absorption, **AUC = dose/CL** — a
model-independent identity our population mean should reproduce.

## 3. The algorithm

```
for each virtual patient p:                        # INDEPENDENT -> parallel
    sample ka,CL,Vc,Vp,Q ~ lognormal(median, CV)   # patient physiology
    A_gut=dose, A_cen=0, A_per=0
    for each step:  RK4 advance;  update Cmax/Tmax;  accumulate AUC (trapezoid)
```

**Complexity.** Each patient is `O(steps)` RK4 steps of a tiny 3-ODE system; the
population multiplies by `n_patients`. Patients are independent — perfect
parallelism.

## 4. The GPU mapping

**Decomposition.** One thread per patient. Thread `p` seeds a reproducible RNG
from its index, **samples** its parameters, then runs the **entire RK4 loop in
registers** and writes one `PatientResult`. No global-memory traffic during
integration, no inter-thread communication — compute-bound, the GPU's sweet spot.

**Sampling on-device.** Each patient's physiology is drawn log-normally
(`median · exp(CV·z)`, `z` standard-normal via Box-Muller) from a shared
`__host__ __device__` splitmix64 RNG. Using a *shared deterministic* RNG (not
cuRAND) means the CPU reproduces the identical population, so the two results match
to round-off (`~1e-14`) — the same exact-verification strategy as project 5.01.
(Production uses cuRAND with statistical, not exact, comparison.)

**Why double precision.** Concentrations span orders of magnitude over hundreds of
steps, and AUC accumulates many small contributions; double keeps the CPU and GPU
in lock-step and the metrics accurate.

## 5. Numerical considerations

- **Determinism:** no reductions/atomics during integration → reproducible and
  CPU-matching. Population summary stats are computed afterward on the host.
- **Stability:** RK4 is fine for these non-stiff PK dynamics at the chosen `dt`.
  Real PBPK can be **stiff** (fast tissue equilibria), needing implicit solvers
  (Rosenbrock/RODAS — nvQSP) — Exercise 2.
- **AUC:** trapezoidal integration of `Cc(t)`; finer `dt` improves it.

## 6. How we verify correctness

`main.cu` integrates the population on CPU and GPU and compares every patient's
Cmax, Tmax, and AUC (`worst diff ≈ 1e-14`). Beyond CPU/GPU parity, the result is
pharmacologically sensible: plasma peaks after absorption, and the **population
mean AUC ≈ dose/CL**, the parameter-independent identity — strong evidence the ODE
and integration are right, not just that two codes agree.

## 7. Where this sits in the real world

Full PBPK (PK-Sim, Simcyp, nvQSP) uses ~15 compartments with literature tissue
volumes/blood flows, compound-specific partition coefficients and enzyme kinetics
(saturable Michaelis-Menten metabolism), absorption models, and **quantitative
systems pharmacology (QSP)** coupling to disease models — often 30–60 ODEs per
subject, sometimes stiff, solved for large virtual populations and many compounds.
The one-thread-per-subject ensemble RK4 you learn here is exactly the parallel
backbone nvQSP accelerates.

## References

- Rowland & Tozer, *Clinical Pharmacokinetics and Pharmacodynamics*.
- Jones & Rowland-Yeo (2013), *Basic concepts in PBPK modeling*.
- NVIDIA **nvQSP** — GPU stiff-ODE solvers for QSP/PBPK populations.
- Press et al., *Numerical Recipes* — Runge-Kutta integration.
