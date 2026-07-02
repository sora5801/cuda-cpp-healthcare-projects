# THEORY — 6.27 Parameter Estimation & Data Assimilation for Physiological Models

> The deep didactic explanation (the "why"). Written for a sharp student who knows
> C++ but is new to CUDA and new to data assimilation. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

## 1. The science

Clinicians and modelers want **patient-specific** cardiovascular models: given a
person's measured pressure or flow waveforms, what are *their* arterial properties?
Two lumped quantities summarize a great deal of arterial physiology:

- **Peripheral resistance R** — how hard it is for blood to drain out of the arteries
  into the tissues (rises with vasoconstriction, hypertension).
- **Arterial compliance C** — how much the large arteries stretch to buffer each
  heartbeat (falls with age and arterial stiffening).

The **two-element Windkessel** ("air chamber" in German) is the classic minimal model:
the aorta and large arteries act like an elastic reservoir (a capacitor `C`) that fills
during systole and drains through the peripheral resistance `R` during diastole. It
famously explains the exponential pressure decay between beats.

The **inverse problem** — recover `(R, C)` from a noisy measured pressure waveform — is
*parameter estimation*, and doing it as observations stream in is *data assimilation*.
That is the compute task this project parallelizes.

## 2. The math

**Forward model.** With aortic pressure `P(t)` (mmHg) and a known ventricular inflow
`Q(t)` (mL/s), the two-element Windkessel is a single first-order ODE:

```
C dP/dt = Q(t) − P / R
```

- `R` = peripheral resistance (mmHg·s/mL), `C` = arterial compliance (mL/mmHg).
- `RC` = the diastolic decay time constant (s): when `Q = 0`, `P(t) = P0 e^(−t/RC)`.
- We model `Q(t)` as a per-beat half-sine ejection during systole and zero during
  diastole — a smooth, physiologically-shaped stand-in for a measured flow probe.

**Joint state-parameter estimation.** To estimate `(R, C)` with a *state* estimator we
**augment** the state with the parameters, and carry the parameters in **log space**:

```
x = [ P , θ_R , θ_C ]        with   R = e^(θ_R),  C = e^(θ_C)
```

Log space (i) keeps `R, C > 0` under the linear Kalman update and (ii) matches their
multiplicative biological variation. The parameters have trivial dynamics
`dθ/dt = 0`; they only move in the *analysis* step. So the augmented ODE is

```
dP/dt   = ( Q(t) − P/R ) / C ,   dθ_R/dt = 0 ,   dθ_C/dt = 0 .
```

**Observations.** At the end of window `k` we measure the pressure with noise:
`y_k = P(t_k) + ε_k`, `ε_k ~ N(0, R_o)`, observation operator `H = [1, 0, 0]` (we see
`P`, not the parameters). `R_o = obs_noise²` is the measurement-error variance.

**EnKF analysis (the estimator).** With an ensemble `{x_i}` of size `m`, sample mean
`x̄` and sample covariance `P_f`, the ensemble Kalman update of member `i` is

```
K   = P_f Hᵀ (H P_f Hᵀ + R_o)⁻¹          (Kalman gain; here a 3-vector)
x_i ← x_i + K ( y + ε_i − H x_i )         (ε_i ~ N(0, R_o), per member)
```

Because `H` selects `P`, the only covariances we need are `Cov(x_j, P)` for each
component `j`, and `H P_f Hᵀ = Var(P)` is a scalar — so the "matrix inverse" is a single
division. Perturbing the observation per member (`ε_i`) is the **stochastic EnKF**
(Burgers et al. 1998); it keeps the posterior spread statistically correct.

## 3. The algorithm

One **assimilation cycle** per observation window `k = 0 … n_obs−1`:

1. **Forecast.** Advance every member's state one window (`substeps` RK4 steps of size
   `dt`) by integrating the augmented ODE. `RK4` samples the derivative four times per
   step for `O(dt⁴)` local error — accurate and stable for this smooth, non-stiff ODE.
2. **Analysis.** Compute `x̄`, the cross-covariances `Cov(x_j, P)`, the gain `K`, and
   pull every member toward its perturbed observation.

After all windows, the estimate is `R̂ = e^(mean θ_R)`, `Ĉ = e^(mean θ_C)` (a geometric
mean — the right average for a multiplicative quantity), with the ensemble std as the
posterior uncertainty.

**Complexity.** Let `m` = ensemble size, `W` = `n_obs` windows, `s` = substeps, `n` =
state dim (= 3). Forecast is `O(m · W · s · n)` — dominated by `m` (hundreds) times the
per-member RK4 work, and **fully parallel over members**. Analysis is `O(m · n)` per
window (a few sums) — negligible here. So the forecast is the target, and its
**work is embarrassingly parallel**: parallel depth is just `O(W · s)` per member.

## 4. The GPU mapping

**Decomposition.** *One thread per ensemble member.* Thread `idx` loads member `idx`'s
augmented state `ens[idx·3 .. idx·3+2]` into registers, runs the **entire RK4 window in
registers/local memory**, and writes the state back. No inter-thread communication, no
shared memory, no atomics — the textbook "ensemble ODE" pattern (docs/PATTERNS.md §1),
the same one as flagships `9.02` (SEIR) and `13.02` (PBPK).

```
  member 0  [P,θR,θC] ── RK4 window ──▶ [P',θR,θC]     (thread 0)
  member 1  [P,θR,θC] ── RK4 window ──▶ [P',θR,θC]     (thread 1)
     ⋮                                                   ⋮  (all in parallel)
  member m-1 ──────────── RK4 window ──▶ ...            (thread m-1)
        │
        └── D2H ──▶  host EnKF analysis (mean, Cov, gain, update)  ──▶ next window
```

**Launch config.** `block = 128` threads (a multiple of the 32-lane warp; enough warps
to hide latency while the per-thread state + RK4 scratch keep register pressure in
check), `grid = ceil(m / 128)` blocks. The ragged last block is guarded by
`if (idx >= m) return;`.

**Memory.** The only global traffic is one load and one store of `3` doubles per member
per window; the integration itself lives entirely in registers, so the kernel is
compute-bound — the regime where the GPU's many cores win. (The demo's per-window
H2D/D2H copy is a teaching simplification; see §5.)

**Where are cuBLAS / cuSOLVER / Thrust?** The catalog lists them for the analysis step,
and they are the right tools **when the state is high-dimensional** (assimilating a whole
PDE field: `P_f` becomes a real `N×N` matrix, the gain a real linear solve — cuBLAS
`gemm` + cuSOLVER `potrf/potrs`; particle-filter resampling → Thrust scan). For our
**3-vector** state the covariance is a handful of scalar sums and the "inverse" is one
division, so shipping it to a library would be a black box over two lines of arithmetic
(CLAUDE.md §6.1.6). We do it on the host, in plain code you can read, and point at the
library route in README Exercise 4. **The expensive, parallel part — the forecast — is
what we put on the GPU.**

## 5. Numerical considerations

- **Precision: FP64.** Pressures span tens of mmHg while the log-parameters live near 0;
  double precision keeps the ensemble statistics clean over many windows and keeps the
  CPU and GPU forecasts in lock-step.
- **Determinism.** No atomics, no parallel float reductions on the device — each thread
  writes only its own member. All host-side sums (mean, covariance) run in a fixed member
  order, and every random draw comes from a **deterministic SplitMix64** stream seeded
  from the config (we roll our own Box-Muller because `std::normal_distribution` is *not*
  bit-reproducible across compilers). Result: stdout is byte-identical every run
  (docs/PATTERNS.md §3), and the CPU/GPU paths, sharing seeds, match exactly.
- **The exact-match trick.** The forecast RK4 lives in one `__host__ __device__` header
  (`windkessel.h`) and the analysis is one host function called by *both* paths. So the
  GPU differs from the CPU only in float **re-association** across the RK4 arithmetic —
  `≈ 1e-14` after 40 windows, far under the `1e-6` tolerance (docs/PATTERNS.md §4).
- **Teaching vs throughput.** We copy the ensemble host↔device each window to keep the
  analysis on the host and the code linear. A throughput build keeps the ensemble
  resident on the device and only ships the (tiny) observation down and statistics up.

## 6. How we verify correctness

Two independent checks:

1. **CPU vs GPU (implementation correctness).** `run_enkf_cpu` and `run_enkf_gpu` use the
   same initial ensemble (same seed), the same per-window analysis seeds, and the same
   shared integrator + analysis. `main.cu` compares the **final ensembles
   member-for-member**; the worst difference (`≈ 7e-14`) is round-off, comfortably under
   `TOLERANCE = 1e-6`. Agreement between an obviously-correct serial loop and the parallel
   kernel is strong evidence the kernel is right.
2. **Estimate vs truth (scientific correctness).** Because it is a *twin experiment*, we
   know the true `(R, C)` used to synthesize the observations. The filter starts from a
   prior that is 40% (R) and 33% (C) wrong and recovers R to ~0.1% and C to ~2%, with the
   final ensemble-mean pressure RMSE (~1.7 mmHg) near the measurement-noise floor (1.0
   mmHg). That is the *method* working, not just two codes agreeing. (C is less
   identifiable than R from end-of-window pressures alone — a real, honest lesson about
   observability, and the motivation for Exercise 2's richer observations.)

## 7. Where this sits in the real world

Production cardiovascular data assimilation adds:

- **Richer forward models** — 3/4-element Windkessel, or distributed **1-D wave-propagation
  PDEs** (pressure along the whole arterial tree). The state is then thousands of cells and
  the EnKF covariance update is a genuine dense/low-rank linear-algebra step — **cuBLAS +
  cuSOLVER**, exactly the catalog's note (Exercise 4).
- **Sensitivity-aware ODE solvers** (SUNDIALS/**CVODES**) for *gradient*-based fitting
  (adjoint / 4D-Var / L-BFGS), an alternative to the ensemble route.
- **Full digital twins** (**simcardems**) coupling electrophysiology + mechanics, calibrated
  against imaging.
- **Fully-Bayesian inference** (**PyMC**, MCMC/VI) for non-Gaussian posteriors that the
  EnKF's Gaussian approximation cannot capture; **GP emulators** to make each forward solve
  cheap enough to afford millions of them; **SALib** to decide which parameters are worth
  estimating at all.

The ensemble-over-threads forecast you learn here is the computational backbone under all
of these.

---

## References

- Frank, O. (1899) — the original Windkessel model of arterial hemodynamics.
- Westerhof, Lankhaar & Westerhof (2009), *The arterial Windkessel*, Med. Biol. Eng. Comput.
- Evensen, G. (2009), *Data Assimilation: The Ensemble Kalman Filter*, Springer — the
  canonical EnKF text.
- Burgers, van Leeuwen & Evensen (1998), *Analysis scheme in the EnKF* — the
  perturbed-observation (stochastic) update used here.
- Vigna, S. (2015) — SplitMix64, the deterministic PRNG we use for reproducibility.
- SUNDIALS/CVODES, simcardems, SALib, PyMC — see README "Prior art" for what each teaches.
