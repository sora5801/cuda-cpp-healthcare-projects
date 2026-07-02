# THEORY — 6.26 Virtual Population Generation & Sensitivity Analysis

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

Two patients given the *same* dose of the *same* drug can end up with very
different amounts of drug in their blood. Body weight, liver enzyme levels, gut
absorption, kidney function — all vary across people, and all feed into a
pharmacokinetic (PK) model that predicts the **concentration–time curve** of the
drug. Model-informed drug development uses this to build a **virtual population**:
thousands of simulated patients, each a plausible draw of the physiological
parameters, so a drug's behavior can be studied *in silico* before (and
alongside) real trials.

Once you have such a population, the natural question is **sensitivity**: of all
the uncertain parameters, which ones actually *matter* for the outcome we care
about? If clearance `CL` explains 80% of the variation in drug exposure and
absorption rate `ka` explains ~0%, then measuring `CL` more precisely in a real
patient is worth a lot, while pinning down `ka` is a waste of effort. **Global
sensitivity analysis** answers this quantitatively, and the gold-standard method
is the **Sobol variance decomposition**.

Our outcome variable is total drug exposure, the **area under the
concentration–time curve** (AUC, units mg·h/L). Our uncertain inputs are the four
classic PK parameters: absorption rate `ka`, clearance `CL`, distribution volume
`V`, and oral bioavailability `F`.

## 2. The math

**The PK model (one compartment, first-order oral absorption).** Drug enters a
central compartment of volume `V` at rate `ka` from the gut, and leaves by
first-order elimination at rate `kel = CL/V`. The plasma concentration is the
classic *Bateman function*:

```
C(t) = (F · Dose · ka) / (V · (ka − kel)) · ( e^{−kel·t} − e^{−ka·t} )
```

with the removable-singularity limit at `ka = kel` (the "flip-flop" case):

```
C(t) = (F · Dose · ka / V) · t · e^{−ka·t}.
```

**The outcome.** Total exposure integrates the curve over all time:

```
AUC = ∫₀^∞ C(t) dt = F · Dose / CL.
```

That closed form is the linchpin of this project: **AUC depends only on `F` and
`CL`** — `ka` and `V` cancel out entirely. We nonetheless compute AUC *numerically*
(trapezoid rule over `[0, t_end]`) so the code exercises a real forward model and
the closed form stays an *independent* correctness check.

**Sobol indices.** Treat the model output `Y = f(X₁,…,X_k)` as a random variable
induced by the random inputs `X_j`. The law of total variance gives the ANOVA-style
decomposition

```
Var(Y) = Σ_j V_j + Σ_{i<j} V_{ij} + … ,   where V_j = Var_{X_j}( E[ Y | X_j ] ).
```

- **First-order index** `S_j = V_j / Var(Y)`: the fraction of output variance
  explained by `X_j` *alone* (its "main effect").
- **Total-order index** `ST_j = 1 − V_{~j}/Var(Y)`: `X_j`'s main effect *plus all
  interactions* it participates in. Always `ST_j ≥ S_j`; if `Σ ST_j > 1` the model
  has interactions.

Symbols: `Dose` [mg], `ka` [1/h], `CL` [L/h], `V` [L], `F` [–], `kel` [1/h],
`t` [h], `N` = Saltelli base sample size, `k = 4` parameters.

## 3. The algorithm

We estimate the indices with the **Saltelli sampling scheme** (the estimator SALib
uses), which turns the abstract conditional variances into simple sums over model
evaluations.

1. **Sample two matrices.** Draw two independent `N×k` matrices `A` and `B` of
   points in the unit hypercube using a **Halton low-discrepancy sequence**
   (`vpop.h::vpop_unit`). Low-discrepancy points fill space more evenly than
   pseudo-random ones, so the estimator's variance falls like `~1/N` in practice
   rather than the `~1/√N` of plain Monte Carlo — fewer model runs for the same
   accuracy.
2. **Build the hybrid matrices.** For each parameter `j`, form `AB^{(j)}` = matrix
   `A` with *only* column `j` replaced by `B`'s column `j`.
3. **Scale to physical units.** Map each unit coordinate `u ∈ [0,1)` affinely into
   `[lo_j, hi_j]`.
4. **Evaluate the model** `f = AUC` on all `k+2` matrices: `A`, `B`, and the `k`
   hybrids. That is `N·(k+2)` independent runs.
5. **Reduce to indices** (Saltelli 2010 / Jansen estimators):

   ```
   f₀   = mean(f(A))
   Var  = (1/N) Σ_i (f(A)_i − f₀)²
   V_j  = (1/N) Σ_i f(B)_i · ( f(AB^{(j)})_i − f(A)_i )          → S_j  = V_j / Var
   VT_j = (1/2N) Σ_i ( f(A)_i − f(AB^{(j)})_i )²                 → ST_j = VT_j / Var
   ```

**Complexity.** Sampling + reduction are `O(N·k)`; the cost is dominated by the
`N·(k+2)` model solves, each `O(steps)`. Serial total: `O(N·(k+2)·steps)`. The
*parallel depth* is just `O(steps)` (one solve), because all `N·(k+2)` solves run
concurrently — the reduction adds an `O(N·k)` serial tail that is negligible.
Arithmetic intensity is high (each thread does hundreds of `exp()` calls and
writes a single double), so this is compute-bound, not memory-bound — ideal for a
GPU.

## 4. The GPU mapping

**Thread-to-data mapping.** Flatten the `k+2` matrices into one block index:
block `0 = A`, block `1 = B`, block `2+j = AB^{(j)}`. A global evaluation index
`g ∈ [0, N·(k+2))` decodes as `block = g / N`, `row = g % N`. **Thread `g` owns
evaluation `g`**: it builds its parameter vector from the Halton sequence,
integrates the PK curve in registers, and writes exactly one `out[g]`.

**Launch configuration.** `block = 128` threads (4 warps — enough to hide the
latency of the `exp()` pipeline while keeping many blocks resident on `sm_75`);
`grid = ceil(N·(k+2) / 128)` blocks. The index is computed in 64-bit because
`N·(k+2)` can exceed 2³¹ for large studies.

**Memory hierarchy.**
- **Registers:** the entire per-thread state (4 parameters, running AUC, loop
  scalars) lives in registers — no shared memory, no spills for our small model.
- **Global memory:** touched once, to write `out[g]`. Writes are fully coalesced
  because consecutive threads write consecutive `g`.
- **Constant/param space:** the tiny `VpopParams` POD is passed *by value* into
  the kernel, so every thread reads it from the fast constant-bank kernel-argument
  space — no explicit `__constant__` needed.
- No **atomics** and no inter-thread communication: the evaluations are
  independent, so there is nothing to synchronize.

```
grid of N*(k+2) threads (here 24576)         one thread g:
┌──────── block 0 = A ────────┐              g ──► (block=g/N, row=g%N)
│ g=0  g=1  ...        g=N-1  │                 │
├──────── block 1 = B ────────┤                 ├─ Halton draw -> params {ka,CL,V,F}
│ g=N  ...            g=2N-1  │                 ├─ integrate C(t), trapezoid -> AUC
├─── block 2 = AB^(ka) ───────┤                 └─ out[g] = AUC   (one global write)
│ ...                         │
├─── block 3 = AB^(CL) ───────┤              (Sobol reduction runs afterward on host)
│ ...    block 5 = AB^(F)     │
└─────────────────────────────┘
```

**Which library does what (no black boxes).** The catalog suggests **cuRAND** for
the quasi-random draw and **Thrust** for the reduction. cuRAND's
`CURAND_RNG_QUASI_SOBOL64` generator would fill the `A`/`B` matrices with Sobol
points on the device; hand-rolling it (as we do with Halton) means implementing a
radical-inverse / direction-number generator — a dozen lines, shown in `vpop.h`.
We hand-roll deliberately so the CPU reference and GPU kernel draw *byte-identical*
points (parity). A Thrust `reduce_by_key` could compute the block means/variances
on the GPU; our reduction is `O(N·k)` and trivially cheap, so we do it serially on
the host for clarity. In production the model solve would be a **CVODE batch ODE**
integration (SUNDIALS) rather than a closed-form curve.

## 5. Numerical considerations

- **Precision: FP64 throughout.** Sobol first-order indices are computed as
  `V_j = mean(B·(AB−A))`, a difference of nearly-equal large numbers when a
  parameter is unimportant; in FP32 the cancellation would swamp the true tiny
  `S(ka) ≈ 0` with noise. Double precision keeps the estimator honest.
- **The total-order (Jansen) form** `Σ(A−AB)²` is used precisely because it never
  subtracts two large *means* — it squares per-row differences first, which is far
  more stable than the naive `1 − V_{~j}/Var`.
- **Determinism.** There are **no atomics** and no floating-point reduction on the
  GPU: each thread writes one independent output, and the (associative-order-
  sensitive) summation happens serially on the host in a fixed order. Therefore
  the GPU output array and the derived indices are bit-reproducible run to run —
  the demo's stdout is byte-identical every time (PATTERNS.md §3).
- **Integration horizon.** `t_end = 72 h` is many elimination half-lives for our
  parameter ranges (`kel` from `CL/V` ≈ 0.06–0.4 /h), so the trapezoid AUC is a
  faithful approximation of `∫₀^∞`. Too short an horizon would truncate exposure
  and bias every index equally.

## 6. How we verify correctness

Two independent checks, both required to print `PASS`:

1. **CPU reference (`src/reference_cpu.cpp`).** `evaluate_cpu` runs the *identical*
   `vpop_eval()` from the shared header in a plain serial loop, and `compute_sobol`
   runs the Saltelli reduction. Because the GPU calls the same
   `__host__ __device__` functions in the same order, the raw AUC arrays and the
   derived indices agree to floating-point round-off. **Tolerance = 1e-9**
   (PATTERNS.md §4, "same exact operations, double precision"); the observed worst
   difference is ~1e-14, far below it. This is convincing because an independent
   serial implementation reproducing the parallel one rules out both algorithm and
   race-condition bugs.
2. **Analytic science check.** Because `AUC = F·Dose/CL`, a *correct* Sobol
   analysis must attribute ~all variance to `CL` and `F` and ~0 to `ka` and `V`.
   The run reports `S(CL)+S(F) = 0.986` and `|S(ka)|+|S(V)| = 0.0001` →
   `CONSISTENT`. This validates the *algorithm*, not merely CPU==GPU agreement —
   the strongest kind of test (a known-answer check).

Edge cases handled: the `ka = kel` removable singularity (analytic limit branch),
`Var(Y) = 0` (indices reported as 0), and malformed / non-physical config (loader
throws).

## 7. Where this sits in the real world

Production virtual-population + sensitivity workflows differ from this teaching
version in scale and fidelity:

- **The model.** Real PBPK has ~15 physiological compartments (liver, kidney,
  lung, fat, muscle, gut, …) with literature tissue volumes, blood flows, and
  compound-specific partition coefficients and metabolism. Tools: **PK-Sim /
  Open Systems Pharmacology**. Our single compartment is a deliberate reduction so
  the AUC has a closed-form check.
- **The population.** Instead of uniform priors, parameters are sampled from
  **measured distributions** (NHANES anthropometrics, WHO growth data) correlated
  by covariates (age, sex, weight). PK-Sim's virtual-population module does this.
- **The solver.** Stiff whole-body ODEs need an adaptive implicit integrator —
  **SUNDIALS batch CVODE** on the GPU, one trajectory per thread — not a fixed-step
  trapezoid on a formula.
- **The methods.** Beyond first/total-order Sobol, practitioners use **Morris**
  screening (cheap pre-filter), **polynomial chaos expansion** and **Gaussian-
  process surrogates** (to replace expensive model runs), **MCMC** (Metropolis-
  Hastings / NUTS) for parameter estimation, and **bootstrap** confidence intervals
  on the indices. **SALib** implements the sensitivity methods; our estimators
  match its Saltelli/Jansen formulas so you can cross-check.
- **Scale.** `O(N·(k+2))` with `N ~ 10⁴–10⁶` and `k ~ 10–50` reaches 10⁷+ solves —
  the regime where the GPU turns days into hours.

---

## References

- Saltelli, A. et al. *Variance based sensitivity analysis of model output.*
  Comput. Phys. Commun. 181 (2010) — the first-order/total-order estimators used here.
- Sobol, I. M. *Global sensitivity indices for nonlinear mathematical models.*
  Math. Comput. Simul. 55 (2001) — the variance-decomposition foundation.
- Jansen, M. J. W. *Analysis of variance designs for model output.* (1999) — the
  robust total-order estimator `(A−AB)²`.
- **SALib** — <https://github.com/SALib/SALib> — reference Morris/Sobol/FAST; match
  and cross-check our indices.
- **PK-Sim / Open Systems Pharmacology** — <https://github.com/Open-Systems-Pharmacology> —
  whole-body PBPK and virtual-population generation.
- **mrgsolve** — <https://github.com/metrumresearchgroup/mrgsolve> — fast batch PK ODE simulation.
- **SUNDIALS CVODE** — <https://github.com/LLNL/sundials> — GPU ensemble ODE solver.
- Halton, J. H. (1960) — low-discrepancy sequences; the sampling used in `vpop.h`.
