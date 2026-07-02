# 6.27 — Parameter Estimation & Data Assimilation for Physiological Models

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.27`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Given a **noisy aortic-pressure waveform** and the (known) blood inflow into the
arteries, estimate the two patient-specific parameters of a **Windkessel** model of
the arterial system: peripheral resistance **R** and arterial compliance **C**. We
do it with an **Ensemble Kalman Filter (EnKF)** — the workhorse of *data
assimilation*: keep an ensemble of candidate patients, march them all forward
through the physiological ODE (the **GPU-parallel forecast**), then nudge them
toward each new measurement (the cheap **host-side analysis**). The demo recovers R
and C to within a few percent from a deliberately-wrong prior — a self-contained
example of fitting an ODE model to a time series on the GPU.

## What this computes & why the GPU helps

Fitting ODE/PDE physiological models to patient-specific clinical data (ECG,
pressure waveforms, biomarker time series) requires **repeated forward simulation**
inside an inference loop. An Ensemble Kalman Filter runs an ensemble of 50–500 model
copies in parallel and updates them as observations arrive; the **forward
integration of that ensemble is the bottleneck**.

**The parallel bottleneck:** the *forecast* step — advancing every ensemble member's
ODE state forward one observation window with RK4. Members are completely
independent, so we assign **one GPU thread per member** and run its whole time loop
in registers. The *analysis* step (the Kalman correction) is, for this small state,
a few scalar covariance sums, so it stays on the host — exactly the catalog's
"ensemble-parallel forward solves + host-side EnKF analysis" pattern.

## The algorithm in brief

- **Model:** two-element Windkessel `C dP/dt = Q(t) − P/R` (aorta as an RC circuit).
- **Joint state:** augment the state with the parameters, `x = [P, log R, log C]`, so
  the filter estimates parameters and pressure together (log keeps R, C positive).
- **EnKF cycle per observation:** *forecast* every member (RK4, on the GPU) →
  *analysis* (stochastic/perturbed-observation Kalman update, on the host).
- **Estimate:** the ensemble-mean R and C after all windows.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/parameter-estimation-data-assimilation-for-physiological-models.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/parameter-estimation-data-assimilation-for-physiological-models.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\parameter-estimation-data-assimilation-for-physiological-models.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/enkf_config.txt`, prints the
recovered (R, C), shows the GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/enkf_config.txt` — a tiny **synthetic**
  twin-experiment config (a known true patient + assimilation settings) so the demo
  runs with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print pointers to real
  clinical waveform datasets (all credentialed) and never bypass registration.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: PhysioNet MIMIC clinical waveforms (<https://physionet.org>);
UK Biobank cardiac functional parameters (<https://www.ukbiobank.ac.uk>); Zenodo
cardiac mechanics emulation dataset (<https://zenodo.org/records/7075055>); openCARP
community experiments (<https://opencarp.org/community/community-experiments>).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): the filter
moves R from a 40%-off prior to ~0.1% error and C from 33%-off to ~2% error, and
prints `RESULT: PASS`. The program runs the whole filter twice — once with the
forecast on the **GPU** (`src/kernels.cu`) and once fully on the **CPU**
(`src/reference_cpu.cpp`) — and asserts the two final ensembles agree
member-for-member to ~round-off (`worst diff ≈ 1e-14`, tolerance `1e-6`). Because
both paths share the `__host__ __device__` integrator (`src/windkessel.h`) and the
one host-side analysis, that agreement is exact by construction.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads config, synthesizes observations, runs the
   CPU + GPU filters, verifies, and reports.
2. [`src/windkessel.h`](src/windkessel.h) — **the physiology**: the Windkessel ODE,
   the inflow waveform, RK4, and the shared per-member forecast (host + device).
3. [`src/reference_cpu.h`](src/reference_cpu.h) — the config, the deterministic RNG,
   and the reference interface.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — observation synthesis, initial
   ensemble, **the EnKF analysis step**, and the serial reference filter.
5. [`src/kernels.cuh`](src/kernels.cuh) / [`src/kernels.cu`](src/kernels.cu) — the
   GPU forecast kernel (one thread per member) and the GPU-path driver.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **SUNDIALS/CVODES** (<https://github.com/LLNL/sundials>) — sensitivity-aware ODE
  integrator; learn the adjoint/forward-sensitivity route to *gradient*-based fitting
  (an alternative to the ensemble route used here).
- **simcardems** (<https://github.com/ComputationalPhysiology/simcardems>) — a cardiac
  "digital twin" with parameter fitting; see how full 3-D models are calibrated.
- **SALib** (<https://github.com/SALib/SALib>) — global sensitivity analysis; learn to
  decide *which* parameters are worth estimating before you fit.
- **PyMC** (<https://github.com/pymc-devs/pymc>) — probabilistic programming (MCMC/VI);
  the fully-Bayesian sibling of the EnKF's Gaussian approximation.
- **Reference:** Evensen, *Data Assimilation: The Ensemble Kalman Filter* (2009).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Ensemble ODE forecast + host-side EnKF analysis.** One GPU thread per ensemble
member runs the entire RK4 window in registers (the compute-bound bottleneck the
catalog names); the tiny Kalman update is done on the host from the ensemble's own
sample covariance. The shared `__host__ __device__` integrator makes the GPU and CPU
forecasts bit-identical, so verification is exact (docs/PATTERNS.md §1, §2).

## Exercises

1. **Bigger ensemble.** Run `python scripts/make_synthetic.py --ensemble 1024
   --windows 80` and watch the posterior spread shrink and the GPU/CPU timing gap
   grow. How does the estimate error scale with ensemble size?
2. **Estimate a third parameter.** Add characteristic aortic impedance `Z` (the
   *three*-element Windkessel), extend the state to `[P, logR, logC, logZ]`, and see
   whether pressure-only observations can identify all three.
3. **Unscented Kalman Filter (UKF).** Replace the random ensemble with `2N+1`
   deterministic sigma points; compare accuracy at equal forecast cost.
4. **High-dimensional EnKF.** Assimilate a spatially-resolved PDE field (e.g. a 1-D
   arterial pressure wave) so the state is thousands of cells — now the covariance
   update is a genuine `N×N` operation and **cuBLAS/cuSOLVER earn their keep**.
5. **Ensemble collapse.** Turn off the observation perturbation in `enkf_analysis`
   and watch the posterior spread underestimate the true uncertainty — the classic
   failure the stochastic EnKF fixes.

## Limitations & honesty

- **Synthetic twin experiment.** The "measurements" are generated from a known true
  (R, C) plus Gaussian noise — there is no real patient data here, and the model is a
  deliberately simple 2-element Windkessel. Real arterial hemodynamics need
  3-element/4-element or distributed (1-D wave) models.
- **Host-side analysis on purpose.** For a 3-vector state the Kalman gain is a few
  scalars, so linking cuBLAS/cuSOLVER would be a black box over two lines of algebra
  (see THEORY §4). Their role is spelled out for the high-dimensional case (Exercise 4).
- **Per-window PCIe round-trip.** The teaching build re-uploads the ensemble each
  window; a throughput build would keep it resident on the device (THEORY §5).
- Fixed-step RK4, Gaussian noise, and an EnKF's Gaussian approximation — a
  particle filter or full MCMC would handle non-Gaussian posteriors. Not for any
  clinical decision.
