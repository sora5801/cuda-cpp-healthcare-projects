# 6.26 — Virtual Population Generation & Sensitivity Analysis

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.26`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Before running a virtual clinical trial, quantitative pharmacologists ask two
questions: *how much does drug exposure vary across a diverse population?* and
*which physiological parameter drives that variability?* This project answers
both. It generates a **virtual population** by quasi-randomly sampling four
uncertain pharmacokinetic (PK) parameters — absorption `ka`, clearance `CL`,
distribution volume `V`, and bioavailability `F` — runs a PK model for each
virtual patient to compute drug exposure (area-under-the-curve, AUC), and then
performs a **global sensitivity analysis** (Sobol variance decomposition) to rank
the parameters by how much of the AUC variance each explains. The catch: Sobol
needs `N·(k+2)` independent model runs — tens of thousands here, millions in
practice — and every run is independent, which is exactly what a GPU accelerates.
We assign **one GPU thread per model evaluation**.

## What this computes & why the GPU helps

Virtual patient populations are created by sampling physiological parameter distributions (body weight, organ volumes, enzyme expression, sex, age) from measured databases (NHANES, WHO) and propagating them through PBPK/PD models to generate simulated trial cohorts. Sobol sensitivity analysis requires O(N×(2k+2)) model evaluations for k parameters—typically millions of forward ODE integrations. GPU batch simulation reduces this from days to hours.

**The parallel bottleneck:** the Saltelli estimator needs `N·(k+2)` forward
model solves (here `4096·6 = 24,576`; production QSP studies reach 10⁶+). Each
solve is a completely independent numerical integration of the PK
concentration–time curve — no shared state, no communication. That is textbook
*embarrassing parallelism*: the whole batch maps to a 1-D grid of GPU threads,
each thread owning one Saltelli sample. The subsequent Sobol reduction (turning
the AUC array into indices) is cheap and serial, so the GPU targets exactly the
part that dominates wall-clock time. On the test machine the GPU ran the batch
~30× faster than the single-threaded CPU reference.

## The algorithm in brief

Latin hypercube sampling (LHS), Sobol quasi-random sequences, Morris one-at-a-time elementary effects, Sobol variance-based sensitivity indices, polynomial chaos expansion (PCE), Gaussian process surrogate (emulator), MCMC parameter estimation (Metropolis-Hastings, NUTS), bootstrap confidence intervals.

In this teaching implementation, concretely:

- **Quasi-random sampling** — a Halton low-discrepancy sequence generates two
  independent sample matrices `A` and `B` (4 dims each), spreading points far
  more evenly than pseudo-random draws so the variance estimator converges fast.
- **Saltelli cross-sampling** — build `k` hybrid matrices `AB^{(j)}` (matrix `A`
  with only column `j` swapped from `B`); this is the `N·(k+2)` design.
- **Forward PK model** — a one-compartment oral model whose AUC is integrated by
  the trapezoid rule (and has a closed form `F·Dose/CL` used as an independent
  check).
- **Sobol estimators** — first-order `S_j` and total-order `ST_j` indices from
  the Saltelli/Jansen formulas.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/virtual-population-generation-sensitivity-analysis.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/virtual-population-generation-sensitivity-analysis.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\virtual-population-generation-sensitivity-analysis.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/vpop_config.txt`, prints the
Sobol sensitivity table and population summary, shows the GPU-vs-CPU agreement
check plus the analytic science cross-check (stderr), and prints a timing line.

## Data

- **Sample (committed):** `data/sample/vpop_config.txt` — a tiny study
  configuration (dose, 4 parameter ranges, horizon, sample size) so the demo runs
  with zero downloads. The virtual patients are generated deterministically
  inside the program (Halton sequence), so no patient table is shipped.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print pointers to real
  physiology/PBPK sources (documented, idempotent, no credential bypass).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: NHANES anthropometric/physiological data (https://www.cdc.gov/nchs/nhanes/); WHO growth reference datasets (https://www.who.int/tools/growth-reference-data-for-5to19-years); OSP PBPK model library (https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library); FDA drug label PK data (https://www.fda.gov/drugs).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program computes the `N·(k+2)` AUC evaluations on both the **GPU**
(`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`), then runs
the Sobol reduction on each. It asserts (a) the raw AUC arrays agree to `1e-9`
and (b) the Sobol indices agree to `1e-9` — that dual agreement is the
correctness guarantee. Beyond CPU==GPU, the run confirms an **analytic** fact:
because `AUC = F·Dose/CL`, the indices for `CL` and `F` must dominate
(`S(CL)+S(F) ≈ 0.99`) while `ka` and `V` are ~zero — printed as
`[science] ... -> CONSISTENT`.

## Code tour

Read in this order:

1. [`src/vpop.h`](src/vpop.h) — the shared `__host__ __device__` core: Halton
   sampling, the Saltelli matrix layout, and the PK forward model. Start here; it
   is the heart of the project and is compiled by *both* the CPU and the GPU.
2. [`src/main.cu`](src/main.cu) — loads the config, runs CPU + GPU, verifies both
   the raw array and the indices, reports.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-
   evaluation mapping.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline
   plus the Sobol/Saltelli reduction.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

SALib sensitivity analysis library (https://github.com/SALib/SALib) — Morris, Sobol, FAST methods for Python; Open Systems Pharmacology (https://github.com/Open-Systems-Pharmacology) — virtual population creation module (PK-Sim); mrgsolve (https://github.com/metrumresearchgroup/mrgsolve) — fast ODE PK batch simulation; SUNDIALS batch CVODE (https://github.com/LLNL/sundials) — GPU ODE ensemble.

- **SALib** — the reference open-source implementation of Morris/Sobol/FAST; our
  first-order and total-order estimators match its Saltelli/Jansen formulas.
  Cross-check your GPU indices against `SALib.analyze.sobol` on the same design.
- **PK-Sim / Open Systems Pharmacology** — how real virtual populations are built
  from measured physiology; the production analogue of our sampling step.
- **mrgsolve** — fast batch ODE PK simulation (the CPU-side workhorse this project
  moves onto the GPU).
- **SUNDIALS batch CVODE** — a stiff-aware GPU ODE ensemble solver; the
  drop-in for our fixed-step integrator when the model becomes stiff.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuRAND for Sobol/Halton quasi-random sequences; batch CVODE GPU for ensemble ODE; cuBLAS for PCE coefficient matrix operations; pattern: one CUDA thread per virtual patient, Sobol sensitivity via GPU-parallel model evaluations; thrust::transform for per-sample output extraction.

This teaching version realizes the **ensemble-evaluation** pattern (PATTERNS.md
§1, exemplars `9.02` SEIR and `13.02` PBPK): one thread per Saltelli evaluation,
all math in registers, no atomics, no shared memory. We hand-roll the Halton
sequence in a shared `__host__ __device__` header (PATTERNS.md §2) rather than
calling cuRAND so the CPU and GPU draw byte-identical points and verification is
exact; `THEORY.md` explains what cuRAND's Sobol generator would do instead.

## Exercises

1. **More parameters (bigger k).** Extend the model to `k=5` by making the dose
   uncertain, and update `VPOP_K` + the Halton dimensions. Watch the Sobol design
   grow to `N·(k+2)` and confirm the new parameter's index makes physical sense.
2. **cuRAND Sobol generator.** Replace the hand-rolled Halton draw with cuRAND's
   `curandGenerateUniformDouble` on a `CURAND_RNG_QUASI_SOBOL64` generator. Note
   how CPU/GPU parity now needs a tolerance instead of being exact — why?
3. **Morris screening.** Add the cheaper Morris "elementary effects" method
   (`O(N·(k+1))` runs) as a fast pre-screen and compare its ranking to Sobol's.
4. **Bootstrap confidence intervals.** Resample the `N` rows with replacement to
   put error bars on each `S_j` — how large must `N` be before `S(ka)`'s interval
   excludes 0.05?
5. **A model where interactions matter.** Make bioavailability depend on
   clearance (`F` scaled by `CL`) so `ST_j > S_j`; verify `sum(ST) > 1` and read
   off the interaction from the gap.

## Limitations & honesty

- **Synthetic & illustrative.** The parameter ranges are plausible but not fitted
  to any drug; the 1-compartment oral model is a teaching reduction of full PBPK
  (which uses ~15 physiological compartments with literature tissue volumes and
  blood flows). Nothing here is a pharmacokinetic prediction or clinical guidance.
- **Fixed-step, non-stiff.** We integrate AUC with a fixed-step trapezoid rule on
  a closed-form concentration curve. Real PBPK ODEs can be stiff and need an
  adaptive solver (CVODE); production Sobol wraps that solver, not a formula.
- **First/total order only.** We estimate `S_j` and `ST_j`, not the full set of
  higher-order interaction indices or a polynomial-chaos/GP surrogate (both listed
  in the catalog and left as extensions).
- **Sampling error.** Sobol indices are Monte-Carlo estimates; small negative
  `S(ka)`/`S(V)` values (like `-0.0001`) are sampling noise around the true 0,
  which is why the science check uses tolerances, not exact zeros.
