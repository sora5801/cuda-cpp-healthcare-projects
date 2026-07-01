# 6.15 — PK/PD & PBPK Modeling

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟢 Beginner · Established** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.15`
>
> _Educational only — not for clinical/dosing use (see CLAUDE.md §8)._

## Summary

Simulate a **virtual population PK/PD study**: for thousands of virtual patients,
solve a *coupled* pharmacokinetic–pharmacodynamic ODE system and summarize each
patient's drug exposure and biological response. The **PK** half (a one-compartment
oral model) tracks the plasma concentration after an oral dose; the **PD** half (an
indirect-response turnover model) tracks a biomarker whose loss the drug inhibits,
so the concentration curve *drives* the response curve. Every patient is an
independent ODE solve, so each GPU thread integrates one patient with RK4 — the
ensemble-ODE pattern (cf. flagships `9.02`, `13.02`). The one idea this project
teaches beyond a pure-PK model is the **PK→PD coupling**: solving "what the body
does to the drug" and "what the drug does to the body" together.

## What this computes & why the GPU helps

PK/PD and PBPK models are compartmental ODE systems describing drug absorption,
distribution, metabolism, and excretion, plus the drug's downstream effect.
Population analysis solves the model for hundreds–thousands of individuals with
Monte-Carlo-sampled parameters — perfectly GPU-parallel; reported speedups reach
10–100× for population simulation and Bayesian sampling.

**The parallel bottleneck** is the *ensemble of RK4 integrations*. Each of the
`n_patients` patients runs a full time loop (here 960 RK4 steps of a 3-state
system) that is independent of every other patient. That is thousands of
independent trajectories — one GPU thread each, the whole loop in registers, no
inter-thread communication. The integration dominates runtime and is exactly what
we parallelize.

## The algorithm in brief

- **Sample** each patient's PK/PD physiology (`ka, CL, Vc, IC50`) log-normally
  around the population medians (between-subject variability).
- **Couple & integrate** the 3-state ODE with RK4:
  - PK: `dA_gut/dt = −ka·A_gut`, `dA_cen/dt = ka·A_gut − CL·Cc`, with `Cc = A_cen/Vc`.
  - PD: `dR/dt = kin − kout·(1 − I(Cc))·R`, inhibition `I(Cc) = Imax·Cc/(IC50+Cc)`.
- **Summarize**: PK exposure (Cmax, Tmax, AUC) and PD effect (Rmax, its time, and
  the peak fractional rise above baseline `R0 = kin/kout`).

See [THEORY.md](THEORY.md) for the science → math → algorithm → GPU-mapping → numerics.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/pk-pd-pbpk-modeling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/pk-pd-pbpk-modeling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\pk-pd-pbpk-modeling.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (optional CMake build)
```

The demo builds if needed, integrates the virtual population on CPU + GPU, verifies
the per-patient PK **and** PD metrics agree, prints sample patients and the
population summary, and shows a timing line on stderr.

## Data

- **Sample (committed):** `data/sample/pkpd_params.txt` — the population config
  (one line), so the demo runs offline with zero downloads.
- **Full dataset:** there is nothing to download (the population is generated);
  `scripts/download_data.ps1` / `.sh` point to real PK data and validated models.
- **Provenance & license:** see [data/README.md](data/README.md). The config is
  **synthetic** and labeled so.

Catalog dataset notes: PhysioNet MIMIC clinical PK data (<https://physionet.org>);
FDA FAERS (<https://www.fda.gov/drugs/fda-adverse-event-reporting-system-faers>);
OSP PBPK Model Library
(<https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library>); DDMoRe
(<https://ddmore.eu/models-tools>).

## Expected output

Success looks like `demo/expected_output.txt`. The program integrates the
population on both the **GPU** (`src/kernels.cu`) and a **CPU reference**
(`src/reference_cpu.cpp`), which share the model + RK4 + RNG (`src/pkpd.h`) in
**double precision**, so per-patient metrics agree to ~machine precision (worst
diff ≈ `1e-12`, verified to a `1e-6` tolerance). Two science checks confirm the
model, not just CPU==GPU agreement: the population **mean AUC ≈ dose/CL = 20**
mg·h/L (the standard PK identity), and the biomarker rises above its baseline
`R0 = kin/kout = 50` so the mean PD **effect** is positive.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — load config, CPU + GPU integrate, verify, print samples + summary.
2. [`src/pkpd.h`](src/pkpd.h) — **the coupled PK/PD model + RK4 + per-patient sampling** (host + device).
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU population interface (one thread per patient).
4. [`src/kernels.cu`](src/kernels.cu) — the population kernel + host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline + config loader.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, host I/O helpers.

## Prior art & further reading

- **Open Systems Pharmacology Suite** (<https://github.com/Open-Systems-Pharmacology>)
  — PK-Sim + MoBi, the whole-body PBPK platform.
- **mrgsolve** (<https://github.com/metrumresearchgroup/mrgsolve>) — R-based ODE
  PK/PD simulation; study its indirect-response model library.
- **Pumas-AI** (<https://pumas.ai>) — Julia pharmacometrics with GPU-accelerated
  population PK.
- **GillesPy2** (<https://github.com/GillesPy2/GillesPy2>) — stochastic PK variant simulation.
- Dayneka, Garg & Jusko (1993), *Comparison of four basic models of indirect
  pharmacodynamic responses* — the turnover model used here.
- Rowland & Tozer, *Clinical Pharmacokinetics and Pharmacodynamics* — PK/PD fundamentals.

Study these to learn the production approach; reimplement the pattern didactically
(CLAUDE.md §2), do not copy code wholesale.

## CUDA pattern used here

**Ensemble ODE integration over a virtual population** (PATTERNS.md §1): one thread
per patient, the entire coupled-PK/PD RK4 loop in registers, no inter-thread
communication · a shared `__host__ __device__` model + splitmix64 RNG for exact
CPU/GPU parity (PATTERNS.md §2) · double precision throughout. (The catalog also
mentions SUNDIALS batch CVODE and cuRAND; we hand-roll RK4 and a shared RNG instead
so the CPU reference reproduces the population exactly — see THEORY §7.)

## Exercises

1. **Stimulation instead of inhibition.** Change the PD term to *stimulate*
   production (`kin·(1+S(Cc))`) — one of the four Jusko indirect-response models —
   and compare the response shape.
2. **Hill coefficient.** Add a Hill exponent `n` to `I(Cc) = Imax·Cc^n/(IC50^n+Cc^n)`
   for a steeper concentration–effect curve; see how the effect distribution shifts.
3. **Multiple doses.** Add a dosing schedule (repeated oral doses) and report
   steady-state Cmax/Ctrough and the biomarker's oscillation.
4. **Sensitivity.** Sweep the variability `cv` and measure its effect on the spread
   of AUC and of the PD effect (a Morris-screening flavour).
5. **Two-compartment PK.** Add a peripheral tissue compartment (as in `13.02`) so
   the plasma curve shows a distribution phase; the parallel pattern is unchanged.

## Limitations & honesty

- **Teaching reduction.** One-compartment oral PK + a single indirect-response PD
  state — a deliberate simplification of full PBPK (~15 tissue compartments) and
  QSP. The parallel pattern (one thread per subject) is the same at any scale.
- **Fixed-step RK4**, single oral dose; no stiff/adaptive handling. Real PBPK can be
  stiff (fast tissue equilibria) and needs implicit solvers (Rosenbrock/RODAS).
- **Synthetic, illustrative parameters** — sampled via a shared deterministic RNG
  for exact CPU/GPU parity, **not** a validated population model and **not** fitted
  to any drug. Outputs are a software demonstration, not a pharmacokinetic
  prediction, and not for any clinical/dosing use.
