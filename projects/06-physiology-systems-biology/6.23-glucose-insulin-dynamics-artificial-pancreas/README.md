# 6.23 — Glucose-Insulin Dynamics & Artificial Pancreas

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.23`
>
> _Educational only — not a medical device, not for any clinical decision (see CLAUDE.md §8)._

## Summary

Simulate an **artificial pancreas in-silico trial**: a whole cohort of virtual
type-1-diabetes patients, each with different physiology, eats a meal while a
closed-loop **PID controller** doses insulin to keep blood glucose in range. Each
patient is an independent closed-loop ODE simulation (**Bergman minimal model** +
RK4 + a discrete PID controller), so each **GPU thread runs one patient** — the
embarrassingly-parallel "one thread per virtual patient" pattern behind the
reinforcement-learning and uncertainty studies that modern glucose-control
research runs on GPUs. The demo simulates 1024 patients and reports time-in-range,
hypoglycemia risk, and insulin use per patient.

## What this computes & why the GPU helps

Type 1 diabetes management via a closed-loop artificial pancreas requires
simulating glucose-insulin dynamics (Bergman minimal model; the FDA-accepted
UVA/Padova simulator) for controller design, in-silico trials, and RL training.
Testing a controller means running it on **many** virtual patients — thousands to
millions across many meal scenarios.

**The parallel bottleneck:** the cohort of closed-loop ODE integrations. Each
patient's simulation is a sequential RK4 + PID time loop, but the patients are
mutually independent, so the GPU runs one per thread. Each thread keeps its whole
trajectory in registers and writes a single summary — compute-bound, negligible
memory traffic, near-perfect parallelism.

## The algorithm in brief

- **Bergman minimal model** — 3-state ODE (plasma glucose `G`, remote insulin
  action `X`, plasma insulin `I`); insulin sensitivity `SI = p3/p2`.
- **Meal appearance** — two-exponential gastric-emptying disturbance `Ra(t)`.
- **PID controller** — discrete closed-loop insulin dosing with clamping +
  anti-windup, zero-order-held between control ticks.
- **RK4** — fixed-step 4th-order integration (double precision).
- **Ensemble** — sweep insulin sensitivity `SI` × glucose effectiveness `SG`;
  collect min/max/mean glucose, time-in-range, hypoglycemia %, total insulin.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/glucose-insulin-dynamics-artificial-pancreas.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/glucose-insulin-dynamics-artificial-pancreas.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\glucose-insulin-dynamics-artificial-pancreas.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (optional CMake build)
```

The demo builds if needed, runs on `data/sample/cohort_params.txt`, prints the
per-patient results and cohort summary, shows the GPU-vs-CPU agreement check, and
prints a timing line on stderr.

## Data

- **Sample (committed):** `data/sample/cohort_params.txt` — a one-line synthetic
  cohort config (32×32 = 1024 virtual patients); runs offline, zero downloads.
- **Full dataset / simulators:** `scripts/download_data.ps1` / `.sh` print
  pointers to real CGM datasets and reference simulators (nothing to fetch to run).
- **Provenance & license:** see [data/README.md](data/README.md).

Real data & simulators: OhioT1DM (<https://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html>,
data-use agreement required); JAEB/DirecNet (<https://public.jaeb.org>); simglucose
(<https://github.com/jxx123/simglucose>).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program simulates the cohort on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`), which share the double-precision RK4 + PID
core in [`src/bergman.h`](src/bergman.h), and asserts the per-patient glucose
metrics agree within `1e-4` mg/dL (observed diff ≈ `1e-13`) — that agreement is
the correctness guarantee. The output shows more insulin-sensitive patients
reaching lower peaks, higher time-in-range, and needing less insulin.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the cohort, runs CPU + GPU, verifies, reports.
2. [`src/bergman.h`](src/bergman.h) — **the model**: ODE RHS, meal, PID, RK4, and
   `simulate_patient()` (shared `__host__ __device__` core for exact CPU/GPU parity).
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the cohort kernel (one thread per patient) + host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — loader + trusted serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **simglucose** (<https://github.com/jxx123/simglucose>) — Python UVA/Padova T1D
  simulator + RL gym; study its patient parameters and meal scenarios.
- **GluCoEnv** (<https://github.com/chirathyh/GluCoEnv>) — GPU-accelerated glucose
  control RL environment (the production form of this project's parallelism).
- **G2P2C** (<https://github.com/RL4H/G2P2C>) — an RL artificial-pancreas agent;
  the learned policy that would replace our PID.
- **OpenAPS oref0** (<https://github.com/openaps/oref0>) — a deployed open-source
  dosing algorithm; read it for real safety constraints.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Ensemble ODE integration** (docs/PATTERNS.md §1; same as flagships `9.02`,
`13.02`): one thread per virtual patient, the full RK4 + PID loop in registers, no
inter-thread communication, `CohortConfig` passed by value (no input copies) ·
shared `__host__ __device__` `simulate_patient()` for exact CPU/GPU parity ·
double precision for stability over thousands of steps.

## Exercises

1. **Meal-disturbance Monte Carlo.** Use **cuRAND** to give each patient a random
   meal size / timing, then report the *distribution* of time-in-range — a proper
   uncertainty-quantification study.
2. **Noisy CGM + Kalman filter.** Add measurement noise to the glucose the
   controller sees, then implement a Kalman filter for state estimation and show it
   restores control quality (the real sensing problem).
3. **Tune the controller per patient.** Sweep `Kp, Ki, Kd` and find, per insulin
   sensitivity, the gains that maximize TIR without causing hypoglycemia — the
   in-silico trial's actual purpose.
4. **Step-size study.** Increase `dt` until RK4 rings or blows up; contrast with an
   adaptive RK45 (Dormand-Prince) to see cost vs. accuracy.
5. **Swap PID for MPC.** Replace the PID with a short-horizon model-predictive
   controller that plans insulin over the next hour using the model itself.

## Limitations & honesty

- **Reduced-scope teaching model.** The 3-state Bergman minimal model stands in for
  the ~13-compartment FDA-accepted UVA/Padova simulator; parameters are **synthetic
  and illustrative**, not fitted to any patient.
- **Idealized sensing/actuation.** Glucose is read noiselessly (no CGM noise, no
  Kalman filter) and insulin acts through a single lumped compartment (no
  subcutaneous absorption delay). A plain PID stands in for production MPC/RL.
- **Fixed-step RK4**, single meal, deterministic (no physiological noise).
- **Not a medical device.** Output is a software demonstration of ensemble
  closed-loop ODE simulation — never for diagnosis, treatment, or dosing.
