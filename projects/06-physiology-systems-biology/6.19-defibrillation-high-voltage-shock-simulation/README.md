# 6.19 — Defibrillation & High-Voltage Shock Simulation

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.19`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A defibrillator terminates a lethal arrhythmia by dumping a high-voltage shock
across the heart, forcing the tangled electrical activity to a halt so a normal
rhythm can resume. This project builds a **reduced-scope, GPU-accelerated
simulation of that idea**: a 1-D cardiac cable (the FitzHugh-Nagumo excitable
model) is set up with an ongoing travelling wave — a stand-in for fibrillation —
and we sweep a ladder of shock strengths to find the smallest one that
**defibrillates** it. That smallest successful strength is the **defibrillation
threshold (DFT)**, the number engineers care most about when designing
defibrillators and implantable cardioverter-defibrillators (ICDs). Because every
candidate shock strength is an independent simulation, the sweep maps perfectly
onto the GPU: **one thread runs one whole cable for one shock strength.**

## What this computes & why the GPU helps

Defibrillation delivers a high-voltage electric field across the myocardium to terminate ventricular fibrillation. Simulating shock efficacy requires solving the bidomain equations driven by extracellular electrode currents, capturing virtual electrode polarization (VEP)—regions of depolarization and hyperpolarization induced at tissue boundaries—and subsequent re-entry termination. The nonlinear ionic response during shock (10 V/cm field, sub-ms timescale) and the fine spatial resolution needed (~0.1 mm) make GPU acceleration mandatory for whole-heart shock simulations.

**The parallel bottleneck:** finding a DFT is inherently a **parameter sweep** — you
must re-simulate the tissue for many shock strengths (and, in the real world,
many shock timings and electrode placements). Each simulation is an independent
reaction-diffusion integration over thousands of time steps. We parallelise
across the sweep: **one GPU thread owns one shock strength and runs that entire
cable simulation** (the "ensemble of trajectories" pattern, PATTERNS.md §1, as in
flagships 9.02 and 13.02). Inside each thread the spatial coupling is a 3-point
diffusion **stencil** with ping-pong buffers.

## The algorithm in brief

Bidomain equations with extracellular stimulus, virtual electrode polarization theory, finite volume/element discretization, operator splitting with Rush-Larsen ionic integration, conjugate gradient linear solver, shock-protocol optimization (monophasic vs. biphasic), defibrillation threshold (DFT) estimation.

In this **reduced-scope teaching version** (see *Limitations*) the pieces are:

- **Monodomain reaction-diffusion** on a 1-D cable: `∂V/∂t = D ∂²V/∂x² + f(V,w) + I_stim`.
- **FitzHugh-Nagumo ionic kinetics** `f(V,w) = V(V−a)(1−V) − w` with a slow recovery gate `w`.
- **Operator-split forward Euler** integration (diffusion stencil → reaction+shock → recovery).
- **Virtual electrode polarization (VEP)** modelled as a signed extracellular
  shock current (depolarising on one side, hyperpolarising on the other).
- **Monophasic vs. biphasic** shock protocol switch.
- **DFT estimation**: the weakest amplitude whose post-shock residual activity
  falls below a success threshold.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation, including how production bidomain codes differ.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/defibrillation-high-voltage-shock-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/defibrillation-high-voltage-shock-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\defibrillation-high-voltage-shock-simulation.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/defib_sweep.txt`, prints the
shock-amplitude sweep and the recovered DFT, shows the GPU-vs-CPU agreement
check, and prints a timing line on stderr.

## Data

- **Sample (committed):** `data/sample/defib_sweep.txt` — a tiny, **synthetic**,
  offline input (cable + FHN parameters and the shock-amplitude ladder) so the
  demo runs with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` point at the real research
  data + solvers; this reduced-scope model needs no download for the demo.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: PhysioNet fibrillation/defibrillation recordings (https://physionet.org); openCARP defibrillation tutorial cases (https://opencarp.org); Cardioid (https://github.com/llnl/cardioid) — bidomain shock examples; patient-specific ICD placement datasets (verify institutional access).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): a table
where the three weakest shocks leave a residual travelling wave (~0.052, *failed*)
and every amplitude from **0.15** up drives the residual to 0 (*DEFIBRILLATED*),
so the recovered **DFT is amplitude 0.150**. The program computes the residual for
every amplitude on both the **GPU** (`src/kernels.cu`) and a **CPU reference**
(`src/reference_cpu.cpp`) and asserts they agree within `1e-6` — they in fact
agree to ~`1e-17` (machine precision) because both call the identical shared
physics in `src/defib.h`.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the sweep, runs CPU + GPU, verifies, reports.
2. [`src/defib.h`](src/defib.h) — **the shared `__host__ __device__` physics core**
   (FHN kinetics, the shock current, one cable time step). Read this closely — it
   is the single source of truth both paths execute.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-per-shock idea.
4. [`src/kernels.cu`](src/kernels.cu) — the sweep kernel and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the loader + trusted serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

openCARP (https://git.opencarp.org/openCARP/openCARP) — bidomain solver with extracellular stimulus for defibrillation studies; MonoAlg3D_C (https://github.com/rsachetto/MonoAlg3D_C) — GPU bidomain-capable extension; Cardioid/LLNL (https://github.com/llnl/cardioid) — cardiac EP + shock; Chaste (https://github.com/Chaste/Chaste) — bidomain with electrode boundary conditions.

- **openCARP** — the reference open-source cardiac EP solver; study its bidomain
  formulation and extracellular-stimulus (defibrillation) setup.
- **MonoAlg3D_C** — a GPU monodomain/bidomain solver; a good look at how the
  per-cell ODE and the linear solve are actually laid out on the device.
- **Cardioid (LLNL)** — HPC cardiac EP + shock at whole-heart scale.
- **Chaste** — bidomain with proper electrode boundary conditions.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Catalog target: cuSPARSE conjugate gradient for bidomain elliptic solve; custom CUDA kernels for per-cell ionic ODE during shock timescale (0.01 ms dt); CUDA Unified Memory for large torso+heart mesh; pattern: dual-grid approach—fine heart mesh on GPU, coarse torso on CPU, coupled via interface boundary.

**What this teaching version actually uses:** the **ensemble-of-trajectories**
pattern — one thread per shock amplitude, each running a full 1-D reaction-diffusion
cable (a 3-point **stencil** with ping-pong buffers, plus per-cell FHN ODE
integration). This deliberately trades the full 3-D bidomain elliptic solve
(which is where cuSPARSE CG would go) for a monodomain 1-D cable so the whole
computation fits in one readable kernel. THEORY.md §"Where this sits in the real
world" explains the elliptic solve and the dual-grid approach the catalog names.

## Exercises

1. **Biphasic protocol.** Regenerate the sample with `python scripts/make_synthetic.py
   --biphasic 1` and re-run. Note the DFT *rises* — then read THEORY.md §numerics
   to understand why a 2-variable FHN model cannot reproduce the clinical
   biphasic advantage, and sketch what ionic detail you would need to add.
2. **Refine the DFT.** Add more amplitudes between 0.10 and 0.15
   (`--amps 0.10 0.11 0.12 0.13 0.14 0.15`) to bracket the threshold more tightly.
   How fine can you resolve it before FHN's all-or-nothing behaviour saturates?
3. **Shock timing (the vulnerable window).** Sweep `--shock-start` instead of
   amplitude at a fixed strength. Cardiac tissue has a *refractory* phase where a
   shock does nothing; can you find it?
4. **One-thread-per-cell.** Re-map the GPU work to a 2-D grid (amplitude × cell)
   with a block-wide barrier per step. Compare against the thread-per-trajectory
   version — which wins as the cable grows to thousands of cells?
5. **2-D tissue.** Extend `defib.h` to a 2-D sheet and initiate a spiral wave
   (the true picture of re-entry), then shock it. This is the step toward the
   real bidomain problem.

## Limitations & honesty

- **Reduced scope (deliberate).** This is a **1-D monodomain FitzHugh-Nagumo
  cable**, not the 3-D **bidomain** system the catalog describes. It captures the
  *concepts* — excitable propagation, virtual electrode polarization, an
  all-or-nothing defibrillation threshold, monophasic vs. biphasic — but omits
  the separate intra/extracellular potentials, the elliptic (cuSPARSE CG) solve,
  a realistic ionic model, and 3-D geometry. THEORY.md spells out the full model.
- **Everything is synthetic and dimensionless.** The FHN parameters are chosen to
  make a clean teaching curve, not to match any tissue. The "residual activity"
  metric is a proxy, not a clinical endpoint.
- **The biphasic result is a known model artifact.** Real biphasic shocks
  defibrillate at *lower* energy than monophasic; this 2-variable model shows the
  opposite because it lacks the sodium-channel recharge dynamics responsible for
  the clinical benefit. We keep it honest and explain it rather than hide it.
- **Timing is a teaching artifact.** On this tiny sweep the GPU is launch/copy
  bound and can look slower than the CPU; the advantage grows with sweep size and
  cable length (CLAUDE.md §12).
- **Not for clinical use.** Nothing here may inform diagnosis, device programming,
  or treatment.
