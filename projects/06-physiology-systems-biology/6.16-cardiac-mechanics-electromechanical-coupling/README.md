# 6.16 — Cardiac Mechanics & Electromechanical Coupling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.16`
>
> _Educational only — **not for clinical use** (see CLAUDE.md §8). All numbers are synthetic and illustrative._

## Summary

This project simulates **cardiac electromechanical coupling** — the chain by which
an electrical heartbeat becomes a mechanical squeeze that ejects blood. For each
simulated ventricle we integrate a small coupled ODE: electrical activation drives
an intracellular **calcium transient**; calcium recruits **cross-bridges** (the
force-generating motors); the recruited state raises the chamber's **elastance**
(stiffness); the resulting pressure opens the aortic valve and ejects blood into a
lumped **Windkessel** arterial load. The output is a **pressure–volume (PV) loop**
per heart, from which we read the clinically meaningful **ejection fraction (EF)**,
stroke volume, and peak pressure. The GPU angle is an *ensemble*: we solve this
same ODE for a whole sweep of hearts (varying contractility and afterload) in
parallel, **one GPU thread per heart**.

This is a deliberately **reduced-scope teaching version** of the full research
problem (a 3-D nonlinear finite-element solve); see *Limitations & honesty*.

## What this computes & why the GPU helps

The full research problem (paraphrasing the catalog): *"couples a stiff ODE (ionic
+ cross-bridge) at each integration point to a nonlinear FEM problem (hyperelastic
myocardium with active stress/strain). GPU accelerates the per-Gauss-point ODE
batch and the global Newton-Raphson iterations. Ventricular PV loops, ejection
fraction, and wall stress are clinical outputs."*

**The parallel bottleneck we teach:** the *per-integration-point ODE batch*. In a
real solver, a stiff cell + cross-bridge ODE must be integrated at **every Gauss
point of a large mesh, every timestep** — millions of independent ODE solves. That
batch is embarrassingly parallel and is exactly where a GPU wins. We keep that
structure but make each "integration point" a whole 0-D virtual heart, so the demo
runs on any machine: **thread `i` integrates heart `i`'s full multi-beat PV loop in
registers**, with no inter-thread communication.

## The algorithm in brief

- **Calcium transient** — a difference-of-exponentials pulse (fast rise, slower
  decay) standing in for L-type Ca influx + SERCA re-uptake.
- **Cross-bridge activation** — a first-order state `xb` relaxing toward a **Hill
  curve** of calcium (cooperative troponin binding); this lag is the
  electromechanical delay.
- **Time-varying elastance** (Suga–Sagawa) — chamber pressure `P = E(t)·(V − V0)`
  with `E(t) = Emin + Tref·xb`: soft in diastole (fills), stiff in systole (ejects).
  `Tref` is the **contractility** knob.
- **Valves** — smooth diode-like resistances (aortic ejection, mitral filling).
- **Windkessel afterload** — a 2-element `R_sys`–`C` arterial load; `R_sys` is the
  **afterload** knob.
- **RK4** integrator, shared verbatim between CPU and GPU.
- **Ensemble sweep** — `nT` contractility × `nR` afterload = `nT·nR` independent
  solves.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/cardiac-mechanics-electromechanical-coupling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/cardiac-mechanics-electromechanical-coupling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\cardiac-mechanics-electromechanical-coupling.sln /p:Configuration=Release /p:Platform=x64
```

Only the CUDA runtime is linked (`cudart_static.lib`); no extra CUDA library is
needed for this reduced-scope version.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/heart_ensemble.txt`, prints the
per-heart PV-loop summary, shows the GPU-vs-CPU agreement check, and prints a
timing line.

## Data

- **Sample (committed):** `data/sample/heart_ensemble.txt` — a tiny, **synthetic**
  model configuration (baseline physiology + a 6×6 sweep) so the demo runs offline.
- **Full dataset:** none required (the model is self-contained);
  `scripts/download_data.ps1` / `.sh` print pointers to real cardiac datasets and
  never bypass credentials.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: UK Biobank CMR + strain imaging (https://www.ukbiobank.ac.uk); Zenodo cardiac mechanics emulation dataset (https://zenodo.org/records/7075055); ACDC segmentation challenge (https://www.creatis.insa-lyon.fr/Challenge/acdc/); MICCAI STACOM cardiac mechanics challenge data (verify URL on grand-challenge.org).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): a table
of sample hearts with EDV/ESV/SV/EF/peak-pressure, the best/worst-EF hearts, the
mean EF, and a final `RESULT: PASS` line. The program computes every heart's PV
loop on both the **GPU** (`src/kernels.cu`) and a **CPU reference**
(`src/reference_cpu.cpp`) and asserts they agree within a documented **physical
tolerance** of `0.1` (mL / mmHg / percentage-point). That tolerance is honest, not
lax: this is an ~80,000-step-per-heart solver, and GPU fused-multiply-add vs. the
host compiler diverge by ~`1e-3`–`1e-2` in the recorded scalars (worst for the
peak-pressure *extremum*) — see THEORY.md "Numerical considerations".

The headline teaching result: **EF rises with contractility** (~36% for a weak
ventricle → ~65% for a strong one) and **peak pressure rises with afterload**.

## Code tour

Read in this order:

1. [`src/cardiac.h`](src/cardiac.h) — **start here**: the shared `__host__
   __device__` physics (calcium, cross-bridges, elastance, valves, Windkessel) and
   the RK4 step + `integrate_cycle()`.
2. [`src/main.cu`](src/main.cu) — loads the sweep, runs CPU + GPU, verifies, reports.
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`.cpp`](src/reference_cpu.cpp) —
   the ensemble config, the (index → HeartParams) mapping, the loader, and the
   trusted serial baseline.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-per-heart idea.
5. [`src/kernels.cu`](src/kernels.cu) — the kernel and host wrapper.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **FEBio** (<https://github.com/febiosoftware/FEBio>) — production nonlinear FEM
  for cardiac/soft-tissue mechanics; study how it assembles the tangent stiffness.
- **simcardems** (<https://github.com/ComputationalPhysiology/simcardems>) —
  FEniCS-based EP + mechanics coupling; the canonical modern coupling reference.
- **OpenCMISS/cm** (<https://github.com/OpenCMISS/cm>) — multi-physics FEM framework.
- **Chaste** (<https://github.com/Chaste/Chaste>) — has a clear cardiac
  electromechanics tutorial; good for the equations behind the full model.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Batch ODE, one integration point per thread** (the ensemble-RK4 pattern of
PATTERNS.md §1, shared with flagships `9.02` SEIR and `13.02` PBPK). Each thread
runs the full multi-beat RK4 loop for one heart in registers/local memory; the
per-element physics lives in one `__host__ __device__` header
([`src/cardiac.h`](src/cardiac.h)) so CPU and GPU compute identical math
(PATTERNS.md §2). The full solver's *other* GPU pieces (cuSOLVER for the Newton
linear solve, cuSPARSE SpMV for stiffness assembly, a two-level element/Gauss-point
grid) are described in THEORY.md "Where this sits in the real world".

## Exercises

1. **Frank–Starling.** Raise the preload `P_ven` (or `V_fill`) and confirm stroke
   volume increases — the heart pumps out more when filled more. Plot SV vs. preload.
2. **Afterload mismatch.** Widen the `R_sys` sweep and find where EF collapses
   because the ventricle can no longer open the aortic valve. What is the critical
   afterload for a weak (`Tref` low) heart?
3. **Emit the full PV loop.** Add an option to write the last beat's `(V, P)`
   samples for one heart to a file and plot the loop; measure its area (= stroke work).
4. **Tolerance study.** Reduce `steps_per_beat` (larger `dt`) and watch both the
   accuracy and the CPU–GPU divergence change; at what `dt` does RK4 become unstable?
5. **Scale the ensemble.** Regenerate with `--nT 64 --nR 64` (4096 hearts) and watch
   the GPU time grow far more slowly than the CPU time — the batch-ODE speed-up.

## Limitations & honesty

- **Reduced scope (declared).** This is a **0-D lumped** model, not the 3-D
  nonlinear FEM the catalog describes. There is no mesh, no Gauss points, no
  Holzapfel–Ogden tensor, no Newton–Raphson equilibrium solve, no
  monodomain diffusion. Those are described in THEORY.md but not implemented — the
  goal is to teach the *coupling chain* and the *batch-ODE GPU pattern* on hardware
  everyone has (CLAUDE.md §13).
- **Phenomenological, not first-principles.** The calcium transient and elastance
  are curve-fits standing in for full ionic and cross-bridge models
  (Rice–Wang–Bers). Parameters are **synthetic**, in a physiological ballpark, and
  **not fitted to any patient**.
- **The GPU can be slower here.** On the tiny 36-heart sample the long, branch-heavy
  per-thread integrations are launch/latency-bound; the GPU's advantage only appears
  as the ensemble grows (PATTERNS.md §7). Timing is a teaching artifact, never a
  benchmark claim.
- **Not clinical.** EF, PV loops, and the wall-stress proxy here are software
  demonstrations, not diagnoses or forecasts.
