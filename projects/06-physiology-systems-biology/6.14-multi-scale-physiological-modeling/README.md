# 6.14 — Multi-Scale Physiological Modeling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.14`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

This project couples **two physical scales** — a cell-level ODE and a
tissue-level PDE — and solves them together on the GPU, the essence of the
Virtual Physiological Human (VPH) idea. Concretely it simulates a 1-D strand of
cardiac tissue (a "cable"): at every node lives a tiny **FitzHugh-Nagumo** cell
model (a didactic stand-in for ion-channel kinetics), and the nodes are coupled
by **electrical diffusion**. A stimulus at the left end launches an **action
potential** that *propagates* to the far end as a traveling wave — the same
phenomenon behind every heartbeat. The demo recovers the wave's **activation map**
and its **conduction velocity**, and checks the GPU result against a serial CPU
reference.

## What this computes & why the GPU helps

The multi-scale challenge (from the catalog deep dive) is that a **fine-scale
cell ODE must be solved at every node of a coarse mesh, simultaneously** — in a
real heart model, *millions* of coupled ODEs per time step. We keep that
structure but shrink it to a laptop-sized 1-D cable. Each global time step does
**operator splitting**: (1) a **reaction** sub-step advances every node's cell
ODE independently (embarrassingly parallel — the bottleneck the GPU flattens),
then (2) a **diffusion** sub-step couples neighbours with a 3-point stencil. The
GPU maps this as **one thread per node** (the catalog's "grid over mesh elements,
threads over the per-element ODE RHS"), with **ping-pong buffers** for the
stencil so the parallel update matches the serial one exactly. On a short cable
this is *launch-bound* (the GPU can be slower — see the timing note); the GPU's
edge grows with mesh size toward organ scale.

## The algorithm in brief

- **FitzHugh-Nagumo cell ODE** per node — fast excitation `v` + slow recovery `w`.
- **RK4** integration of the reaction term (the fine, sub-grid scale).
- **Monodomain cable diffusion** — explicit forward-Euler 3-point Laplacian with
  zero-flux (Neumann) boundaries (the coarse, tissue scale).
- **Operator splitting** (Godunov) to couple the two scales each global step.
- **Activation mapping** — first-crossing times → conduction velocity.

Depth, equations, and the GPU-mapping rationale live in [THEORY.md](THEORY.md).

## Build

Prerequisites: **Visual Studio 2026** (v145 toolset, "Desktop development with
C++") and **CUDA Toolkit 13.3** — see [`docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open `build/multi-scale-physiological-modeling.sln` in Visual Studio 2026.
2. Select the **`Release`** configuration and **`x64`** platform.
3. **Build → Build Solution** (`Ctrl+Shift+B`).

The executable lands in `build/x64/Release/multi-scale-physiological-modeling.exe`.
A command-line build (used by the demo):

```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe" `
  build\multi-scale-physiological-modeling.sln /p:Configuration=Release /p:Platform=x64 /m
```

A cross-platform **CMake** build is provided too (optional, for Linux/CI):
`cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release && cmake --build build/cmake`.

## Run the demo

```powershell
powershell -ExecutionPolicy Bypass -File demo/run_demo.ps1
```

(Linux/CMake: `bash demo/run_demo.sh`.) It builds if needed, runs on
`data/sample/cable.txt`, prints the activation map + conduction velocity, and
diffs stdout against [`demo/expected_output.txt`](demo/expected_output.txt).

## Data

The committed sample is a single line of **synthetic** configuration numbers
(`data/sample/cable.txt`): the cable geometry, time-stepping, left-end stimulus,
and FHN + diffusion parameters. It is not measured data — it is the *setup* of a
self-contained toy simulation, tuned so the stimulus launches a clean traveling
wave. Regenerate/resize with `python scripts/make_synthetic.py [--n … --steps …]`.
This project is **simulation-only**, so no download is needed to run it;
`scripts/download_data.*` point to the real VPH model repositories (Physiome
Model Repository, BioModels, OpenCMISS examples, UK Biobank). Provenance, field
meanings, and licensing are in [`data/README.md`](data/README.md).

## Expected output

```
6.14 -- Multi-Scale Physiological Modeling
1-D monodomain cable: 128 nodes, dx=0.500, dt=0.0200, 5000 steps (T=100.00)
FHN cell: a=0.130 eps=0.005 b=0.500 | tissue D=2.000 | stim 5 left nodes
activation map (node : x : t_activation):
  n0       0.000    0.0000
  n25     12.500   17.6200
  n50     25.000   36.5600
  n76     38.000   56.2600
  n101    50.500   75.2200
  n127    63.500   90.2000
nodes activated: 128 / 128
conduction velocity: 0.6790 (space/time)
RESULT: PASS (GPU field matches CPU within tol=1.0e-06)
```

The activation times **increase with position** — that monotone map *is* the
propagating action potential. `RESULT: PASS` means the GPU's final `(v, w)`
fields and activation map match the CPU reference within `1e-6` (they actually
agree to ~`1e-16`; the split arithmetic is identical on both sides). The GPU/CPU
timings are on **stderr** (they vary and are not diffed).

## Code tour

Suggested reading order:

1. **`src/multiscale.h`** — the shared `__host__ __device__` core: the FHN
   reaction terms, the RK4 step, the diffusion stencil, and the config struct.
   *This is where the physics lives; both CPU and GPU call it.*
2. **`src/main.cu`** — the 5-step driver: load → CPU reference → GPU → verify →
   report (deterministic result to stdout, timing to stderr).
3. **`src/kernels.cu`** — the three per-step GPU kernels (react / diffuse /
   record) and the host wrapper with **ping-pong buffers**.
4. **`src/reference_cpu.cpp`** — the serial split-step baseline + the loader +
   the conduction-velocity summary (shared with the GPU path).

## Prior art & further reading

From the catalog "Starter repos / tools" — study these; do not copy wholesale:

- **[OpenCMISS](https://github.com/OpenCMISS/cm)** — a multi-physics, multi-scale
  FEM framework. *Learn how real inter-scale coupling is structured.*
- **[SUNDIALS batch-CVODE (GPU)](https://github.com/LLNL/sundials)** — the
  production way to batch-solve the sub-grid cell ODEs we hand-roll with RK4.
- **[simcardems](https://github.com/ComputationalPhysiology/simcardems)** —
  cardiac electromechanics multi-scale coupling. *The next scale up (mechanics).*
- **[Chaste](https://github.com/Chaste/Chaste)** — multi-scale cardiac / lung /
  tumor modeling. *A broad, well-tested reference implementation.*

Background: FitzHugh (1961) & Nagumo et al. (1962) for the cell model; Keener &
Sneyd, *Mathematical Physiology* for reaction-diffusion in excitable tissue.

## Exercises

1. **Break stability.** Raise `D` (or `dt`) in `make_synthetic.py` until
   `D·dt/dx² > 0.5` and watch the explicit diffusion step blow up. Then fix it
   with an *implicit* diffusion solve (Thomas algorithm on the tridiagonal system).
2. **Strang splitting.** Replace Godunov splitting (react-then-diffuse) with
   second-order Strang splitting (half-diffuse → react → half-diffuse) and measure
   the accuracy improvement at large `dt`.
3. **2-D tissue.** Extend the cable to a 2-D sheet (5-point Laplacian); launch a
   wave from a corner and visualize the curved wavefront. This is where the GPU
   starts to clearly beat the CPU.
4. **Re-entry.** Stimulate the left end twice with the right timing to create a
   unidirectional block, and observe a re-entrant (rotating) wave — the
   mechanism behind cardiac arrhythmias.
5. **A real cell model.** Swap FHN for a Beeler-Reuter or ten-Tusscher ionic
   model at each node (more state variables, stiffer) and note why an implicit /
   adaptive integrator (CVODE) becomes attractive.

## Limitations & honesty

This is a **reduced-scope teaching version** of a research-grade problem
(CLAUDE.md §13). It is **1-D**, uses the **FitzHugh-Nagumo caricature** rather than
a biophysical ionic model, an **explicit** (stability-limited) diffusion step
rather than an implicit/FEM solve, and **Godunov** (first-order) operator
splitting. There is no mechanics, no circulation, and no fiber anisotropy. All
data is **synthetic** and the numbers are **illustrative only** — they have no
clinical meaning and must not inform any medical decision. The GPU timing is a
*teaching artifact*, not a benchmark claim: on this small cable the many tiny
per-step kernel launches are launch-bound and can be slower than the CPU; the
GPU's advantage appears at organ-scale mesh sizes.
