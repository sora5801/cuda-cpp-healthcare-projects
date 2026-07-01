# 6.3 — Hemodynamics / Blood-Flow CFD

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.3`
>
> _Educational only — not for clinical use (see CLAUDE.md §8). This is a
> **reduced-scope teaching version** (CLAUDE.md §13) — see "Limitations & honesty"._

## Summary

This project solves the **2-D incompressible Navier-Stokes equations** for blood
flowing through a straight channel and computes the **wall shear stress (WSS)** —
the tangential force blood exerts on the vessel wall, a well-established
atherosclerosis risk factor. It implements **Chorin's fractional-step (projection)
method**: a predictor step, a Jacobi-iterated pressure Poisson solve, and a
projection step — every one a nearest-neighbour **stencil** that maps cleanly onto
the GPU (one thread per grid cell). The solver includes the **Carreau-Yasuda**
non-Newtonian viscosity model of real blood, and is verified two ways: the GPU
field matches an independent CPU reference to machine precision, and the resulting
velocity profile converges to the exact analytic **Poiseuille parabola**.

## What this computes & why the GPU helps

The full research problem (from the catalog): *solve the incompressible
Navier-Stokes equations on patient-specific vascular geometries reconstructed from
CT/MRI angiography, with non-Newtonian rheology and compliant walls (FSI), to map
wall shear stress and the oscillatory shear index across the cardiac cycle.*

This teaching version keeps the **same governing equations and the same clinical
output (WSS)** but on a rigid straight channel (details in "Limitations").

**The parallel bottleneck:** the runtime is dominated by the **pressure Poisson
solve** — `p_iters` Jacobi sweeps *every* time step, each sweep touching all
nx·ny cells. Because a Jacobi sweep (and the predictor and corrector) are pure
stencils where every cell updates independently from its four neighbours, they map
to **one GPU thread per cell** with no data races. That data parallelism over the
mesh is exactly what GPUs accelerate, and what production hemodynamics codes scale
to millions of 3-D cells.

## The algorithm in brief

- **Predictor** — provisional velocity u* from advection (first-order upwind) +
  diffusion (5-point Laplacian) + body force; viscosity from **Carreau-Yasuda**.
- **Pressure Poisson** — solve ∇²p = (ρ/Δt)·∇·u* by **Jacobi iteration**
  (double-buffered).
- **Corrector / projection** — u = u* − (Δt/ρ)∇p → divergence-free velocity.
- **Wall shear stress** — τ_w = ρν·(du/dy) at the wall, the clinical output.
- **Verification** — CPU==GPU to 1e-9, and u_max vs analytic Poiseuille.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/hemodynamics-blood-flow-cfd.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/hemodynamics-blood-flow-cfd.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\hemodynamics-blood-flow-cfd.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/channel_params.txt`, prints the
velocity profile + WSS, shows the GPU-vs-CPU agreement check, and prints a timing
line. (It runs 40000 steps, so expect ~20 s — the Jacobi solver is deliberately
simple, not fast; see the exercises.)

## Data

- **Sample (committed):** `data/sample/channel_params.txt` — a tiny, **synthetic**
  parameter set (not patient data) so the demo runs offline with zero downloads.
- **Regenerate / vary:** `python scripts/make_synthetic.py [--steps N --nu-inf V]`.
- **Full datasets:** `scripts/download_data.ps1` / `.sh` print pointers to the
  credential-gated real datasets (they never bypass logins).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: Vascular Model Repository (patient geometries,
<http://www.vascularmodel.com>); PhysioNet MIMIC-III waveforms
(<https://physionet.org/content/mimiciii/1.4/>); Zenodo Cardiac Mechanics
Emulation (<https://zenodo.org/records/7075055>); UK Biobank aortic 4D-flow MRI
(<https://www.ukbiobank.ac.uk>).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): a
symmetric parabolic velocity profile, `centerline u_max ≈ 0.0305` (analytic
Poiseuille = 0.0320, ~4.8% off because 40000 steps is near but not fully at steady
state), the wall shear stress, and `RESULT: PASS`. The program computes the result
on both the **GPU** (`src/kernels.cu`) and a **CPU reference**
(`src/reference_cpu.cpp`) and asserts they agree within **1e-9** — that agreement
is the correctness guarantee (here they are bit-identical).

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads params, runs CPU + GPU, verifies, reports.
2. [`src/nse_channel.h`](src/nse_channel.h) — **the shared per-cell physics**
   (predictor/divergence/pressure/corrector as `__host__ __device__` functions,
   plus Carreau-Yasuda viscosity and WSS). The heart of the project — start here
   for the math.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the stencil idea.
4. [`src/kernels.cu`](src/kernels.cu) — the four kernels + the ping-pong time loop.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **SimVascular/svFSI** (<https://github.com/SimVascular/svFSI>) — open-source
  image-to-simulation pipeline; learn how a segmented geometry becomes an FSI solve.
- **OpenFOAM** (<https://github.com/OpenFOAM/OpenFOAM-dev>) — general finite-volume
  CFD; read `icoFoam`/`pimpleFoam` for the pressure-velocity coupling this project's
  projection step approximates.
- **Chaste** (<https://github.com/Chaste/Chaste>) — includes a vascular-network
  flow module.
- **HemeLB** (<https://github.com/hemelb-codes/hemelb>) — sparse lattice-Boltzmann
  hemodynamics; the alternative to an explicit pressure solve (see flagship 6.04).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Stencil + ping-pong buffers** (PATTERNS.md §1, exemplar 6.04), with the shared
`__host__ __device__` core idiom (PATTERNS.md §2) for exact CPU/GPU parity, and a
**Jacobi projection + double buffer** for the pressure solve (as in flagship 10.02).
One thread per grid cell; four kernels per time step; no atomics; fully
deterministic stdout. Production codes swap the Jacobi solve for **AmgX** (GPU
algebraic multigrid) and **cuSPARSE** SpMV on an unstructured mesh — described but
not used here (no black boxes: THEORY.md §4 explains what those would compute).

## Exercises

1. **Converge fully.** Rerun with `--steps 80000`; watch `u_max` approach the
   analytic 0.0320 (error < 0.3%). Plot the profile against the exact parabola.
2. **Turn on non-Newtonian blood.** Regenerate with `--nu-inf 0.03` (shear
   thinning). How does the velocity profile flatten near the centre and steepen at
   the wall? How does WSS change? (The analytic Poiseuille check no longer applies —
   why?)
3. **Shared-memory tiling.** The predictor re-reads each neighbour from global
   memory. Stage each 16×16 tile (plus a halo) into `__shared__` memory and measure
   the bandwidth saving (THEORY.md §4).
4. **A faster pressure solver.** Replace Jacobi with red-black Gauss-Seidel (2×
   fewer sweeps) or a geometric multigrid V-cycle, and compare iterations-to-
   convergence.
5. **Add a stenosis.** Mark an interior band of cells as solid (bounce-back) to
   narrow the channel; observe the WSS spike downstream — the atherosclerosis
   mechanism this project exists to illustrate.

## Limitations & honesty

- **Reduced scope (CLAUDE.md §13).** The catalog project is 3-D, patient-specific,
  non-Newtonian, fluid-structure-interaction CFD on unstructured meshes with a
  multigrid pressure solve — a research effort. This ships a **2-D structured-grid
  rigid-channel** solver that implements the *same governing equations* and the
  *same clinical output (WSS)* in a form a learner can read and fully verify.
- **What is omitted:** compliant walls / FSI / ALE mesh motion; pulsatile cardiac-
  cycle inlets and the oscillatory shear index (OSI); unstructured geometry; a
  scalable multigrid pressure solver. THEORY.md §7 details each.
- **Synthetic data.** The sample is a synthetic channel parameter set, labelled
  synthetic everywhere. No patient data is used or redistributed.
- **Not a benchmark.** The GPU is *slower* than the CPU here — 40000×43 tiny kernel
  launches on a 32×17 grid are launch-bound. The GPU's advantage appears at 3-D,
  patient-scale mesh sizes; the timing is a teaching artifact only (CLAUDE.md §12).
- **Not clinical.** No output of this project is valid for diagnosis or treatment.
