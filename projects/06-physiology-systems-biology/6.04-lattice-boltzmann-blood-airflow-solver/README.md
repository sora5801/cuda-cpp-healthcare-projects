# 6.04 — Lattice-Boltzmann Blood/Airflow Solver

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.04`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Simulate fluid flow (blood, airflow) with the **Lattice-Boltzmann Method (LBM)**:
instead of solving the Navier-Stokes PDEs directly, track 9 "populations" of
fictitious particles at each grid node (the **D2Q9** stencil) and repeatedly
**collide** (relax toward equilibrium) and **stream** (move to neighbours). Each
node updates from its nearest neighbours only — a pure **stencil**, the fifth
distinct GPU pattern in the flagships and the workhorse of GPU computational
fluid dynamics. The demo develops textbook **Poiseuille (parabolic) channel flow**.

## What this computes & why the GPU helps

LBM replaces continuum Navier-Stokes with a mesoscale kinetic model on a regular
lattice. It is ideal for GPUs because a node updates using **only nearest-neighbour
communication** — no global solves — so the collide+stream step is embarrassingly
local. Blood flow in vascular trees, red-blood-cell rheology, and pulmonary
airflow all use it; HemeLB reaches tens of billions of lattice-site updates per
second, and GPU versions push further.

**The parallel bottleneck** is the per-node collide+stream stencil over the whole
lattice, every timestep; we give each node a thread on a 2-D grid and iterate.

## The algorithm in brief

Per node, per step: **stream** (pull each population from its upstream neighbour;
bounce-back at walls), compute density/velocity moments, then **collide** (BGK
relaxation toward the local Maxwellian equilibrium), with a body force driving the
flow.

See [THEORY.md](THEORY.md) for the kinetic theory, the BGK operator, and the 3-D extension.

## Build

Requires **Visual Studio 2026** (v145) + **CUDA 13.3** ([docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/lattice-boltzmann-blood-airflow-solver.sln`.
2. **`Release|x64`** → **Build** → `build/x64/Release/lattice-boltzmann-blood-airflow-solver.exe`.

CLI: `msbuild build\lattice-boltzmann-blood-airflow-solver.sln /p:Configuration=Release /p:Platform=x64`

## Run the demo

```powershell
./demo/run_demo.ps1
```

Runs the channel flow on CPU + GPU and verifies the velocity fields match.

## Data

- **Sample (committed):** `data/sample/channel_params.txt` — the channel + run parameters.
- **Realistic geometry:** 3-D segmented vessels/airways with HemeLB / PALABOS —
  see `scripts/download_data.ps1` and [data/README.md](data/README.md).
- Bigger 2-D grid: `python scripts/make_synthetic.py --nx 128 --ny 64 --steps 20000`.

## Expected output

`demo/expected_output.txt` holds the deterministic across-channel velocity
profile — a parabola peaking at the centerline. The GPU (`src/kernels.cu`) and CPU
(`src/reference_cpu.cpp`) share the per-node update (`src/lbm_d2q9.h`), so their
velocity fields agree to ~machine precision (`max diff ≈ 2e-16`).

## Code tour

1. [`src/main.cu`](src/main.cu) — load, run CPU + GPU LBM, verify, print the velocity profile.
2. [`src/lbm_d2q9.h`](src/lbm_d2q9.h) — **the shared per-node collide+stream update** (host + device).
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU stencil interface (1 thread/node, ping-pong loop).
4. [`src/kernels.cu`](src/kernels.cu) — the step kernel + host time loop.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the serial reference + velocity moments.

## Prior art & further reading

- **HemeLB** (<https://github.com/hemelb-codes/hemelb>) — sparse-geometry vascular LBM, MPI+GPU.
- **PALABOS** (<https://gitlab.com/unigespc/palabos>) — full-featured C++ LBM framework.
- **USERMESO-2.0** (<https://github.com/AnselGitAccount/USERMESO-2.0>) — GPU red-blood-cell hemodynamics.

Study these for production LBM; this project reimplements the core pattern didactically (CLAUDE.md §2).

## CUDA pattern used here

Nearest-neighbour **stencil** (one thread per lattice node, 2-D grid) ·
**ping-pong** double buffering across timesteps · pull-streaming + halfway
bounce-back walls · BGK collision · shared `__host__ __device__` per-node update
for exact CPU/GPU parity.

## Exercises

1. **Shared-memory tiling.** Cache a tile of the lattice (plus a halo) in shared
   memory so streaming reads coalesce — the standard LBM GPU optimization.
2. **Obstacle.** Add a solid cylinder (bounce-back nodes) and watch a wake / von
   Kármán vortex street form. Plot the velocity field.
3. **MRT collision.** Replace single-relaxation BGK with multi-relaxation-time
   (MRT) for better stability at low viscosity.
4. **3-D D3Q19.** Extend the stencil to 3-D (19 velocities) — the geometry real
   blood/airflow solvers use.
5. **Throughput.** Grow the grid (`--nx 512 --ny 512`) and measure MLUPS
   (mega-lattice-updates/s) on CPU vs GPU; where does the GPU pull ahead?

## Limitations & honesty

- **2-D D2Q9, single-relaxation BGK**, straight channel — real solvers are 3-D
  (D3Q19/D3Q27), often MRT, in complex vessel geometries.
- **One kernel launch per timestep** (no shared-memory tiling), so on a small grid
  the GPU is launch-bound; the win grows with grid size (Exercise 5).
- The body force uses a simple equilibrium-velocity shift; production uses Guo
  forcing. Flow is incompressible-limit only (low Mach).
