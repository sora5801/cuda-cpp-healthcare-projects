# 14.02 — Spatial / Whole-Cell Reaction-Diffusion (teaching stencil)

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Emerging%2C%20Theoretical%20%26%20Grand--Challenge%20Frontiers-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 14: Emerging, Theoretical & Grand-Challenge Frontiers · Catalog ID `14.02`
>
> _Educational only (see CLAUDE.md §8). **Reduced-scope teaching version** of a frontier problem — see Limitations._

## Summary

Simulate a **reaction-diffusion** system — the Gray-Scott model — on a 2-D grid:
two chemicals `U` and `V` diffuse and react, and from a tiny seed they
self-organize into **Turing patterns** (spots, stripes, labyrinths) that resemble
biological patterning (skin, coral, cell signaling). Each grid cell updates from
its four neighbours — a pure **stencil**, one thread per cell. This is the
continuum (grid) teaching version of the catalog's molecular-resolution
reaction-diffusion frontier project.

## What this computes & why the GPU helps

Spatial reaction-diffusion underlies how cells organize signaling gradients,
receptor clusters, and patterning. The continuum form is a PDE solved on a grid;
the per-cell stencil update is data-parallel — one thread per cell, double-buffered
across timesteps (like the lattice-Boltzmann project `6.04`). The full molecular
version tracks every particle and needs multi-GPU systems (see THEORY).

**The parallelized work** is the per-cell reaction-diffusion stencil, every
timestep, over the whole grid.

## The algorithm in brief

- **Gray-Scott PDE:** `dU/dt = Du·∇²U − UV² + F(1−U)`, `dV/dt = Dv·∇²V + UV² − (F+k)V`.
- **Discretize:** explicit Euler in time, 5-point Laplacian in space (periodic).
- **Pattern:** the feed `F` and kill `k` rates select spots / stripes / mazes.

See [THEORY.md](THEORY.md) for Turing instability, stability limits, and the particle-based frontier.

## Build

Requires **Visual Studio 2026** (v145) + **CUDA 13.3** ([docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/spatial-whole-cell-reaction-diffusion-at-molecular-resolution.sln`.
2. **`Release|x64`** → **Build** → the `.exe` under `build/x64/Release/`.

CLI: `msbuild build\spatial-whole-cell-reaction-diffusion-at-molecular-resolution.sln /p:Configuration=Release /p:Platform=x64`

## Run the demo

```powershell
./demo/run_demo.ps1
```

Runs the reaction-diffusion on CPU + GPU and verifies the final fields match.

## Data

- **Sample (committed):** `data/sample/grayscott_params.txt` — grid + Gray-Scott rates.
- **Molecular-resolution RD:** ReaDDy / Smoldyn / MCell / STEPS — see
  `scripts/download_data.ps1` and [data/README.md](data/README.md).
- Bigger grid: `python scripts/make_synthetic.py --nx 256 --ny 256 --steps 12000`.

## Expected output

`demo/expected_output.txt` holds the deterministic pattern metrics. The GPU
(`src/kernels.cu`) and CPU (`src/reference_cpu.cpp`) share the per-cell update
(`src/rd.h`) in double precision, so the final fields agree to ~`1e-7`. From the
seed, V forms a Turing labyrinth covering ~half the grid.

## Code tour

1. [`src/main.cu`](src/main.cu) — load, seed, CPU + GPU simulate, verify, print metrics.
2. [`src/rd.h`](src/rd.h) — **the Laplacian + Gray-Scott per-cell update** (host + device).
3. [`src/kernels.cuh`](src/kernels.cuh) — the stencil interface (1 thread/cell, ping-pong).
4. [`src/kernels.cu`](src/kernels.cu) — the step kernel + host time loop.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — field init + the serial reference.

## Prior art & further reading

- **ReaDDy** (<https://github.com/readdy/readdy>) — GPU particle-based reaction-diffusion.
- **Smoldyn** (<https://github.com/ssandrews/Smoldyn>) — off-lattice PBRD.
- **MCell** (<https://mcell.org/>) — Monte-Carlo 3-D RD for neurons.
- Gray & Scott (1984); Pearson (1993), *Complex Patterns in a Simple System* — the model + its phase diagram.

Study these for the frontier approach; reimplement the pattern didactically (CLAUDE.md §2).

## CUDA pattern used here

Nearest-neighbour **stencil** (one thread per cell, 2-D grid) · **ping-pong** double
buffering across timesteps · periodic boundaries · shared `__host__ __device__`
per-cell update for exact CPU/GPU parity. (The same stencil shape as `6.04`.)

## Exercises

1. **Explore the phase diagram.** Sweep `(F, k)` — spots, stripes, mazes,
   self-replicating "mitosis", chaos. Plot which (F,k) gives which.
2. **Shared-memory tiling.** Cache a tile + halo of U,V in shared memory to cut
   global-memory traffic (the classic stencil optimization).
3. **Visualize.** Write the final V field to a PGM/PNG image and watch the pattern.
4. **3-D.** Extend to a 3-D grid (7-point Laplacian) — closer to a cell volume.
5. **Toward particles.** Read about ReaDDy/Smoldyn and contrast the continuum PDE
   with tracking individual molecules (the catalog's frontier version).

## Limitations & honesty

- **Reduced-scope teaching version** (CLAUDE.md §11): a continuum 2-D Gray-Scott
  PDE, **not** the catalog's particle-based molecular-resolution RD (that tracks
  every molecule with cell-list neighbour search — a 🔴 multi-GPU frontier problem,
  described in THEORY).
- Explicit Euler (conditionally stable: `dt < 1/(4·max(Du,Dv))`); periodic
  boundaries; Gray-Scott is an abstract model, not real biochemistry.
- One kernel launch per step ⇒ launch-bound on small grids; the win grows with size.
