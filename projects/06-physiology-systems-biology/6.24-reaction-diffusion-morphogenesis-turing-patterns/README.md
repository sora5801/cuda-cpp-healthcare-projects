# 6.24 — Reaction-Diffusion Morphogenesis (Turing Patterns)

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟢 Beginner · Established** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.24`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

In 1952 Alan Turing showed that two chemicals which **diffuse at different speeds**
and **react nonlinearly** can spontaneously turn a featureless, uniform state into
a stationary spatial **pattern** — spots, stripes, or labyrinths. This is the
leading mathematical account of how a nearly-uniform embryo lays down periodic
structure: leopard spots, zebrafish stripes, the regular spacing of hair
follicles, the pre-pattern of digits. This project simulates the canonical
activator–inhibitor system (**Gierer–Meinhardt**) on a 2-D grid and watches a
pattern emerge from a near-uniform seed. It runs the same physics on a **CPU
reference** and a **GPU stencil kernel**, checks that they agree, and — as an
independent scientific test — computes the **Turing dispersion relation** to
predict the pattern's wavelength from first principles.

## What this computes & why the GPU helps

Turing's 1952 reaction-diffusion system produces spatial patterns (spots, stripes, labyrinthine) from uniform initial conditions through short-range activation and long-range inhibition. Biological applications include skin pigmentation, hair follicle spacing, digit patterning, and cortical folding. GPU simulation on large 2D/3D domains enables the parameter sweep needed to map pattern-forming regions of parameter space and to study stochastic effects on pattern selection.

**The parallel bottleneck:** the simulation is thousands of timesteps, and *every*
timestep must update *every* grid cell from its four neighbours (a 5-point
Laplacian plus local reaction). On an `N×N` grid that is `N²` independent cell
updates per step — the dominant cost. Because each cell's new value depends only
on the *frozen* previous state of its neighbours, all `N²` updates within a step
are independent, so we map **one GPU thread per cell** and advance the whole grid
in parallel. This is the classic **stencil + ping-pong** pattern (see
`docs/PATTERNS.md §1`, exemplified by flagship `6.04` lattice-Boltzmann and
`14.02` reaction-diffusion).

## The algorithm in brief

Turing activator-inhibitor ODE (Gierer-Meinhardt, Schnakenberg, Gray-Scott), explicit or semi-implicit Euler FD, 5-point/7-point Laplacian stencil, Turing instability linear stability analysis (dispersion relation), stochastic Turing patterns (reaction-diffusion master equation), level-set for 3D surface reaction-diffusion.

Concretely, this teaching version implements:

- **Gierer–Meinhardt kinetics** — activator `a` (short-range, autocatalytic) and
  inhibitor `h` (long-range), `da/dt = Da·∇²a + ρa²/h − μ_a·a + ρ_a`,
  `dh/dt = Dh·∇²h + ρa² − μ_h·h`.
- **Explicit (forward) Euler** time integration.
- **5-point Laplacian** with periodic (toroidal) boundaries.
- **Ping-pong double buffering** so every cell reads the frozen previous state.
- **Linear-stability / dispersion relation** — a 2×2 eigenvalue calculation that
  predicts whether (and at what wavelength) a pattern forms, validating the sim.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/reaction-diffusion-morphogenesis-turing-patterns.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/reaction-diffusion-morphogenesis-turing-patterns.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\reaction-diffusion-morphogenesis-turing-patterns.sln /p:Configuration=Release /p:Platform=x64
```

This project links only `cudart_static.lib` (the CUDA runtime) — the stencil is a
hand-written kernel, no extra CUDA libraries.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/turing_params.txt`, prints the
pattern metrics and the linear-stability prediction, shows the GPU-vs-CPU
agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/turing_params.txt` — a tiny, one-line
  **synthetic** model configuration (grid size, diffusion coefficients, reaction
  rates, timestep, step count, noise seed) so the demo runs with zero downloads.
  The grid itself is generated deterministically from a seed at run time.
- **Full dataset:** there is no downloadable "Turing dataset" — the data *is* the
  simulation configuration. `scripts/make_synthetic.py` regenerates it (and lets
  you sweep parameters); `scripts/download_data.ps1` / `.sh` document where real
  *pattern image* references (leopard, zebrafish) and *cortical-folding atlases*
  live if you want to compare against biology.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: Synthetic datasets generated by simulation (no dedicated repository); pigmentation pattern image datasets (leopard, zebrafish from public image sources); cortical folding atlases from HCP (https://db.humanconnectome.org); DANDI morphogenesis imaging (https://dandiarchive.org).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
6.24 -- Reaction-Diffusion Morphogenesis (Turing Patterns)
Gierer-Meinhardt: 64x64 grid, 3000 steps, Da=0.0200 Dh=0.5000 (Dh/Da=25.0) rho=0.050 mu_a=0.100 mu_h=0.140
pattern: mean a=0.624556, min a=0.003645, max a=7.132460, contrast=7.128815, peak cells (a>mean)=907 of 4096
linear stability: Turing regime=YES, max growth=0.040278 at k*=1.1436 (predicted wavelength=5.49 cells)
a along center row (8 samples): 0.3352 0.0146 0.1802 0.0347 1.1905 1.1998 0.1173 0.0649
RESULT: PASS (GPU field matches CPU within tol=1.0e-06)
```

Two independent correctness signals:

1. **GPU vs CPU.** The program computes the result on both the **GPU**
   (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) — running
   the *identical* per-cell update from `src/turing.h` — and asserts they agree
   within `1e-6` (see THEORY §"How we verify correctness").
2. **Simulation vs theory.** The `contrast=7.13` (a clearly non-flat field)
   confirms a pattern formed, and the analytic dispersion relation *independently*
   says `Turing regime=YES` with a fastest-growing mode near `k*≈1.14`. Theory and
   simulation agree — the science, not just CPU==GPU.

Timings print to **stderr** (they vary run-to-run and are not part of the diff).

## Code tour

Read in this order:

1. [`src/turing.h`](src/turing.h) — the shared `__host__ __device__` per-cell
   physics (the model, the Laplacian, the steady state). **Start here.**
2. [`src/main.cu`](src/main.cu) — loads params, runs CPU + GPU, verifies, and
   computes the dispersion relation; reports to stdout/stderr.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the stencil kernel and the ping-pong host loop.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline
   plus the deterministic seeding.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

Custom CUDA stencil kernel (textbook starting point — NVIDIA cuda-samples: https://github.com/NVIDIA/cuda-samples); VCell (https://vcell.org) — GUI reaction-diffusion PDE simulator with spatial stochastic mode; MOOSE (https://github.com/BhallaLab/moose-core) — compartmental spatial simulation; GillesPy2 (https://github.com/GillesPy2/GillesPy2) — stochastic Turing pattern simulation.

- **NVIDIA cuda-samples** — the canonical reference for hand-written stencil
  kernels and shared-memory tiling; study its finite-difference examples to see
  the halo/tiling optimization this project leaves as an exercise.
- **VCell** — a production reaction-diffusion PDE simulator (with a spatial
  stochastic mode); shows what a full, validated morphogenesis tool offers beyond
  this teaching stencil (complex geometry, membrane flux, deterministic + Gillespie).
- **MOOSE** — compartmental spatial simulation used in systems neuroscience;
  illustrates unstructured meshes vs our regular grid.
- **GillesPy2** — stochastic Turing patterns via the reaction-diffusion master
  equation; the "noise selects the pattern" story our deterministic sim omits.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

A **2-D stencil with ping-pong double buffering**: one thread per grid cell, a
16×16 thread-block tile, and two device buffer pairs swapped each timestep so
every cell reads the frozen previous state (no races, no atomics). The catalog
also lists texture memory for read-only species arrays and shared-memory tiling
for the halo — genuine optimizations that this teaching version deliberately
omits for clarity and calls out as exercises below. (Catalog note: "Custom 2D/3D
CUDA stencil kernels with halo-exchange; texture memory for read-only species
arrays; shared memory for 7-point stencil tile computation; CUDA Thrust for
reduction; pattern: 3D thread-block tiling, one thread per grid cell.")

## Exercises

1. **Change the pattern.** Edit `data/sample/turing_params.txt` (or rerun
   `scripts/make_synthetic.py --Dh 0.2`) and watch the contrast and predicted
   wavelength change. Find the `Dh/Da` ratio below which `Turing regime=NO`.
2. **Shared-memory tiling.** Rewrite `rd_step_kernel` to load a 16×16 tile plus a
   1-cell halo into `__shared__` memory, so each cell reads neighbours from shared
   instead of global memory. Measure the speed-up on a 512×512 grid.
3. **Neumann (zero-flux) boundaries.** Replace the periodic wrap in
   `tu_laplacian` with reflecting boundaries — biologically, a tissue with edges —
   and observe how the pattern changes near the border.
4. **A second model.** Add Schnakenberg or Gray-Scott kinetics as an alternate
   `tu_update` and compare the pattern zoo. Keep the shared-header idiom so CPU
   and GPU stay in lockstep.
5. **Stochastic Turing.** Add per-cell Gaussian noise each step (a crude
   reaction-diffusion master equation) and study how noise shifts pattern
   selection — the effect GillesPy2 models rigorously.

## Limitations & honesty

- **Synthetic, not measured.** The input is a chosen parameter point, labeled
  synthetic everywhere. The output is a *model* pattern; it is **not** a claim
  about any real animal's markings and carries **no clinical meaning**.
- **Reduced scope.** This is a 2-D, single-precision-free (FP64), deterministic,
  explicit-Euler teaching stencil. Production morphogenesis tools (VCell, MOOSE)
  add complex geometry, membrane fluxes, implicit/adaptive time-stepping, 3-D
  surfaces (level-set), and stochastic (Gillespie) modes — described in THEORY
  §"Where this sits in the real world."
- **Explicit-Euler stability.** The timestep must satisfy the diffusion CFL limit
  (`dt·Dh·4 < 1` roughly); too large a `dt` makes the field blow up to NaN. The
  committed sample is well inside the stable region.
- **GPU vs CPU drift.** Over 3000 nonlinear steps the GPU's fused multiply-add and
  the host compiler diverge at the ~`1e-12` level; we verify to `1e-6`, far below
  "same pattern" but strict enough to catch real bugs (PATTERNS.md §4).
