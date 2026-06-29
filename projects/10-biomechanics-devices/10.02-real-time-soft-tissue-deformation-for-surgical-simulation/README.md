# 10.02 — Real-Time Soft-Tissue Deformation for Surgical Simulation

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biomechanics%2C%20Devices%20%26%20Surgery-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 10: Biomechanics, Biomedical Devices & Surgery · Catalog ID `10.02`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Deform soft tissue in real time with **Position-Based Dynamics (PBD)**: model the
tissue as a grid of mass **particles** linked by distance **constraints**, then
each frame predict positions under gravity and iteratively **project** them to
satisfy the constraints. With a **Jacobi** scheme every particle computes its
correction from its neighbours independently — one thread per particle. Ninth
distinct GPU pattern in the flagships: **parallel constraint projection**.

## What this computes & why the GPU helps

Surgical simulators need **sub-10 ms** deformation updates on organ meshes of
10⁵+ elements so haptic devices feel responsive. PBD makes this tractable: it
skips force integration and directly moves positions to meet constraints, and its
projections are data-parallel. The demo pins the top edge of a sheet and lets the
rest drape under gravity, reaching a stable equilibrium.

**The parallelized work** is the per-particle constraint projection, run for
several Jacobi iterations per timestep; each particle reads neighbours and writes
its own corrected position (no atomics).

## The algorithm in brief

- **Predict:** `p = x + v·dt + g·dt²` (free particles; pinned ones hold).
- **Project (×iters):** for each distance constraint, move endpoints toward the
  rest length, weighted by inverse mass; Jacobi-average per particle.
- **Finalize:** `v = (p − x)/dt`, then commit `x = p`.

See [THEORY.md](THEORY.md) for the constraint math, XPBD, and the FP-reproducibility note.

## Build

Requires **Visual Studio 2026** (v145) + **CUDA 13.3** ([docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/real-time-soft-tissue-deformation-for-surgical-simulation.sln`.
2. **`Release|x64`** → **Build** → `build/x64/Release/real-time-soft-tissue-deformation-for-surgical-simulation.exe`.

CLI: `msbuild build\real-time-soft-tissue-deformation-for-surgical-simulation.sln /p:Configuration=Release /p:Platform=x64`

## Run the demo

```powershell
./demo/run_demo.ps1
```

Simulates the draping sheet on CPU + GPU and verifies the meshes match.

## Data

- **Sample (committed):** `data/sample/cloth_params.txt` — the mesh + solver settings.
- **Realistic meshes:** SOFA / iMSTK / FleX (organ meshes, haptics) — see
  `scripts/download_data.ps1` and [data/README.md](data/README.md).
- Bigger mesh: `python scripts/make_synthetic.py --R 128 --C 128 --steps 600`.

## Expected output

`demo/expected_output.txt` holds the deterministic sampled positions and drape
depth. The GPU (`src/kernels.cu`) and CPU (`src/reference_cpu.cpp`) share the PBD
math (`src/pbd.h`); over thousands of iterations they drift at ~`1e-5` (float
FMA), so we verify to `1e-3` on positions of magnitude ~10 — agreement to ~6
significant figures (see THEORY).

## Code tour

1. [`src/main.cu`](src/main.cu) — load, build mesh, CPU + GPU simulate, verify, print.
2. [`src/pbd.h`](src/pbd.h) — **Vec3 math + the per-particle predict/project/finalize** (host + device).
3. [`src/kernels.cuh`](src/kernels.cuh) — the three GPU kernels (one thread per particle).
4. [`src/kernels.cu`](src/kernels.cu) — predict / Jacobi-project (ping-pong) / finalize + the time loop.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — mesh init + the serial reference.

## Prior art & further reading

- **SOFA** (<https://github.com/sofa-framework/sofa>) — physics engine with GPU PBD + haptics.
- **iMSTK** (<https://github.com/Kitware/iMSTK>) — interactive medical simulation toolkit (CUDA).
- **NVIDIA FleX** (<https://github.com/NVIDIAGameWorks/FleX>) — GPU PBD particle solver.
- Müller et al. (2007), *Position Based Dynamics*; Macklin et al. (2016), *XPBD*.

Study these for production deformation; reimplement the pattern didactically (CLAUDE.md §2).

## CUDA pattern used here

**Parallel constraint projection** (Jacobi): one thread per particle, double-
buffered across iterations, no atomics · shared `__host__ __device__` PBD math for
CPU/GPU parity · three kernels per step (predict / project / finalize).

## Exercises

1. **XPBD.** Add compliance (inverse stiffness) so the material's behaviour is
   independent of iteration count — the modern standard.
2. **Gauss-Seidel via graph colouring.** Colour the constraints so same-colour
   constraints are independent, then project colour-by-colour (faster convergence).
3. **Self-collision / a probe.** Add a moving sphere (the surgical tool) that
   pushes the tissue — collision constraints.
4. **Bending constraints.** Add dihedral/bending constraints so the sheet resists
   folding (more tissue-like).
5. **Scale up.** Run `--R 256 --C 256` and measure where the GPU decisively beats
   the CPU; profile the constraint kernel.

## Limitations & honesty

- **Grid sheet, distance constraints only** (no volume/tetrahedral elements, no
  bending, no collisions). Real organs use tetra meshes + XPBD/FEM/MPM.
- **One kernel launch per (predict/iter/finalize)** ⇒ launch-bound on small meshes;
  the GPU win grows with mesh size.
- **FP reproducibility:** CPU and GPU drift ~`1e-5` over thousands of iterations
  (different FMA); we verify to a physically-negligible `1e-3` (THEORY).
