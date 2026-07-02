# 6.22 — Bone Remodeling Simulation

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.22`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._ This is a
> **reduced-scope teaching version** (CLAUDE.md §13): it keeps the remodeling
> biology and swaps the production finite-element stress solve for a cheap,
> physically-motivated stencil proxy. THEORY.md §7 spells out the difference.

## Summary

Living bone constantly rebuilds itself in response to the loads it carries —
**Wolff's law**: material is added where bone is over-loaded and removed where it
is idle, so the internal trabecular lattice ends up aligned with the stress path.
This project simulates that feedback on a 2-D voxel grid: each remodeling step
(1) settles a **mechanical-stimulus field** through the tissue, then (2) applies
the **mechanostat rule** (Frost's dead band / Huiskes' strain-energy-density rule)
to grow or shrink each voxel's density. Given a localized load on the top edge,
the model carves an oriented **trabecular strut** down the load path while the
lightly-loaded flanks resorb — a recognizable illustration of bone adaptation,
computed on both the GPU and a CPU reference and verified to agree.

## What this computes & why the GPU helps

Bone continually remodels in response to mechanical loading: osteoclasts resorb
bone and osteoblasts form new bone in a coupled feedback loop. GPU simulation
enables voxel-level analysis of trabecular microstructure (µCT at 10–50 µm
resolution yields ~10⁸ voxels) and tracking remodeling over years of simulated
time. Topology-optimization algorithms (SIMP) on GPU-FEM underlie both bone
remodeling models and prosthesis design.

**The parallel bottleneck:** the per-voxel updates. Both phases — the many
**Jacobi relaxation sweeps** that settle the stimulus field, and the **mechanostat
density update** — touch every voxel independently using only nearest-neighbour
values. That is a **stencil**: give each voxel its own GPU thread and the whole
grid updates in parallel. At µCT scale (10⁸ voxels × thousands of sweeps × years
of steps) this per-voxel parallelism is the difference between minutes and days.

## The algorithm in brief

- **Mechano-regulation (Prendergast/Huiskes) + Frost mechanostat:** a per-voxel
  dead-band rule that forms/resorbs bone based on the local stimulus vs. a setpoint.
- **Strain-energy-density (SED) remodeling signal:** `phi = S / rho` (stimulus per
  unit bone) drives the rule.
- **Density-weighted diffusion (Jacobi relaxation):** a 5-point stencil that
  transports the mechanical stimulus through stiff material — our teaching proxy for
  the production finite-element `K u = f` solve.
- **Stencil + ping-pong buffers on the GPU:** one thread per voxel, two device
  buffers swapped each sweep (no atomics, no races).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/bone-remodeling-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/bone-remodeling-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\bone-remodeling-simulation.sln /p:Configuration=Release /p:Platform=x64
```

Both `Release|x64` and `Debug|x64` build with zero warnings; the project links only
the CUDA runtime (`cudart_static.lib`) — the stencil proxy needs no extra library.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (optional CMake build)
```

The demo builds if needed, runs on `data/sample/bone_params.txt`, prints the
remodeled-bone summary, shows the GPU-vs-CPU agreement check, and prints a timing
line to stderr.

## Data

- **Sample (committed):** `data/sample/bone_params.txt` — a tiny, **synthetic**
  parameter file (12 numbers) so the demo runs offline with zero downloads. It is
  not patient data.
- **Regenerate / resize:** `python scripts/make_synthetic.py [--nx N --ny M ...]`.
- **Full / real datasets:** `scripts/download_data.ps1` / `.sh` print pointers to
  real bone-imaging datasets (OAI µCT, PhysioNet, BoneJ, MICCAI) and never bypass
  their registration.
- **Provenance, field meanings & license:** see [data/README.md](data/README.md).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program remodels the grid on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and asserts the final density fields agree to
within `1e-9` (observed difference ~`1.1e-16`). The headline is the **per-column
bone mass profile**, which peaks under the loaded footprint (columns 10–13) and
falls to the density floor on the flanks — the load-aligned trabecular strut. The
line ends in `RESULT: PASS`.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads params, runs CPU + GPU, verifies, reports.
2. [`src/bone_remodel.h`](src/bone_remodel.h) — the shared `__host__ __device__`
   physics: the Jacobi stencil and the mechanostat rule (start of the "why").
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the two kernels and the host drive loop.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **FEBio** — <https://github.com/febiosoftware/FEBio>: production nonlinear FEM for
  bone & cartilage; study its element assembly and iterative solver — the real
  version of the stress step this project proxies.
- **ParOSol / VoxFEM** (ETH Zürich research): parallel/GPU voxel FEM for trabecular
  bone; the archetype for the cuSPARSE-assembly + PCG-solve pipeline.
- **FreeFEM** — <https://freefem.org>: general PDE/FEM solver adaptable to remodeling.
- **OpenFOAM** — <https://github.com/OpenFOAM/OpenFOAM-dev>: poroelastic
  fluid–structure bone modeling.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Stencil + ping-pong buffers** (PATTERNS.md §1, "grid PDE / nearest-neighbour
update"; same shape as flagships 6.04 and 14.02). A 2-D thread grid matches the
voxel layout — thread `(x,y)` owns voxel `(x,y)` — and two device buffers are
swapped between sweeps so reads and writes never collide (no atomics). The
production pattern noted in the catalog adds **cuSPARSE** (voxel-FEM assembly) and
**cuSOLVER PCG** (the `K u = f` solve); this teaching version replaces that solve
with a custom stencil kernel and documents the trade in THEORY.md §7.

## Exercises

1. **Shared-memory halo tiling.** The stencil is bandwidth-bound. Tile each block's
   `S`/`rho` neighbourhood (including a 1-voxel halo) into shared memory and measure
   the speed-up vs. the current global-memory reads.
2. **Move the loop into one kernel.** Fuse the `relax_iters` sweeps into a single
   kernel with a block-level barrier (cooperative groups) to cut launch overhead on
   large grids — then compare timings.
3. **A different load case.** Edit `data/sample/bone_params.txt` (or use
   `make_synthetic.py --load-x0 ... --load-x1 ...`) to place two separate footprints
   and watch two struts form; add a support only under part of the base.
4. **Anisotropy / a real setpoint law.** Replace the scalar `phi = S/rho` with a
   density-dependent setpoint (e.g. `k` scaling with `rho`) and see how the
   equilibrium architecture changes.
5. **3-D.** Extend the grid and stencil to 3-D (a 7-point stencil, `D3Q7`-style) —
   the step toward a real µCT-scale simulation.

## Limitations & honesty

- **Reduced-scope teaching model (CLAUDE.md §13).** The mechanical-stimulus field is
  a **density-weighted diffusion proxy**, *not* a finite-element stress solve. It
  captures the remodeling feedback qualitatively but is not quantitatively
  calibrated and must not be read as a real stress/strain field.
- **Synthetic data only.** `data/sample/bone_params.txt` is a set of dimensionless
  knobs generated by `make_synthetic.py`, labeled synthetic everywhere. No patient
  data is involved and no output implies clinical validity.
- **2-D and tiny.** The demo grid is 24×16 for legibility; on this size the GPU is
  *slower* than the CPU because the many small kernel launches are launch-bound
  (an honest teaching artifact — see THEORY.md §7). The GPU's advantage appears at
  µCT scale (10⁸ voxels).
- **Simplified biology.** No explicit RANKL/OPG ODEs, cell populations, mineralization
  lag, or anisotropic fabric — all folded into one scalar mechanostat rule.
