# 6.1 — Cardiac Electrophysiology Simulation

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.1`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

The heart beats because an electrical wave sweeps across it, telling muscle cells
when to contract. This project **simulates that wave** on a sheet of cardiac
tissue: we spark a small patch of cells and watch a self-sustaining
*action-potential* front travel outward, exactly as it does across real
myocardium. Mathematically it is a **reaction-diffusion PDE** — a per-cell ODE
("reaction": each cell fires and recovers) coupled to spatial diffusion ("cells
nudge their neighbours"). Both pieces are massively parallel across cells, which
is why cardiac electrophysiology is a flagship GPU application. This teaching
build uses the classic **FitzHugh-Nagumo** two-variable cell model on a 2-D grid
so the mechanism is visible without the 50–200-variable complexity of production
ionic models.

## What this computes & why the GPU helps

Simulates transmembrane voltage propagation across cardiac tissue by solving the monodomain or bidomain reaction-diffusion PDE coupled to stiff ODEs representing ionic channel kinetics (e.g., ten Tusscher-Panfilov, O'Hara-Rudy). Each voxel integrates 50–200 state variables per time step at sub-millisecond temporal resolution; a whole-heart simulation at 0.1 mm spatial resolution yields ~10⁸ nodes, making the per-node ODE update embarrassingly parallel. The GPU eliminates the otherwise serial per-cell Rush-Larsen / RL2 exponential gating integration. Operator splitting decouples the reaction (GPU-parallel ODE) from diffusion (sparse linear solve), and CUDA kernels saturate memory bandwidth on the former while cuSPARSE handles the latter.

**The parallel bottleneck:** the time loop dominates the runtime — thousands of
timesteps, and each timestep touches *every* cell twice (a reaction update and a
diffusion update). Both updates are per-cell and depend only on the cell (reaction)
or its 4 grid neighbours (diffusion), so there is **no serial dependency across
cells within a step**. We therefore assign **one GPU thread per cell** and launch
two kernels per step. A whole heart is ~10⁸ cells; a single CPU core walking them
one at a time is the bottleneck the GPU removes.

## The algorithm in brief

The catalog names the full production toolkit: *monodomain/bidomain
reaction-diffusion, operator splitting (Strang/Godunov), Rush-Larsen explicit
gating, Crank-Nicolson implicit diffusion, conjugate gradient with ILU(0)
preconditioning, finite volume/finite element spatial discretization.* This
teaching build implements the didactic core of that stack:

- **Monodomain reaction-diffusion PDE** — `∂V/∂t = D∇²V − I_ion(V,w)`.
- **FitzHugh-Nagumo cell model** — the 2-variable reduction of the stiff ionic
  ODEs (the slot where ten Tusscher / O'Hara-Rudy would plug in).
- **Operator splitting (Godunov)** — advance reaction and diffusion in separate
  half-steps each timestep, so the pointwise ODE and the spatial coupling
  decouple.
- **Explicit forward-Euler diffusion** — a 5-point Laplacian **stencil** with
  no-flux (Neumann) boundaries, subject to the CFL stability bound
  `dt ≤ dx²/(4D)`.

The pieces we *describe but do not implement* (implicit Crank-Nicolson diffusion,
CG + ILU(0) via cuSPARSE/cuSOLVER, Rush-Larsen gating) are covered in THEORY §"real
world" — they buy larger stable timesteps, not different physics.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/cardiac-electrophysiology-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/cardiac-electrophysiology-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\cardiac-electrophysiology-simulation.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the result, shows the
GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/` — a tiny, offline input so the demo runs
  with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (documented, idempotent).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: PhysioNet MIT-BIH & MIMIC-III Waveform — 40 000+ ICU ECG/hemodynamic waveforms (https://physionet.org); CellML Physiome Repository — curated ionic cell models in CellML/SBML format importable by openCARP (https://models.physiomeproject.org); UK Biobank Cardiac MRI — 100 000+ cine CMR studies, access via application (https://www.ukbiobank.ac.uk); ACDC MICCAI Cardiac Challenge — 100-patient CMR with LV/RV/myocardium ground truth (https://www.creatis.insa-lyon.fr/Challenge/acdc/).

## Expected output

Success looks like `demo/expected_output.txt`. The program computes the result on
both the **GPU** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`)
and asserts they agree within the documented tolerance — that agreement is the
correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the tissue setup, runs CPU + GPU, verifies, reports the voltage slice.
2. [`src/cardiac_cell.h`](src/cardiac_cell.h) — **the physics**: the shared `__host__ __device__` FHN reaction (`react_step`) + diffusion stencil (`diffuse_cell`). Read this to understand the model.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the stencil/ping-pong idea.
4. [`src/kernels.cu`](src/kernels.cu) — the two kernels (react, diffuse) and the host time loop.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline (same physics, plain loops).
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

openCARP (https://git.opencarp.org/openCARP/openCARP) — MPI+CUDA cardiac EP solver, CARPutils Python scripting, v19.0 April 2026; MonoAlg3D_C (https://github.com/rsachetto/MonoAlg3D_C) — finite-volume GPU monodomain solver with Purkinje coupling and MPI batch dispatch; Cardioid/LLNL (https://github.com/llnl/cardioid) — multiscale cardiac suite (EP + mechanics + ECG), CUDA optional, Gordon Bell finalist; Chaste (https://github.com/Chaste/Chaste) — Oxford bidomain solver with cardiac mechanics module.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

The catalog target is: *cuSPARSE (diffusion SpMV), cuSOLVER (linear system), CUDA
custom kernels (per-cell ODE Rush-Larsen); pattern: fine-grained thread-per-cell
ODE + coarse SpMV for diffusion; streams for overlapping compute and halo
exchange.*

This teaching build realizes the **stencil + per-cell-ODE** half of that (see
[docs/PATTERNS.md](../../../docs/PATTERNS.md) §1, the pattern shared with the
`6.04` lattice-Boltzmann and `14.02` reaction-diffusion flagships):

- **One thread per grid cell**, on a 2-D `16×16` block grid.
- **Two kernels per timestep** — `react_kernel` (pointwise FHN ODE) then
  `diffuse_kernel` (5-point Laplacian **stencil**).
- **Ping-pong buffers** for diffusion: read `V_in`, write `V_out`, swap — so a
  thread never sees a half-updated neighbour (race-free, no atomics).
- **Shared `__host__ __device__` core** (`cardiac_cell.h`) so the CPU reference
  and GPU kernels run byte-for-byte identical math (exact verification).

The production route — implicit diffusion as a sparse `SpMV` + conjugate-gradient
solve via **cuSPARSE/cuSOLVER**, plus streams overlapping halo exchange — is
described in THEORY §"real world"; we keep the explicit stencil here because it is
the clearest first version of the same idea.

## Exercises

1. **Watch the wave move.** Re-run with more steps
   (`python scripts/make_synthetic.py --steps 800`) and compare the voltage slice
   — the `activated(V>0.5)` count should grow as the front advances. How many
   steps until the wave reaches the right edge?
2. **Measure conduction velocity.** From two runs at different step counts, find
   how far the front (the `V=0.5` crossing) moved per step. That ratio is the
   *conduction velocity* — the single most clinically important EP number.
3. **Create a re-entry / spiral.** Change `make_synthetic.py` to add an S2
   stimulus in a partially-refractory region (the classic "S1-S2 cross-field"
   protocol) and see if you can launch a rotating spiral wave — the mechanism of
   many arrhythmias.
4. **Break stability on purpose.** Set `--dt 3.0` (above the CFL limit
   `dx²/4D = 2.5`) and observe the loader reject it; then relax the check and
   watch the explicit solver blow up. This is why production uses *implicit*
   diffusion.
5. **Swap the cell model.** Replace `react_step` in `cardiac_cell.h` with a
   3-variable model (e.g. Aliev-Panfilov) — the kernels and verification are
   unchanged, proving the value of the shared `__host__ __device__` core.

## Limitations & honesty

- **Synthetic & dimensionless.** The setup is generated by
  `scripts/make_synthetic.py`; FitzHugh-Nagumo units are a nondimensional
  caricature, **not** millivolts/milliseconds. Nothing here is patient data.
- **Reduced-scope teaching model.** FHN is a 2-variable stand-in for the
  50–200-variable ionic models (ten Tusscher-Panfilov, O'Hara-Rudy) used in
  research. It reproduces the *qualitative* excitable dynamics (threshold,
  upstroke, refractoriness, propagation), not quantitative APD/restitution.
- **Explicit diffusion.** We use an explicit stencil (CFL-limited timestep), not
  the unconditionally-stable implicit Crank-Nicolson + CG solve that production
  codes (openCARP, MonoAlg3D) use. This caps how large `dt` can be.
- **2-D, single domain.** Real hearts are 3-D anisotropic tissue with
  fiber orientation, a Purkinje network, and (for bidomain) a separate
  extracellular potential. This is a 2-D isotropic *monodomain* sheet.
- **Not for clinical use.** This is study material demonstrating a GPU pattern; it
  is not a validated electrophysiology simulator.
