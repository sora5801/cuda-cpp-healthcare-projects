# 6.8 — Tumor Growth & Treatment-Response Modeling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.8`
>
> _Educational only — not for clinical use (see CLAUDE.md §8). All data is synthetic._

## Summary

This project simulates how a solid tumor **grows and spreads** through tissue and
how it **responds to radiotherapy** — the two core dynamics of mathematical
oncology — using the smallest honest model. A single density field `u(x,y)` (tumor
cells per unit tissue, normalized to `[0,1]`) evolves by the **Fisher-KPP
reaction-diffusion** equation (proliferation + infiltration), and a **fractionated
radiotherapy** course kills cells according to the **linear-quadratic (LQ)**
radiobiological model. The program runs the whole simulation on the **GPU** as a
finite-difference **stencil**, checks it against a serial **CPU reference**, and
reports the modelled treatment response (percent tumor-burden reduction) versus an
untreated control.

## What this computes & why the GPU helps

Continuum-PDE models of tumor cell density (reaction-diffusion), combined with
radiobiological kill models, capture avascular tumor growth and response to
radiation. Solving the PDE means advancing every grid cell **many times** (once per
timestep of a multi-day course), and each cell's update depends only on its
immediate neighbours.

**The parallel bottleneck:** the per-timestep **field update** — a 5-point
Laplacian plus a logistic reaction at every one of `nx·ny` cells (and, in
production, `256³–512³` voxels). It is `O(steps · cells)` and dominates the
runtime. Every cell within a timestep is **independent**, so we map **one GPU
thread per cell** and advance the whole field in one parallel sweep, double-buffered
so there are no races. Parameter sweeps for *in-silico* clinical trials add a second,
embarrassingly-parallel axis on top.

## The algorithm in brief

- **Fisher-KPP reaction-diffusion** for tumor cell density: `∂u/∂t = D∇²u + ρu(1−u)`,
  integrated with explicit Euler and a 5-point finite-difference Laplacian.
- **Linear-quadratic (LQ) radiobiology** for treatment: each fraction multiplies
  every cell by the surviving fraction `S(d) = exp(−(αd + βd²))`.
- **Stencil + ping-pong** GPU pattern: one thread per cell, two buffers swapped
  each step; a separate per-cell kernel applies each radiotherapy fraction.
- **Verification** against an independent serial CPU implementation of the same
  physics, plus analytic checks (Fisher wave speed, LQ survival by hand).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/tumor-growth-treatment-response-modeling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/tumor-growth-treatment-response-modeling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\tumor-growth-treatment-response-modeling.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/tumor_params.txt`, prints the
result, shows the GPU-vs-CPU agreement check for **both** the treated and control
scenarios, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/tumor_params.txt` — a tiny, **synthetic**
  parameter line (grid, growth constants, RT schedule) so the demo runs offline
  with zero downloads. The tumor field is built deterministically from it.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print real-data pointers;
  there is nothing to download to run the demo.
- **Provenance & license:** see [data/README.md](data/README.md).

Real mathematical-oncology models calibrate against: TCGA (multi-omics + imaging,
<https://portal.gdc.cancer.gov>), TCIA (tumor imaging,
<https://www.cancerimagingarchive.net>), PhysioNet (<https://physionet.org>), and
Zenodo simulation datasets. None are required here.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). From a
1 mm seed the untreated tumor grows to a core radius of ~9 mm; the treated run
(10 × 2 Gy, per-fraction survival `S ≈ 0.70`, BED = 24 Gy) ends with **~28% less**
tumor burden. The program computes each scenario on both the **GPU**
(`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and asserts
they agree within `1e-6` (in practice `~2e-16`, machine epsilon) — that agreement,
plus the analytic wave-speed / LQ checks, is the correctness guarantee.

## Code tour

Read in this order:

1. [`src/tumor.h`](src/tumor.h) — the shared `__host__ __device__` physics: the
   Laplacian, the Fisher-KPP update, and the LQ survival. **Start here.**
2. [`src/main.cu`](src/main.cu) — loads params, seeds the tumor, runs CPU + GPU
   for the treated and control scenarios, verifies, reports.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the stencil idea.
4. [`src/kernels.cu`](src/kernels.cu) — the growth/treatment kernels + time loop.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **PhysiCell** (<https://github.com/MathCancer/PhysiCell>) — 3-D agent-based
  multicellular simulator with diffusing substrates; learn its BioFVM diffusion
  solver and linear scaling in cell count.
- **PhysiBoSS** (<https://github.com/PhysiBoSS/PhysiBoSS>) — PhysiCell + MaBoSS
  Boolean intracellular signaling; learn how intracellular decisions couple to
  the tissue field.
- **Chaste** (<https://github.com/Chaste/Chaste>) — tumor spheroid and crypt
  models; a mature computational-biology framework to study structure.
- **OpenFOAM** (<https://github.com/OpenFOAM/OpenFOAM-dev>) — CFD used for
  drug-delivery flow coupled to tumor models.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Custom finite-difference stencil kernels** with **ping-pong double buffering**:
a 2-D CUDA thread grid over the density field runs a 5-point Laplacian + logistic
reaction each timestep, and a separate per-cell kernel applies each LQ radiotherapy
fraction. This is the stencil pattern of the lattice-Boltzmann flagship (`6.04`)
and the reaction-diffusion project (`14.02`). A full 3-D agent-based version would
add **Thrust** (sort/bin cells) and **cuRAND** (stochastic division/death) — see
THEORY §7.

## Exercises

1. **Untreated vs. treated visually.** Dump the final field to a CSV and plot it;
   compare the control and treated cores. (Hint: add a small writer in `main.cu`.)
2. **Schedule sweep.** Regenerate the config with different schedules
   (`--dose 3 --n-fractions 10`, or hypofractionation) and see how the burden
   reduction and BED change. Which schedule spares the *front* least?
3. **Shared-memory tiling.** Rewrite `tumor_grow_kernel` to stage each block's
   `16×16` tile plus a 1-cell halo into `__shared__` memory, removing the ~5×
   redundant global reads. Measure the speedup; verify the field is unchanged.
4. **Verify the wave speed.** Track the position where `u = 0.5` along the centre
   row over time and confirm the front moves at `c = 2√(Dρ)`.
5. **Add a hypoxia field.** Introduce a second diffusion-consumption PDE for
   oxygen and make `ρ` (and the LQ `α`,`β`) drop in low-O₂ regions — the first
   step toward a production coupled model.

## Limitations & honesty

- **Synthetic, not calibrated.** The parameters are illustrative; nothing here is
  fit to a real patient. Do **not** treat any number as clinically meaningful.
- **Reduced scope (CLAUDE.md §13).** One normalized density field only — no
  explicit oxygen/hypoxia, no necrosis, no drug PK/PD field, no agent-based cells,
  and 2-D rather than 3-D. LQ kill is applied as an instantaneous per-cell
  multiply. THEORY §7 describes what a production model adds.
- **Periodic boundaries** (a torus), chosen for a clean stencil; the sample keeps
  the tumor away from the edges so this does not affect the result.
- **Timing is a teaching artifact**, never a benchmark claim (CLAUDE.md §12): on
  this tiny grid the many small kernel launches are partly launch-bound; the GPU's
  advantage grows with grid size, which is why 3-D clinical models need it.
