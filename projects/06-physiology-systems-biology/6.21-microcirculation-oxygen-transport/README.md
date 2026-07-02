# 6.21 — Microcirculation & Oxygen Transport

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.21`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Tissue stays alive only if oxygen carried by red blood cells in the capillaries
can diffuse out to every cell before the cells consume it. This project computes
the steady-state **oxygen partial pressure (PO2) field** in a small block of tissue
fed by a few capillaries, using the **Green's-function method** of Secomb & Hsu: it
treats each short capillary segment as a point source of oxygen, superposes each
source's diffusion field at every tissue grid point, and subtracts the tissue's
Michaelis-Menten oxygen consumption. The output is a map of where the tissue is
well-oxygenated and where it is **hypoxic** — the corners farthest from any vessel.
It is a compact, faithful teaching version of a real quantitative-physiology tool,
built to showcase a fundamental GPU pattern: an all-pairs (source × grid-point) sum
mapped as one thread per output point with shared-memory tiling of the sources.

## What this computes & why the GPU helps

Oxygen delivery from red blood cells to tissue parenchyma involves convection in
capillaries, diffusion through capillary walls and interstitium (Krogh cylinder /
Green's-function models), and intracellular O₂ reaction/consumption (Michaelis-
Menten kinetics). A realistic tissue volume (~1 mm³) contains thousands of
capillaries; the Green's-function O2 field is a volumetric superposition — an
O(N_grid × N_source) all-pairs integral.

**The parallel bottleneck:** the tissue PO2 at each grid point is a sum over *every*
capillary source: `PO2_i = po2_inflow + Σ_j q_j·G(|x_i − x_j|) − consumption`. Every
grid point is independent, so we assign **one GPU thread per grid point**; each
thread gathers the contribution of all sources. With N grid points and M sources
this is N×M independent flops — exactly the kind of arithmetically-dense,
data-parallel workload the GPU excels at. (A production solver replaces the O(N²)
direct sum with a fast multipole method; see THEORY.md.)

## The algorithm in brief

- **Krogh / Green's-function O2 transport:** each capillary segment is a point
  source; PO2 is the linear superposition of every source's `G(r) = 1/(4πK·r)`
  diffusion field (regularized at the capillary radius).
- **Hill hemoglobin saturation:** `S(P) = Pⁿ/(P50ⁿ+Pⁿ)` sets each segment's O2
  source strength from its blood PO2.
- **Michaelis-Menten O2 consumption:** `M(P) = M0·P/(P+Km)` is the tissue O2 demand.
- **All-pairs GPU sum:** one thread per tissue grid point; sources staged through
  shared memory in tiles; the fast multipole method (FMM) that accelerates this to
  O(N log N) in production is described in THEORY.md.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/microcirculation-oxygen-transport.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/microcirculation-oxygen-transport.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\microcirculation-oxygen-transport.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/microvessel_network.txt`, prints
the PO2-field summary, shows the GPU-vs-CPU agreement check, and prints a timing
line to stderr.

## Data

- **Sample (committed):** `data/sample/microvessel_network.txt` — a tiny,
  **synthetic** tissue-grid + capillary layout so the demo runs offline with zero
  downloads. Engineered so one corner is a hypoxic pocket (a "known answer").
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print pointers and
  registration instructions (idempotent; never bypass credentials).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: Vascular Model Repository (http://www.vascularmodel.com);
two-photon microscopy microvascular datasets from Allen Institute
(https://portal.brain-map.org); PhysioNet oxygen saturation waveforms
(https://physionet.org); published microvascular network datasets (Secomb group,
verify at secomb.org).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): a
12×12×8 = 1152-point tissue block fed by 18 capillary sources, with min PO2 ≈ 8.4
mmHg, mean ≈ 18.3 mmHg, and ~3.5% of the tissue hypoxic (< 10 mmHg), concentrated
at the far corner (8.4 mmHg) while the well-perfused centre reaches ~24 mmHg. The
program computes the field on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and asserts they agree within **1e-9 mmHg**
(they actually match to ~1e-14 because both call the identical `solve_point()` math
in the identical summation order) — that agreement is the correctness guarantee.

## Code tour

Read in this order:

1. [`src/oxygen.h`](src/oxygen.h) — the physics: Green's function, Hill saturation,
   Michaelis-Menten consumption, all shared `__host__ __device__`.
2. [`src/reference_cpu.h`](src/reference_cpu.h) — the data containers (`TissueGrid`,
   `OxySource`) and `solve_point()`, the shared per-grid-point evaluator.
3. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
5. [`src/kernels.cu`](src/kernels.cu) — the kernel (one thread per grid point, with
   shared-memory tiling of the sources) and its host wrapper.
6. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the loader + trusted serial baseline.
7. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **Secomb group Green's-function method** (verify at <https://secomb.org>) — the
  canonical reference implementation of exactly this O2-transport model; study how
  it solves for the source strengths self-consistently.
- **HemeLB** (<https://github.com/hemelb-codes/hemelb>) — sparse lattice-Boltzmann
  for capillary *flow*; learn how the convective side is modelled at scale.
- **USERMESO-2.0** (<https://github.com/AnselGitAccount/USERMESO-2.0>) — GPU red-
  blood-cell hemodynamics with deformable membranes; the cellular-scale picture.
- **APBS** (<https://github.com/Electrostatics/apbs>) — an electrostatics solver;
  its Poisson machinery is mathematically the *same* Laplace/Green's-function
  problem, which is why it is repurposable for O2 diffusion.
- **OpenFOAM** (<https://github.com/OpenFOAM/OpenFOAM-dev>) — volume-averaged tissue
  oxygenation via continuum CFD.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent jobs + gather with shared-memory tiling** (`docs/PATTERNS.md` §1, the
`1.12`/`12.01` family). One thread owns one tissue grid point and gathers the
Green's-function contribution of every capillary source; the block cooperatively
stages sources into shared memory one tile at a time so the source array is read
from fast on-chip memory rather than repeatedly from global memory. The per-point
physics is shared with the CPU via a `__host__ __device__` header (`docs/PATTERNS.md`
§2) for exact CPU/GPU parity. The catalog's forward-looking note — cuFMM/NUFFT for
the Green's-function sums, cuSPARSE for the network flow solve, junction mass-balance
reductions — describes the full-scale version discussed in THEORY.md.

## Exercises

1. **Grow the problem.** Regenerate a bigger block
   (`python scripts/make_synthetic.py --nx 32 --ny 32 --nz 24`) and watch the
   GPU-vs-CPU timing gap widen as N_grid×N_src grows (edit the tolerance? no — it
   stays exact). Which crosses over first, the copy cost or the compute win?
2. **Make it self-consistent.** Right now consumption is a fixed background sink
   evaluated at the inflow PO2. Iterate: recompute the field, re-evaluate `M(PO2_i)`
   locally per point, and repeat until it converges. How many iterations?
3. **Tile-size sweep.** Change `THREADS_PER_BLOCK` (32, 64, 128, 256) in
   `kernels.cu` and observe the effect on kernel time; explain it with occupancy.
4. **Add a second metric.** Report the *fraction of tissue within 5 mmHg of the
   inflow PO2* (over-oxygenated regions) alongside the hypoxic fraction.
5. **Sketch the FMM.** Read THEORY.md §GPU-mapping, then describe how you would
   group distant sources into a single multipole to turn the O(N²) sum into
   O(N log N).

## Limitations & honesty

- **Reduced-scope teaching model.** This is a *linear superposition* of fixed
  source strengths, not the fully-coupled nonlinear Secomb solve (which solves for
  the `q_j` so that each source releases exactly the O2 the tissue draws). The
  consumption is applied as a uniform background sink, not point-by-point.
- **Lumped constants.** Diffusivity and solubility are folded into one constant `K`,
  and the source-strength scale is a chosen number — the values are order-of-
  magnitude physiological, **not** tuned to any specific tissue.
- **Direct O(N²) sum.** We compute the honest all-pairs sum, not the fast multipole
  method a production solver would use. That is deliberate: it is the baseline the
  FMM accelerates, and it makes the GPU pattern legible.
- **Synthetic data, no clinical validity.** The committed sample is synthetic and
  labelled as such. Nothing here may be used for diagnosis or treatment.
