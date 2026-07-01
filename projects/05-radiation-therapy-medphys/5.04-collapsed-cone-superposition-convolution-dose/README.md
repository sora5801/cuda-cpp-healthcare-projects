# 5.4 — Collapsed-Cone / Superposition-Convolution Dose

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🟢 Beginner · Established** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.4`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

This project is a small, heavily-commented **superposition-convolution (SC) photon
dose engine** — the algorithm family behind most commercial radiotherapy planning
systems (Eclipse AAA/AXB, RayStation, Pinnacle). It computes, on a 2-D density map
that stands in for a CT slice, how much radiation dose a photon beam deposits in
each voxel. It does this in the two stages every SC engine uses: (1) a **TERMA
ray-trace** that tracks how the beam attenuates with depth, and (2) a
**collapsed-cone convolution** that spreads the released energy outward along a
handful of discrete cone directions with density-scaled exponential kernels. The
GPU runs both stages; a plain-C++ reference runs the identical math so we can prove
the GPU is correct. Everything is a **reduced-scope 2-D teaching version** of the
full 3-D production algorithm — see *Limitations*.

## What this computes & why the GPU helps

Superposition-convolution dose computation convolves Monte-Carlo-derived photon
energy-deposition kernels (dose-spread arrays) with the **TERMA** (total energy
released per unit mass) computed from CT. **Collapsed-cone convolution (CCC)**
discretizes that kernel into angular cones and propagates dose along ray paths at
each angle. For a 512³ CT volume and ~400 cone directions, each cone sweep is a 1-D
scan along the CT — embarrassingly parallel across cones and voxels. GPU
parallelization across cone directions and voxel planes reduces a CCC plan from
~10 min to <10 s. This algorithm underlies most commercial photon dose engines.

**The parallel bottleneck:** the collapsed-cone superposition. Every source voxel
spreads its released energy along every cone direction — an O(voxels × cones ×
ray-length) nested loop that dominates the runtime and is trivially parallel
because each source voxel's spread is independent. We map **one GPU thread per
source voxel** (Stage 2); Stage 1 (TERMA) maps **one thread per beam column**.

## The algorithm in brief

- **TERMA ray-trace (Siddon / ray-voxel):** march the beam down each column,
  accumulating *radiological* path length (∫ρ·dl), and set TERMA = (μ/ρ)·Ψ with
  Ψ = Ψ₀·exp(−(μ/ρ)·∫ρ·dl) (Beer-Lambert with heterogeneity).
- **Collapsed-cone convolution (CCC):** collapse the dose-spread kernel onto a few
  cone directions; along each cone, dose transport is a 1-D exponential recurrence
  `carry ← carry·exp(−a·d_rad)`, depositing the attenuated fraction per step.
- **Heterogeneity correction via density scaling:** the step length is scaled by
  the local density, so the kernel reaches farther in lung and less in bone.
- **Superposition:** sum every source voxel's cone spread into the dose grid.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation, including the general oblique-ray Siddon tracer and the polyenergetic
kernel used in production.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/collapsed-cone-superposition-convolution-dose.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/collapsed-cone-superposition-convolution-dose.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\collapsed-cone-superposition-convolution-dose.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/phantom.txt`, prints the
central-axis depth-dose curve, shows the GPU-vs-CPU agreement check (exact integer
match), and prints a timing line.

## Data

- **Sample (committed):** `data/sample/phantom.txt` — a tiny, **synthetic** 16×16
  density phantom (water / lung / water / bone / water) so the demo runs offline
  with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print links + registration
  instructions for the reference benchmarks (they never bypass credentials).
- **Provenance, layout & license:** see [data/README.md](data/README.md).

Catalog dataset notes: AAPM TG-105 test cases (heterogeneous media dose
benchmarks); IROC lung phantom CT + dosimetry data; TCIA clinical photon planning
datasets; CIRS IMRT verification phantom data.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program computes the dose grid on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and asserts they agree:

- the **dose grid** matches **exactly** (0 integer mismatches — the fixed-point
  determinism trick, see below), and
- the **TERMA** matches within `1e-9` (double precision).

The headline `RESULT: PASS` line encodes that guarantee, and the central-axis
depth-dose curve visibly shows build-up, a lung dip, and a bone pile-up.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the phantom, runs CPU + GPU, verifies, reports.
2. [`src/ccc_physics.h`](src/ccc_physics.h) — the shared `__host__ __device__`
   physics (TERMA, cone recurrence, fixed-point quantization). **The heart of the
   project**: both CPU and GPU call these exact functions.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the two thread mappings.
4. [`src/kernels.cu`](src/kernels.cu) — the TERMA and CCC kernels + host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, I/O helpers.

## Prior art & further reading

- **matRad** (<https://github.com/e0404/matRad>) — an open MATLAB/Octave treatment
  planning toolkit with a photon pencil-beam + collapsed-cone dose engine. Read it
  to see how cone kernels and beamlet superposition are organized in a real planner.
- **Plastimatch** (<https://plastimatch.org/>) — C++/CUDA image-processing and dose
  components; study its ray-tracing and GPU patterns.
- **CERR** (<https://github.com/cerr/CERR>) — a research dose-calculation and plan
  analysis framework; useful for the data model around dose grids.
- **AAPM TG-105** (report at <https://www.aapm.org/pubs/reports/>) — the reference
  for benchmarking convolution/superposition and Monte-Carlo dose engines against
  heterogeneous-media test cases.

Study these to learn the production approach; **do not copy code wholesale** —
this project reimplements the ideas didactically and credits the sources (CLAUDE.md §2).

## CUDA pattern used here

Two independent, embarrassingly-parallel kernels (PATTERNS.md §1 — the "gather /
per-ray scan" and "parallel-assign + atomic-reduce" idioms combined):

- **Stage 1 — TERMA:** custom kernel, **one thread per beam column**, each marching
  a Siddon-style ray down its column (no atomics, disjoint writes).
- **Stage 2 — CCC:** **one thread per source voxel**, scattering its collapsed-cone
  dose into the grid with **`atomicAdd` on integer dose-units** so overlapping cone
  contributions sum deterministically (PATTERNS.md §3). Both kernels call the shared
  `__host__ __device__` physics (PATTERNS.md §2) so the GPU result is bit-identical
  to the CPU reference.

## Exercises

1. **More cones, smoother spread.** Extend `ccc_cone_dx/dy` and `n_cones` past 8 to
   a finer in-plane angular set (e.g. 16 directions). How does the depth-dose curve
   change? (Watch out: adding cones changes total deposited units.)
2. **Anisotropic kernel weights.** Real dose-spread kernels weight forward cones
   more than backward ones. Replace the uniform `cone_weight` with a per-cone weight
   and observe the forward-peaked deposition.
3. **Shared-memory density strip.** Stage 2 re-reads `rho` along each cone ray from
   global memory. Cache a tile of the density map in shared memory (see THEORY.md
   §GPU-mapping) and measure the effect on larger grids.
4. **A second precision.** Switch the dose accumulation to `float` atomics and show
   that the grid is no longer bit-reproducible run-to-run — then explain why the
   integer version is (PATTERNS.md §3).
5. **Oblique beams.** Generalize the TERMA tracer from vertical columns to an
   arbitrary beam angle using the full Siddon parametric intersection (THEORY.md
   derives it).

## Limitations & honesty

This is a **reduced-scope 2-D teaching model**, not a clinical dose engine:

- **2-D, not 3-D.** Production CCC runs on 512³ voxels with ~48–400 cones on a
  sphere; here it is a 2-D grid with 8 in-plane cones.
- **Monoenergetic, single analytic kernel.** Real engines use *polyenergetic*
  Monte-Carlo dose-spread arrays that vary with depth; we use one exponential cone
  kernel `a·exp(−a·r)`. The μ/ρ and kernel constants are illustrative, not
  calibrated to a real linac spectrum.
- **Axis-aligned beam.** The TERMA tracer marches straight down; oblique rays,
  divergence, and beam penumbra are omitted (described in THEORY.md).
- **Synthetic data.** The phantom is generated, clearly labeled synthetic, and
  produces **no clinically meaningful dose** — units are arbitrary.
- **Timing is a teaching artifact.** On this tiny grid the GPU is launch-bound and
  slower than the CPU; the GPU advantage appears only at clinical problem sizes.
