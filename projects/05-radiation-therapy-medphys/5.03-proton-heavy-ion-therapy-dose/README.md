# 5.3 — Proton & Heavy-Ion Therapy Dose

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.3`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Proton and carbon-ion beams have a superpower that photon (X-ray) beams lack: they
deposit most of their energy in a sharp **Bragg peak** at a depth set by the beam's
energy, then stop — depositing **zero** dose beyond that depth. That lets a
treatment concentrate dose on a tumour while sparing healthy tissue behind it. This
project builds a small **analytic pencil-beam dose engine**: it takes a list of
proton *spots* (each a thin beam with a lateral position, a range, and a weight),
and computes the 3-D dose they deposit in a voxel grid. The GPU gives **one thread
per voxel**; that thread sums the contribution of every spot. We verify the GPU
volume against a plain-C++ CPU reference and recover the Bragg-peak depth from the
result. Everything is a **reduced teaching model in arbitrary dose units** — never
a clinical calculation.

## What this computes & why the GPU helps

Proton and carbon-ion beams deposit dose with a sharp Bragg peak distal to the
target, enabling sparing of surrounding normal tissue. Analytical dose engines
(pencil-beam algorithm, PBA) convolve pencil-beam kernels with CT stopping-power
maps; the GPU parallelizes the **per-spot convolution** across the ~10⁴ spots in a
plan, reducing a full plan from minutes to seconds. Full Monte Carlo (FRED, TOPAS,
GATE) simulates hadronic physics including nuclear fragmentation (dominant for
carbon ions), requiring the GPU for clinical throughput. Range uncertainty (from
the CT Hounsfield-unit → stopping-power conversion) is managed by robust
optimization over 3 mm / 3.5 % scenarios, multiplying the GPU compute requirement.

**The parallel bottleneck:** the dose at every voxel is an independent sum over all
spots, `dose(voxel) = Σ_spots weight · IDD(depth; range) · Lateral(offset; σ(depth))`.
With ~10⁷ voxels and ~10⁴ spots that is ~10¹¹ independent kernel evaluations — the
step that dominates the runtime and the one we parallelize. Because different voxels
never write the same memory, this is a pure **gather** (no atomics, no contention):
each voxel's thread accumulates into a private register and writes once.

## The algorithm in brief

- **Pencil-beam algorithm (PBA):** factor each spot's dose into a **depth term**
  (integral depth-dose with the Bragg peak) times a **lateral term** (a 2-D
  Gaussian that widens with depth from multiple Coulomb scattering).
- **Analytical Bragg-peak model:** a smooth Bortfeld-style surrogate expressed
  through the residual range `u = R − z` — near-flat entrance plateau, sharp peak as
  `u → 0`, hard zero for `u < 0`.
- **Superposition:** sum all spots per voxel (a spot map ⊛ pencil-beam kernel).
- **GPU mapping:** one thread per voxel; spots staged in **constant memory**; a
  grid-stride loop covers any voxel count.
- **Verification:** compare the GPU volume to a serial CPU reference (same shared
  formula), and check the recovered Bragg-peak depth.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation, including LET/RBE and nuclear fragmentation (why carbon ions need Monte
Carlo), which this teaching version deliberately omits.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/proton-heavy-ion-therapy-dose.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/proton-heavy-ion-therapy-dose.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\proton-heavy-ion-therapy-dose.sln /p:Configuration=Release /p:Platform=x64
```

`Debug|x64` also builds cleanly (device debug `-G` + lineinfo for Nsight).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (uses the optional CMake build)
```

The demo builds if needed, runs on `data/sample/proton_plan_sample.txt`, prints the
integral depth-dose (Bragg) curve, shows the GPU-vs-CPU agreement check, and prints
a timing line. Run the executable directly with no arguments to use the identical
built-in synthetic plan.

## Data

- **Sample (committed):** `data/sample/proton_plan_sample.txt` — a tiny **synthetic**
  proton plan (one 12 cm spot) so the demo runs with zero downloads.
- **Full dataset:** none required. `scripts/download_data.ps1` / `.sh` print pointers
  to TOPAS/GATE benchmark beams, the POPI model, and TCIA collections, and never
  bypass any registration/data-use agreement.
- **Provenance, format & license:** see [data/README.md](data/README.md).

Dose values are **arbitrary teaching units** normalised to the Bragg peak — not
calibrated Gray.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): an
integral depth-dose table with a low entrance plateau (~0.27 of peak), a sharp
**Bragg peak at 11.75 cm** (half a voxel proximal to the 12 cm range), and a hard
zero beyond it — then `RESULT: PASS`. The program computes the dose on both the
**GPU** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and
asserts they agree within `1.0e-4` in absolute dose units *and* that the Bragg-peak
bin matches. That agreement is the correctness guarantee (see THEORY §7 for why the
tolerance is small but not bit-exact).

## Code tour

Read in this order:

1. [`src/proton_physics.h`](src/proton_physics.h) — the shared `__host__ __device__`
   physics core (Spot/Grid/BeamModel + `dose_from_spot`). **Start here** — it is the
   one formula both the CPU and GPU evaluate.
2. [`src/main.cu`](src/main.cu) — loads the plan, runs CPU + GPU, verifies, and
   prints the deterministic Bragg curve.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the "one thread per
   voxel, gather over spots, constant-memory spot list" idea.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel and its host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the plan loader and the trusted
   serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **FRED** (<https://www.fredonline.eu/>) — GPU fast Monte Carlo for ions,
  clinical-grade, DICOM-RT input. Study how a *full MC* handles the physics our
  analytic model approximates (nuclear halo, fragmentation).
- **MOQUI** (<https://github.com/mghro/moquimc>) — open GPU proton MC (MGH) for quick
  dose recalculation. A readable modern GPU MC codebase.
- **OpenTOPAS** (<https://github.com/OpenTOPAS/OpenTOPAS>) — open Geant4-based proton
  MC; the reference for *what the ground truth is* when commissioning an engine.
- **matRad** (<https://github.com/e0404/matRad>) — analytic proton dose engine with
  GPU-parallel spot convolution. **The closest sibling to this project** — read its
  pencil-beam kernel and spot-weight optimizer.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent per-voxel jobs · gather over a constant-memory spot list.** One thread
owns one voxel and loops over all spots, accumulating into a register (no atomics —
distinct voxels never collide). The spot list lives in `__constant__` memory (read
by every thread, never written → broadcast cache), the same read-only-shared-data
trick as flagship 1.12. A grid-stride loop makes a fixed grid cover any voxel count.
Contrast this **analytic gather** with the Monte-Carlo **scatter + atomics** engine
in [5.01](../5.01-monte-carlo-dose-calculation/) — the two halves of GPU dose
calculation. (The catalog also lists cuFFT for k-space convolution and cuRAND/texture
memory for the MC and CT-lookup variants; this teaching version does the convolution
directly in real space so the per-spot arithmetic is fully visible — see THEORY §8.)

## Exercises

1. **Build a spread-out Bragg peak (SOBP).** Run
   `python scripts/make_synthetic.py --ranges 8 9 10 11 12` and feed the file to the
   exe. Observe the flat-topped plateau. Then tune the *weights* (edit the script) so
   the plateau is genuinely flat — this is the core of proton plan optimization.
2. **Lateral profile.** Add a second print that extracts a lateral dose profile at
   the Bragg-peak depth and confirm its width matches `σ(depth)` from
   `proton_physics.h`.
3. **Range shifter.** Add a constant offset to every spot's `range` (a "range
   shifter" slab) and watch the whole curve translate in depth.
4. **FP64 vs FP32.** Template `dose_from_spot` on the float type and compare the
   GPU/CPU `max_abs_err` in single vs double precision. Where does it change?
5. **Tile the spots.** For >2048 spots, stream the spot list from global memory in
   shared-memory tiles instead of constant memory; measure the effect on runtime.

## Limitations & honesty

- **Reduced teaching model.** The depth term is a *regularised* Bortfeld surrogate,
  not the exact parabolic-cylinder solution; the lateral term is a single Gaussian
  (real beams need a double-Gaussian "nuclear halo"). There is **no** CT stopping-power
  map, **no** LET/RBE weighting, and **no** nuclear fragmentation (which dominates for
  carbon ions and forces full Monte Carlo). THEORY §8 says what production tools do
  differently.
- **Arbitrary units.** Dose is normalised to the Bragg peak; it is **not** Gray and
  carries no clinical meaning.
- **Synthetic data.** The committed plan is made up and labeled synthetic everywhere.
- **Not for clinical use.** No output here may inform diagnosis or treatment
  (CLAUDE.md §1, §8).
