# 5.9 — Gamma-Index Dose Comparison

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🟢 Beginner · Established** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.9`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Before a modulated radiotherapy plan (IMRT / VMAT) is delivered to a patient, the
physicist checks that the dose the machine *actually* produces matches the dose
the planning computer *predicted*. The **gamma index** is the standard way to
score that match: at every point of the reference (planned) dose map it searches
the evaluated (measured) dose map for the nearest point that agrees, blending a
**dose-difference** tolerance and a **distance-to-agreement (DTA)** tolerance into
one number γ. Points with γ ≤ 1 pass; the fraction that pass is the **gamma
pass-rate** a clinic signs off on. This project computes a 2-D gamma map two ways
— a plain CPU reference and a CUDA kernel with **one thread per reference voxel** —
and shows they agree *exactly*, on a small synthetic dose pair engineered to have
a known, localized failure region.

## What this computes & why the GPU helps

The gamma index (γ) at each reference point searches for the minimum normalized
Euclidean distance in combined dose-difference / distance-to-agreement (DTA)
space over all evaluated points:
γ(r_ref) = min over r of √[(Δd/Δd_crit)² + (Δr/DTA_crit)²]. For 3-D clinical
distributions the exhaustive search over an N³ evaluation grid from each of N³
reference points is O(N⁶) naively, reduced to O(N³·K) by limiting the search
radius. This is critical for patient-specific IMRT/VMAT pre-treatment
verification, where a physicist needs an answer in seconds, not minutes.

**The parallel bottleneck:** the gamma at each reference voxel is an
**independent search** over a neighborhood of evaluated voxels — voxel *i*'s
answer never depends on voxel *j*'s. That embarrassing parallelism is exactly
what the GPU exploits: we launch one thread per reference voxel, and every thread
does its own local min-search concurrently. The dominating cost is the
`N_voxels × K_window` inner search, which the GPU spreads across thousands of
threads.

## The algorithm in brief

- **Global gamma, distance-limited exhaustive search** — for each reference
  voxel, scan evaluated voxels inside a physical search window and keep the
  minimum √(dose-term² + distance-term²).
- **Per-thread minimum reduction in a register** — no atomics, no shared memory;
  each thread owns one output voxel (contrast the Monte-Carlo flagship 5.01,
  which *does* need atomics).
- **Global gamma pass-rate statistics** — integer counts of analyzed vs. passing
  voxels above a low-dose threshold.
- **Shared `__host__ __device__` core** — the per-pair math lives in one header
  ([`src/gamma_core.h`](src/gamma_core.h)) so CPU and GPU are bit-identical.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/gamma-index-dose-comparison.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/gamma-index-dose-comparison.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\gamma-index-dose-comparison.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the gamma pass-rate and
a small slice of the gamma map, shows the GPU-vs-CPU agreement check, and prints a
timing line.

## Data

- **Sample (committed):** `data/sample/dose_pair.txt` — a tiny, **synthetic**
  32×32 dose pair (a Gaussian "dose hill" reference + an evaluated map with a
  +1.5% global bias and a deliberate central hot spot). Offline, zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (documented; prints
  registration instructions for credentialed clinical sets, never bypasses them).
- **Regenerate / resize the sample:** `python scripts/make_synthetic.py --n 64`.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: AAPM TG-218 patient-specific IMRT QA reference data;
plan+measurement DICOM pairs from departmental QA systems; IROC-Houston phantom
dose datasets; linac EPID measurement datasets.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): a
99.7% pass-rate, three failing voxels in the injected hot spot (γ_max = 1.141),
and a passing background (center-row γ ≈ 0.5). The program computes the gamma map
on both the **GPU** (`src/kernels.cu`) and a **CPU reference**
(`src/reference_cpu.cpp`) and asserts they agree to within `1e-6` — in practice
the observed `max_abs_err` is **exactly 0**, because both call the identical
`gamma_core.h` math and reduce with an exact floating-point `min` (see THEORY §6).

## Code tour

Read in this order:

1. [`src/gamma_core.h`](src/gamma_core.h) — the ONE per-pair gamma formula,
   shared `__host__ __device__` by CPU and GPU (start here — it is the whole idea).
2. [`src/dose_problem.h`](src/dose_problem.h) — the `DoseProblem` struct (two dose
   maps + grid geometry + criteria).
3. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
5. [`src/kernels.cu`](src/kernels.cu) — the per-reference-voxel kernel and host wrapper.
6. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline.
7. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **[PyMedPhys](https://github.com/pymedphys/pymedphys)** — open-source Python
  gamma index + DICOM dose tools. Study its `gamma_shell` for how a production
  gamma handles interpolation of the evaluated grid and 1D/2D/3D generality.
- **[Plastimatch](https://plastimatch.org/)** — a C++ (with GPU) `gamma`
  command-line tool; a good model for DICOM-RTDOSE I/O and resampling two dose
  grids onto a common frame before comparison.
- **UCSD GPU gamma (Gu et al., 2011, [PMID 21317484](https://pubmed.ncbi.nlm.nih.gov/21317484/))**
  — the paper that popularized the one-thread-per-reference-point GPU gamma with a
  geometric search-space reduction; read it for the distance-limiting argument.
- **[OpenGATE](https://github.com/OpenGATE/opengate)** — Monte-Carlo dose engine
  whose utilities include dose-comparison tooling; useful for generating
  realistic reference/evaluated pairs.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Gather + per-thread minimum reduction** (PATTERNS.md §1; closest flagship
`4.01` CT backprojection, which likewise gathers per output pixel). One CUDA
thread per reference voxel; each thread reads a small window of the evaluated
dose map from global memory (routed through the read-only cache), reduces to a
minimum in a **register**, and writes one gamma value. No atomics or shared
memory are needed because each thread owns exactly one output voxel — and because
`min` is exact and associative in floating point, the GPU result is bit-identical
to the CPU (unlike a reordered `atomicAdd` sum; see PATTERNS.md §3).

## Exercises

1. **Shared-memory tiling.** The catalog's suggested optimization: cooperatively
   load the evaluated tile (plus a halo the width of the search radius) into
   shared memory once per block, then have all threads in the block search that
   tile. Measure the speedup vs. the naive gather here. (Hint: mirror the tiling
   in flagship `6.04`.)
2. **Local vs. global gamma.** This code normalizes the dose difference to the
   *global* maximum dose. Switch to *local* normalization (Δd_crit = P% of the
   local reference dose) and observe how the pass-rate changes in low-dose regions.
3. **Sub-voxel DTA.** Real gamma tools bilinearly interpolate the evaluated grid
   so the DTA is not quantized to the voxel pitch. Add interpolation to the search
   and check that γ_max drops.
4. **Extend to 3-D.** Add a `depth` and a third search loop — the kernel becomes a
   3-D grid of 3-D blocks. Confirm CPU==GPU still holds exactly.
5. **FP32 vs FP64 in the reduction.** The per-pair math here is `double`. Try
   `float` throughout and quantify the change in γ (and whether CPU==GPU survives).

## Limitations & honesty

- **Synthetic data.** The committed dose pair is generated, not clinical — a
  Gaussian hill with a hand-placed hot spot. It is labeled synthetic everywhere.
  No output here has any clinical validity.
- **2-D, reduced scope.** The catalog frames a 3-D clinical gamma; this teaching
  version is 2-D (which matches a film/EPID/2-D-array QA measurement) to keep the
  sample tiny and the code legible. The 3-D extension is one more loop (Exercise 4).
- **Same grid, no resampling.** Both maps are assumed already on the same grid.
  Production tools resample two DICOM dose grids onto a common frame first.
- **Voxel-quantized DTA, global normalization only.** No sub-voxel interpolation
  (Exercise 3) and only global dose-difference normalization (Exercise 2).
- **Timing is a teaching artifact, not a benchmark.** On a 32×32 grid the GPU is
  launch/copy-bound and *slower* than the CPU; the GPU's advantage appears only at
  clinical grid sizes. See THEORY §7.
