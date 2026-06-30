# 2.31 — Cryo-EM Tilt-Series Alignment & Tomogram Reconstruction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.31`
>
> _Educational only — not for clinical use (see CLAUDE.md §8). This is a
> **reduced-scope, 2-D teaching version**; the full 3-D pipeline is described in
> [THEORY.md](THEORY.md) §7._

## Summary

Cryo electron tomography (cryo-ET) tilts a frozen specimen through a range of
angles, records a 2-D projection at each tilt, and reconstructs the specimen's 3-D
density. This project builds the two computational hearts of that pipeline on a
single 2-D slice: **tilt-series alignment** (recover the per-projection drift by
cross-correlation) and **weighted back-projection** (the inverse Radon transform,
with a ramp filter applied via **cuFFT** and a custom per-pixel CUDA gather
kernel). It runs on a tiny synthetic disc phantom with a *known* injected drift,
so you can watch the alignment recover the ground truth and the reconstruction
recover the phantom — and every GPU result is checked against a plain CPU
reference.

## What this computes & why the GPU helps

Cryo-ET reconstruction has three GPU-parallelizable stages: frame/beam-induced
motion correction, tilt-series alignment (fiducial or fiducial-free), and tomogram
reconstruction (weighted back-projection, or iterative SART as in the ASTRA
Toolbox). Here we implement alignment + WBP for one slice. WBP uses a **GPU FFT**
for the ramp filter and a **back-projection gather** for the inverse transform.
Cryo-ET is fundamentally limited by the **missing wedge** (only ~±60° is sampled),
which streaks the result; deep-learning tools (IsoNet) correct it post hoc.

**The parallel bottleneck:** **back-projection** dominates — `O(img² · n_tilts)`
in 2-D and `O(vox³ · n_tilts)` in 3-D. Every output pixel/voxel independently
gathers one interpolated sample from every projection, so it maps perfectly onto
one GPU thread per pixel (a bandwidth-bound gather). The ramp filter is a batched
FFT, the other classic GPU workload — hence cuFFT.

## The algorithm in brief

- **Tilt-series alignment** — sequential (neighbor-to-neighbor) **cross-
  correlation**, accumulated outward from the least-tilted reference, recovers each
  projection's translational drift. (The coarse pass behind IMOD `tiltxcorr` /
  AreTomo2.)
- **Ramp filter (Ram-Lak)** — multiply each projection's spectrum by `|f|` with a
  Hann roll-off; done with **cuFFT** (R2C → ramp → C2R).
- **Weighted back-projection (WBP)** — for each pixel, sum the interpolated
  filtered value along its ray over all tilts.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)). Links **cuFFT**
(`cufft.lib`) for the ramp filter.

1. Open `build/cryo-em-tilt-series-alignment-tomogram-reconstruction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/cryo-em-tilt-series-alignment-tomogram-reconstruction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\cryo-em-tilt-series-alignment-tomogram-reconstruction.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (optional CMake build)
```

The demo builds if needed, runs on `data/sample/tilt_series_sample.txt`, prints
the recovered alignment shifts and reconstruction samples, shows the two GPU-vs-CPU
agreement checks, and prints timing.

## Data

- **Sample (committed):** `data/sample/tilt_series_sample.txt` — a tiny, offline,
  **synthetic** tilt series (19 projections over ±60° of a disc phantom, with a
  known per-projection drift) so the demo runs with zero downloads.
- **Full datasets:** `scripts/download_data.ps1` / `.sh` print where to get real
  cryo-ET data (they never bypass any registration).
- **Provenance & license:** see [data/README.md](data/README.md).

Real datasets: EMPIAR tilt-series archives (<https://www.ebi.ac.uk/empiar/>, e.g.
EMPIAR-10045 in-situ ribosome); EMDB subtomogram averages
(<https://www.ebi.ac.uk/emdb/>); the SHREC cryo-ET benchmark.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
`estimated shifts (bins)` line recovers the injected drift (`-3 -3 … +3 +3`) to
within ~1 bin; the center pixel reconstructs to ≈ 0.93 and the maximum lands near
the slice center (the bright central disc). `RESULT: PASS` means **both** checks
held: the GPU back-projection matched the CPU reference within `1e-3`, and the
cuFFT ramp matched the CPU DFT ramp on the interior within `5e-2`. The CPU
reference (`src/reference_cpu.cpp`) is the trusted baseline; agreement with it is
the correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — orchestrates align → ramp-filter → back-project,
   verifies, and reports (stdout deterministic, stderr timing).
2. [`src/reference_cpu.h`](src/reference_cpu.h) — the science, geometry, and CPU
   function contracts.
3. [`src/wbp_core.h`](src/wbp_core.h) — the **shared `__host__ __device__` math**
   (per-sample interpolation + ramp weight) used by both CPU and GPU.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the two thread maps.
5. [`src/kernels.cu`](src/kernels.cu) — the cuFFT ramp filter and the per-pixel
   back-projection kernel.
6. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline
   (alignment, explicit-DFT ramp, serial WBP).
7. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, I/O helpers.

## Prior art & further reading

- **IMOD** (<https://bio3d.colorado.edu/imod/>) — the standard tomographic suite;
  study `etomo`/`tiltxcorr` for fiducial-based tilt-series alignment.
- **ASTRA Toolbox** (<https://github.com/astra-toolbox/astra-toolbox>) — GPU CUDA
  WBP/SART reconstruction; study how it batches projection geometry.
- **AreTomo2** (<https://github.com/czimaginginstitute/AreTomo2>) — GPU fiducial-
  free alignment + motion correction; study its patch cross-correlation.
- **IsoNet** (<https://github.com/IsoNet-cryoET/IsoNet>) — GPU deep-learning
  missing-wedge correction; study what artifact it targets and why.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Per-output-pixel gather + cuFFT filter** (PATTERNS.md §1, exemplified by
flagship `4.01` CT backprojection): a custom CUDA WBP kernel (one thread per slice
pixel, gathering an interpolated sample from every tilt) preceded by a cuFFT-based
ramp filter (batched R2C → `|f|` ramp multiply → C2R). Plus a `__host__ __device__`
shared core (PATTERNS.md §2) so CPU and GPU run identical math, and deterministic
integer-bin alignment so stdout is byte-reproducible (PATTERNS.md §3). The full
production system additionally uses iterative SART (ASTRA), multi-GPU volumes, and
a CNN for missing-wedge correction — see [THEORY.md](THEORY.md) §7.

## Exercises

1. **Widen the angular range.** Regenerate with `--maxtilt 75 --step 4` and watch
   the streaking shrink: a smaller missing wedge = a cleaner reconstruction.
2. **Sub-pixel alignment.** The current alignment is integer-bin. Add a parabolic
   fit to the three cross-correlation values around the peak for sub-bin precision,
   and switch `apply_shifts` to linear interpolation.
3. **Texture-memory interpolation.** Bind `filtered` to a `cudaTextureObject_t` and
   let the hardware sampler do the linear interpolation; compare speed and code.
4. **A second precision.** Add an FP64 build of the back-projection kernel and
   measure the accuracy/throughput trade-off on this `sm_75`-class GPU.
5. **Iterative reconstruction.** Implement one SART iteration (forward-project the
   current estimate, back-project the residual) and compare its limited-angle
   artifacts to WBP after a few iterations.

## Limitations & honesty

- **2-D, reduced scope.** We reconstruct a single slice; a real tomogram is a 3-D
  stack of these. Frame/motion correction (step 1) is out of scope.
- **Translational, integer-precision alignment only.** Real alignment also solves
  rotation, magnification, and the tilt-axis position, and refines to sub-pixel
  with gold fiducials. High tilt foreshortens features, so our integer shifts can
  be off by ~1 bin — visible in the output and left as Exercise 2.
- **Synthetic data.** The phantom and its analytic projections are **synthetic and
  labeled synthetic**; nothing here is a real specimen or a clinical/research
  result. The missing wedge streaking is real and intentional.
- **Timing is a teaching artifact.** On this tiny sample the kernels are
  launch-bound, so GPU and CPU times are comparable; the GPU's advantage grows with
  slice size and tilt count (and across the many slices of a 3-D volume).
