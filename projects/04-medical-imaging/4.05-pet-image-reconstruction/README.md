# 4.5 — PET Image Reconstruction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.5`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Reconstruct a 2-D PET tracer-concentration image from noisy coincidence **counts**
(a sinogram) using **Maximum-Likelihood Expectation-Maximization (MLEM)** — the
iterative algorithm that (with its ordered-subsets accelerator OS-EM) dominates
clinical PET. Unlike CT's one-shot filtered backprojection (project 4.01), MLEM
respects the **Poisson statistics** of photon counting and improves the image over
many iterations. Each iteration is a **forward projection** and a **back
projection**, and both are per-output **gathers** — one GPU thread per line-of-
response (LOR) for the forward pass, one thread per pixel for the back pass — with
**no atomics**, so the result is deterministic. This project is the iterative
sibling of the CT flagship, reusing the same clean parallel-beam geometry.

## What this computes & why the GPU helps

PET detects coincident 511 keV gamma pairs; each pair defines a line (an LOR) that
the annihilation lay on. Stacked by angle, the per-LOR counts form a sinogram `y`.
MLEM finds the image `x` whose forward projection best explains `y` under the
Poisson likelihood, via the multiplicative update (Shepp–Vardi 1982):

```
x_j <- (x_j / s_j) · Σ_i A_ij · ( y_i / (A x)_i )
```

where `A` is the system matrix (pixel → LOR), `(A x)_i` is the forward projection,
and `s_j = Σ_i A_ij` is the pixel sensitivity. **Every iteration needs a full
forward projection and a full back projection.** Clinical scanners have ~10⁹
sinogram elements and run tens of iterations → billions of ray/pixel touches per
reconstruction, which is why production PET reconstruction is GPU-accelerated.

**The parallel bottleneck** is the projection pair. We give each **LOR** its own
thread in the forward projection and each **pixel** its own thread in the back
projection / update — both are gathers of independent outputs, so no atomics are
needed and the reduction order is fixed (deterministic stdout).

## The algorithm in brief

- **Sensitivity** `s = Aᵀ1` (back-project a sinogram of ones), computed once.
- **Per iteration:** forward project `ŷ = A x`; form the ratio `r = y / ŷ`;
  back-project `c = Aᵀ r`; multiplicative update `x ← x · c / s`.
- **Projector pair:** pixel-driven parallel-beam with **linear detector
  interpolation**; forward and back use the *same* split weights so `Aᵀ` is the
  exact transpose of `A` (what MLEM needs to converge).

The catalog also lists OS-EM, RAMLA, MAP-EM with Gibbs priors, PSF modelling,
TOF-PET, list-mode ML-EM, and PET/MRI joint reconstruction — described in
[THEORY.md](THEORY.md) under "Where this sits in the real world". This project
implements the clean **MLEM** core; OS-EM is a one-line change (left as an
exercise).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/pet-image-reconstruction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/pet-image-reconstruction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\pet-image-reconstruction.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, reconstructs the committed sinogram with MLEM on both
CPU and GPU, prints deterministic image samples, shows the GPU-vs-CPU agreement
check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/sinogram_sample.txt` — a **synthetic** noisy
  PET sinogram forward-projected from a known two-disc emission phantom (~4 KB), so
  the demo runs offline and the reconstruction is interpretable.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print pointers to real
  (non-clinical) PET sinograms (PETRIC/SIRF) — they never bypass credentials.
- **Provenance & license & file format:** see [data/README.md](data/README.md).

Catalog dataset notes: OpenNEURO PET datasets (https://openneuro.org/); TCIA PET
collections (https://www.cancerimagingarchive.net/); PETRIC challenge datasets
(https://github.com/SyneRBI/PETRIC); Siemens mMR phantom datasets (via STIR/SIRF).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
4.5 -- PET Image Reconstruction (MLEM)
MLEM: 30 iterations, 30 angles x 45 detectors -> 32x32 image
center pixel activity = 10.7865
peak activity = 24.6882 at (px,py)=(21,20)
total reconstructed activity = 3328.4667
central row profile (8 samples): 0.0902 0.3062 11.6823 11.1942 13.2607 12.9219 0.2768 0.0290
RESULT: PASS (GPU matches CPU within tol=1.0e-03)
```

The program reconstructs on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and asserts they agree within `1e-3`. The
CPU (pixel-driven scatter) and GPU (LOR-parallel gather) sum the same terms in a
different order, so they differ only by float rounding compounded over iterations
(`max_abs_err ≈ 6e-5` on the sample). The reconstruction recovers the phantom: the
central disc is bright and the peak sits on the small off-center hot spot.

## Code tour

Read in this order:

1. [`src/pet_geometry.h`](src/pet_geometry.h) — the shared `__host__ __device__`
   projection geometry used by **both** the CPU and GPU (the parity idiom).
2. [`src/main.cu`](src/main.cu) — loads counts, builds the sensitivity image, runs
   CPU + GPU MLEM, verifies, reports.
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`.cpp`](src/reference_cpu.cpp) —
   the data model + the trusted serial MLEM baseline.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
5. [`src/kernels.cu`](src/kernels.cu) — the forward/ratio/update kernels + the
   on-device MLEM loop.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **STIR** (<https://github.com/SyneRBI/STIR>) — the reference C++ tomographic
  reconstruction library: OS-EM, TOF, scatter, GPU projectors via parallelproj.
  Study its `ProjectorByBin` abstraction to see a production system-matrix API.
- **SIRF** (<https://github.com/SyneRBI/SIRF>) — Python/MATLAB framework wrapping
  STIR + Gadgetron for joint PET/MR; the source of the openly usable phantom data.
- **parallelproj** (<https://github.com/gschramm/parallelproj>) — clean CUDA/OpenCL
  PET projectors (Joseph/Siddon); the model for a *fast* forward/back projector.
- **CASToR** (<https://castor-project.org/>) — multi-threaded/GPU PET/SPECT
  reconstruction with a wide algorithm menu (OSEM, MAP, list-mode).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Projection gather** (docs/PATTERNS.md, the 4.01 pattern) applied twice per MLEM
iteration: one thread per **LOR** for the forward projection, one thread per
**pixel** for the back-projection/update — both independent outputs, so **no
atomics** and deterministic stdout. The per-element geometry lives in a shared
`__host__ __device__` header so the CPU reference and GPU kernels compute identical
math. (The catalog also mentions cuBLAS for correction factors, warp-level scatter
reduction, and CUDA streams — production accelerations, described in THEORY and
left as exercises; the teaching core here is the deterministic projector pair.)

## Exercises

1. **OS-EM.** Split the `K` angles into `S` ordered subsets and update after each
   subset (`S`× fewer iterations for similar image quality). Only the LOR loop
   changes — the sensitivity becomes per-subset.
2. **Attenuation & sensitivity.** Multiply the forward model by a per-LOR
   attenuation factor (from a CT μ-map). How does `s_j` change?
3. **A Siddon projector.** Replace the O(N²)-per-LOR pixel sweep with a proper ray
   tracer that visits only the pixels on each LOR — the production speed-up.
4. **A MAP/penalized-likelihood prior.** Add a quadratic (Gibbs) smoothness penalty
   to the update (one-step-late MAP-EM) and watch noise drop.
5. **FP64 vs FP32.** Switch the accumulators/state to `double` and measure the
   effect on the CPU/GPU agreement and on convergence.
6. **Interfile loader.** Write a converter from a real PETRIC/SIRF Interfile
   sinogram into this project's text format and reconstruct real phantom data.

## Limitations & honesty

- **2-D parallel-beam** geometry only. Real PET is a 3-D ring with fan/oblique LORs,
  detector-pair sensitivity, and time-of-flight — described in THEORY, not coded.
- **No physics corrections.** Attenuation, scatter, and randoms are omitted; the
  forward model is a plain line-integral projector.
- **Synthetic data**, labeled synthetic everywhere. Reconstructed values are
  arbitrary activity units, not calibrated Bq/mL or SUV.
- The projector is **pixel-driven and O(N²) per LOR** for clarity (a Siddon ray
  tracer is the production choice). On the tiny sample the GPU is launch-bound and
  can be slower than the CPU — an honest, documented teaching artifact.
- **Not for clinical use.** This is study material, not a diagnostic tool.
