# 4.2 — Iterative / Model-Based CT Reconstruction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.2`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

This project reconstructs a CT image from its X-ray sinogram **iteratively**,
using **SIRT** (Simultaneous Iterative Reconstruction Technique) with an
edge-preserving **total-variation (TV)** prior. Instead of inverting the data in
one analytic pass (that is Project 4.01, Filtered BackProjection), it repeatedly
*simulates* a scan of its current image guess, compares to the real measurement,
and corrects the guess — folding in a prior that suppresses noise. That extra
cost (dozens of forward/backprojection passes) is exactly why a GPU matters, and
it is what lets model-based reconstruction produce a cleaner image at lower
radiation dose. The whole reconstruction runs on the GPU and is checked against a
serial CPU reference and against the known synthetic phantom.

## What this computes & why the GPU helps

Instead of a single analytical inversion, iterative methods repeatedly forward-project a current volume estimate, compare to measured sinogram data, then backproject the residual with statistical weighting. Penalized weighted least squares (PWLS) with total-variation (TV) or dictionary priors reduces noise by 30–50% at matched dose compared with FBP. Each outer iteration performs one full forward-projection and one backprojection — exactly the same GPU kernel bottleneck as FBP but repeated 20–200 times, making GPU mandatory for clinical throughput. ADMM decouples the data-fidelity and regularization sub-problems, enabling efficient GPU-friendly matrix-vector operations. Statistical models (Poisson likelihood for photon counts) can be incorporated for dose-optimal reconstruction.

**The parallel bottleneck:** each iteration is one **forward projection**
(`A x`: simulate the scan of the current image) plus one **backprojection**
(`Aᵀ r`: smear the residual back into image space). These are the same
gather/scatter operations that make FBP GPU-bound — but here they repeat 20–200
times. We parallelize the forward projection as *one thread per detector ray* and
the backprojection as *one thread per image pixel*, keeping the whole image
resident on the device across all iterations.

## The algorithm in brief

- **Forward projection `A`** — voxel-driven, linear-interpolation line integrals.
- **SIRT update** — `x ← max(0, x + λ·C·Aᵀ·R·(b − A x))` with row/column
  normalization diagonals `R`, `C` and relaxation `λ`, plus non-negativity.
- **Total-variation (TV) step** — one edge-preserving smoothing step per
  iteration (the "model-based" prior that beats FBP at low dose).
- Related methods (SART, OS-EM, PWLS, ADMM, Chambolle–Pock, plug-and-play with a
  learned denoiser) are described in [THEORY.md](THEORY.md) §7.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/iterative-model-based-ct-reconstruction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/iterative-model-based-ct-reconstruction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\iterative-model-based-ct-reconstruction.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: 2016 AAPM Low-Dose CT Grand Challenge (https://www.aapm.org/grandchallenge/lowdosect/); Mayo Clinic Low-Dose CT dataset (available via TCIA); LIDC-IDRI via TCIA (https://www.cancerimagingarchive.net/).

## Expected output

Success looks like `demo/expected_output.txt`:

```
4.2 -- Iterative / Model-Based CT Reconstruction
SIRT+TV: 48 angles x 67 detectors -> 48x48 image, 60 iterations
lambda = 1.500  tv_weight = 0.0100
center pixel value = 1.0091
max reconstructed value = 1.6538 at (px,py)=(24,33)
central row profile (8 samples): -0.0074 0.2525 1.1263 1.0324 1.0009 0.9493 0.7014 0.0141
reconstruction RMSE vs truth = 0.1043
RESULT: PASS (GPU matches CPU within tol=2.0e-03)
```

The program reconstructs on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and asserts they agree within `2·10⁻³`
(FMA drift over many iterations makes exact equality impossible — see
[THEORY.md](THEORY.md) §6). The `center pixel value ≈ 1.0` recovers the phantom's
body density and `RMSE vs truth ≈ 0.10` confirms the science, not just CPU==GPU
agreement. Timing goes to **stderr** (not diffed, since it varies).

## Code tour

Read in this order:

1. [`src/ct_geometry.h`](src/ct_geometry.h) — the ONE shared `__host__ __device__`
   projection geometry used by both CPU and GPU (the HD-macro idiom).
2. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU SIRT, verifies, reports.
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`.cpp`](src/reference_cpu.cpp) —
   the trusted serial SIRT baseline (loader, projectors, TV step, the loop).
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
5. [`src/kernels.cu`](src/kernels.cu) — the four kernels and the `sirt_gpu` driver.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, I/O helpers.

## Prior art & further reading

ASTRA Toolbox (https://github.com/astra-toolbox/astra-toolbox) — GPU primitives, build iterative loops in Python/MATLAB; TIGRE (https://github.com/CERN/TIGRE) — includes OS-TV, SART, CGLS with GPU acceleration; ODL (Operator Discretization Library, https://github.com/odlgroup/odl) — Python framework wrapping ASTRA for variational reconstruction; LEAP (https://github.com/LLNL/LEAP) — LLNL GPU-accelerated CT reconstruction library with penalized-likelihood support.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Two adjoint operators in a loop** (PATTERNS.md §7, closest flagship `4.01`).
Custom CUDA kernels for voxel-driven forward/backprojection: the forward
projection is *one thread per detector ray* (each ray sums its own pixels — no
atomics, deterministic), the backprojection + SIRT update is *one thread per
pixel* (a per-pixel gather), and the TV step is a *stencil with ping-pong
buffers*. The image and all scratch buffers stay resident on the device across
all iterations; the host only orchestrates the per-iteration kernel launches.
Production stacks instead build the system matrix as a sparse operator (cuSPARSE)
or hybridize with cuFFT-based filtering — see [THEORY.md](THEORY.md) §4, §7.

## Exercises

1. **Turn off the prior.** Set `tv_weight` to `0` in the sample header (or run
   `make_synthetic.py --tv 0`) and re-capture. Watch the noise/streaks return —
   this is the whole argument for model-based reconstruction.
2. **Step-size sweep.** Vary `lambda` (`0.5`, `1.0`, `2.0`). Too small converges
   slowly; too large oscillates. Add a per-iteration residual print to stderr and
   plot convergence.
3. **Ordered subsets (OS-SART).** Update from a subset of angles each iteration
   instead of all of them, cycling through subsets. This is the standard
   acceleration; measure how many fewer full passes you need.
4. **Faster forward projector.** The current forward kernel is `O(N²)` per ray.
   Replace it with a ray-marching DDA that visits only pixels *on* the ray, and
   compare timing and accuracy.
5. **Cache the trig tables.** Move `cosv/sinv` into constant memory (or use
   `__ldg`) and measure the effect on the bandwidth-bound projection kernels.

## Limitations & honesty

- **Synthetic data only.** The committed sample is a synthetic disc phantom with
  added noise (`scripts/make_synthetic.py`), clearly labeled synthetic. It is not
  a patient scan and nothing here is validated for any clinical use.
- **Reduced scope.** 2-D parallel-beam geometry (not 3-D cone-beam/helical); a
  matrix-free voxel-driven projector (not a calibrated system matrix); plain
  least-squares data fidelity (not the Poisson statistical weight `W`); and one
  explicit TV-descent step (not a full ADMM/primal-dual solver). Each omission is
  described in [THEORY.md](THEORY.md) §7.
- **Teaching-sized.** 48×48 image, 60 iterations — small enough that the GPU's
  advantage is modest and launch overhead is visible; the gap widens with image
  size, view count, and iteration budget. Timing is a teaching artifact, never a
  benchmark claim.
