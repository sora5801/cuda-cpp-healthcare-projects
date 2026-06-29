# 4.01 — CT Reconstruction (Filtered Backprojection)

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟢 Beginner · Established** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.01`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Reconstruct a cross-sectional image from X-ray **projections** (a sinogram) using
**Filtered BackProjection (FBP)**: ramp-filter each projection, then "smear" each
filtered projection back across the image and sum over all angles. Backprojection
is a **per-pixel gather** — every output pixel independently samples one value
from each projection — which maps perfectly onto a 2-D GPU thread grid. This is a
third distinct GPU pattern (after `1.12`'s independent jobs and `3.01`'s
dependency wavefront), and the first flagship with a clear GPU speed-up.

## What this computes & why the GPU helps

FBP applies a ramp (Ram-Lak) filter to each sinogram row, then backprojects.
Backprojection dominates the cost: each of the `img²` pixels reads one
interpolated sample from each of `n_angles` projections. For clinical
3-D volumes (the **Feldkamp-Davis-Kress / FDK** cone-beam extension) this is
~10¹¹ voxel-projection pairs — intractable on a CPU, fast and bandwidth-bound on
a GPU (texture units even do the interpolation for free). Here we do the clearest
case: 2-D parallel-beam FBP.

**The parallel bottleneck** is the backprojection gather; we give each output
pixel its own thread, looping over all projection angles.

## The algorithm in brief

- **Ramp filter** (Ram-Lak): convolve each projection with the discrete ramp
  kernel (production uses an FFT).
- **Backproject:** `image(x,y) = (π/n_angles)·Σ_k filtered(θ_k, x·cosθ_k + y·sinθ_k)`
  with linear interpolation in the detector.

See [THEORY.md](THEORY.md) for the full derivation (and how FDK extends it to cone-beam 3-D).

## Build

Requires **Visual Studio 2026** (v145) + **CUDA 13.3** ([docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/ct-reconstruction-filtered-backprojection.sln`.
2. **`Release|x64`** → **Build** → `build/x64/Release/ct-reconstruction-filtered-backprojection.exe`.

CLI: `msbuild build\ct-reconstruction-filtered-backprojection.sln /p:Configuration=Release /p:Platform=x64`

## Run the demo

```powershell
./demo/run_demo.ps1
```

Reconstructs the committed sinogram, prints image samples, and verifies GPU == CPU.

## Data

- **Sample (committed):** `data/sample/sinogram_sample.txt` — an **analytic**
  sinogram of a disc phantom (deterministic).
- **Full data:** Shepp-Logan phantom or real DICOM CT (TCIA) — see
  `scripts/download_data.ps1` and [data/README.md](data/README.md).
- Larger synthetic: `python scripts/make_synthetic.py --angles 360 --det 367 --img 256`.

## Expected output

`demo/expected_output.txt` holds the deterministic stdout (center pixel, max,
central-row profile). The GPU backprojection (`src/kernels.cu`) and CPU reference
(`src/reference_cpu.cpp`) use **host-precomputed trig**, so they agree to
`~1e-5` (well within the `1e-3` tolerance). On the sample the center pixel
reconstructs to ≈ 1.0 — the main disc's density.

## Code tour

1. [`src/main.cu`](src/main.cu) — load, ramp-filter, CPU + GPU backproject, verify, print.
2. [`src/reference_cpu.h`](src/reference_cpu.h) — geometry (`CTProblem`), filter & backprojection prototypes.
3. [`src/kernels.cuh`](src/kernels.cuh) — the per-pixel kernel interface.
4. [`src/kernels.cu`](src/kernels.cu) — the 2-D backprojection kernel + host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — loader, ramp filter, serial backprojection.

## Prior art & further reading

- **RTK** (<https://github.com/RTKConsortium/RTK>) — ITK-based GPU FDK + iterative, clinical DICOM-RT.
- **ASTRA Toolbox** (<https://github.com/astra-toolbox/astra-toolbox>) — GPU forward/back-projection primitives (parallel/fan/cone).
- **TIGRE** (<https://github.com/CERN/TIGRE>) — CUDA FDK + iterative algorithms, real-data focus.
- **Plastimatch** (<https://plastimatch.org/>) — GPU FDK, DRR, registration.

Study these for the production approach; reimplement didactically (CLAUDE.md §2).

## CUDA pattern used here

Per-pixel **backprojection gather** on a 2-D thread grid · linear detector
interpolation · host-precomputed trig for CPU/GPU parity · independent outputs
(no atomics). Texture memory is the production accelerator for the interpolation.

## Exercises

1. **Texture interpolation.** Bind the filtered sinogram to a `cudaTextureObject_t`
   and replace the manual linear interp with `tex2D` — the production trick.
2. **Filter on the GPU with cuFFT.** Move the ramp filter to the frequency domain
   (`cufftExecR2C` → multiply by |ω| → `cufftExecC2R`). Compare to the spatial filter.
3. **Fan-beam geometry.** Add the fan-beam weighting and rebinning. How do the ray
   equations change?
4. **3-D FDK.** Extend to cone-beam: a 3-D voxel grid, 2-D projections, and the
   FDK cosine weighting — the algorithm clinical scanners use.
5. **Compare without the ramp filter.** Backproject the *raw* sinogram and see the
   characteristic `1/r` blur the filter removes.

## Limitations & honesty

- **2-D parallel-beam only** (real scanners are fan/cone beam; FDK is the 3-D
  extension, described in THEORY).
- The ramp filter is a **spatial** convolution here for clarity; production filters
  via FFT.
- Reconstructed values are arbitrary phantom-density units, not calibrated HU.
- Interpolation is manual linear (no texture hardware) so the math is visible.
