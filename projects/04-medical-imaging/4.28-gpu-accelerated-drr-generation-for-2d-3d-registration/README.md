# 4.28 — GPU-Accelerated DRR Generation for 2D/3D Registration

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟢 Beginner · Established** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.28`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A **Digitally Reconstructed Radiograph (DRR)** is a *simulated* X-ray: you take a
3-D CT volume, pick an X-ray source/detector pose, and for every detector pixel
shoot a ray through the volume, integrating how much the tissue along it attenuates
the beam. This project renders a DRR of a small synthetic CT phantom both on the CPU
(a clear serial reference) and on the GPU (one thread per detector pixel, each
marching its own ray and tri-linearly sampling the volume), then verifies the two
agree. DRR generation is the inner loop of **2D/3D registration** — lining up a
daily treatment X-ray with the planning CT — and because every pixel is independent
it is a textbook GPU "gather" workload.

## What this computes & why the GPU helps

Digitally reconstructed radiographs (DRRs) simulate X-ray images from 3D CT volumes
for 2D/3D registration (aligning daily X-ray portal images to planning CT). Each DRR
pixel integrates CT Hounsfield units along a ray path through the volume (Siddon's
ray-tracing or tri-linear ray-marching); for a 400×400 DRR from a 512³ CT,
~6.4 × 10⁸ tri-linear interpolations are needed per DRR. Intensity-based 2D/3D
registration requires 50–200 DRRs per optimization iteration (~10¹¹ operations total
on CPU). GPU texture memory's built-in tri-linear hardware interpolation and
embarrassing parallelism (one CUDA thread per DRR pixel) make this an ideal GPU
workload, achieving 100×+ speedup.

**The parallel bottleneck:** the **per-pixel ray integral** `∫ μ ds`. Each of the
`W×H` detector pixels marches a ray of ~`n` steps, each step doing a tri-linear
(8-voxel) read — `O(W·H·n)` independent fused-multiply-adds with *no* data sharing
between pixels. That independence is exactly what maps to one GPU thread per pixel.
In the demo the GPU kernel is ~0.6 ms vs ~170–240 ms for the serial CPU render.

## The algorithm in brief

- **HU → μ conversion** (Hounsfield Units to linear attenuation, once at load).
- **Cone-beam geometry**: point source + flat detector; each pixel defines a ray.
- **Ray-marching DRR** (tri-linear ray-casting): fixed-step midpoint quadrature of
  `∫ μ ds` along each ray, sampling the volume with **tri-linear interpolation**.
- Related methods named in the catalog and discussed in THEORY: **Siddon** exact
  ray-tracing, splatting vs. ray-casting, similarity metrics (MI / NCC /
  gradient-magnitude), SGD pose optimization, **differentiable DRR (DiffDRR)**.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/gpu-accelerated-drr-generation-for-2d-3d-registration.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/gpu-accelerated-drr-generation-for-2d-3d-registration.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\gpu-accelerated-drr-generation-for-2d-3d-registration.sln /p:Configuration=Release /p:Platform=x64
```

Both `Debug|x64` and `Release|x64` build with zero warnings. The project links only
`cudart_static.lib` (no extra CUDA libraries — the kernel is hand-written
ray-marching).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/ct_volume_sample.txt`, prints the
deterministic DRR samples, shows the GPU-vs-CPU agreement check, and prints a timing
line on stderr.

## Data

- **Sample (committed):** `data/sample/ct_volume_sample.txt` — a tiny **synthetic**
  32×32×32 CT phantom (air + soft-tissue sphere + an **offset** dense bone sphere)
  so the demo runs offline with zero downloads, and the brightest DRR pixel lands
  off-axis in a predictable place.
- **Generate / resize:** `python scripts/make_synthetic.py --n 64` for a bigger
  synthetic volume (deterministic, no randomness).
- **Real data:** `scripts/download_data.ps1` / `.sh` print pointers to TCIA, Gold
  Atlas, and AAPM TG-132 CT sets and never bypass any data-use agreement.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: Gold Atlas prostate CT (https://www.goldenatlasproject.com/ —
verify URL); TCIA prostate/lung CTs; AAPM TG-132 test cases; clinical CBCT + kV
images (institutional IRB).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
4.28 -- GPU-Accelerated DRR Generation for 2D/3D Registration
CT volume: 32x32x32 voxels, spacing 2.00x2.00x2.00 mm
DRR detector: 128x128 pixels, ray step 1.00 mm (cone-beam, lateral view)
center pixel attenuation = 1.2518
max attenuation = 1.3561 at (u,v)=(76,61)
central row profile (8 samples): 0.0000 0.2550 0.8656 1.0276 1.3355 0.8137 0.0202 0.0000
RESULT: PASS (GPU matches CPU within tol=1.0e-03)
```

The program renders the DRR on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and asserts they agree within `1e-3` (it
achieves ~8e-7, since both call the identical `integrate_ray`). The `max attenuation
at (u,v)=(76,61)` being *right of* the 64-px center column is the geometry sanity
check: the bone sphere is offset toward +y, and +y maps to increasing detector
column `u`.

## Code tour

Read in this order:

1. [`src/drr_core.h`](src/drr_core.h) — **start here.** The shared
   `__host__ __device__` per-ray physics: `hu_to_mu`, `sample_trilinear`, and
   `integrate_ray`. Both CPU and GPU call these, so their results match.
2. [`src/reference_cpu.h`](src/reference_cpu.h) /
   [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the loader, the demo geometry,
   and the trusted serial DRR (`render_drr_cpu`).
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-pixel
   "gather" idea (and the texture-memory upgrade, in comments).
4. [`src/kernels.cu`](src/kernels.cu) — `drr_kernel` (the 2-D grid) and
   `render_drr_gpu` (upload/launch/copy + CUDA-event timing).
5. [`src/main.cu`](src/main.cu) — load → CPU render → GPU render → verify → report.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **Plastimatch** (<https://plastimatch.org/>) — GPU DRR generation tool. Study its
  ray-casting geometry conventions and its CPU/GPU DRR options.
- **CUDA_DigitallyReconstructedRadiographs**
  (<https://github.com/fabio86d/CUDA_DigitallyReconstructedRadiographs>) — a compact
  GPU DRR library; good for seeing the texture-sampling kernel structure.
- **DiffDRR** (<https://github.com/eigenvivek/DiffDRR>) — *differentiable* DRR for
  gradient-based 2D/3D registration; the modern direction (and "neural DRR").
- **RTK** (<https://github.com/RTKConsortium/RTK>) — GPU ray-casting/forward
  projection used for DRR and CBCT.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

The **"gather" pattern** (docs/PATTERNS.md §1, exemplified by flagship 4.01 CT
backprojection): **one CUDA thread per output DRR pixel**, arranged as a 2-D
`16×16`-block grid over the detector panel; each thread runs a **ray-step loop** in
the kernel, tri-linearly sampling the volume; the small geometry struct is passed by
value (constant/parameter space) so every thread reads it cheaply. No shared memory,
no atomics — a pure independent gather. The catalog's full pattern also calls for a
**CUDA 3-D texture** for hardware tri-linear interpolation and **multiple streams**
for simultaneous multi-view DRRs; we keep the interpolation in plain device code so
it is fully visible, and describe the texture/stream upgrades in THEORY §4 and the
exercises below.

## Exercises

1. **Texture the volume.** Bind `d_mu` to a `cudaTextureObject_t` 3-D texture with
   `cudaFilterModeLinear` and replace `sample_trilinear()` with one `tex3D<float>()`
   call. Measure the speedup — this is the single biggest real-world optimization.
2. **Two views at once.** Render a second pose (e.g. an AP view along +y) using a
   separate CUDA **stream**, as registration needs orthogonal views. Compare timing
   to two sequential renders.
3. **Step-size study.** Sweep `STEP_MM` (0.25, 0.5, 1, 2 mm). Plot the center-pixel
   value and runtime; observe the `O(Δ²)` quadrature error and where aliasing starts
   when `Δ` exceeds the voxel size.
4. **Swap in Siddon.** Replace fixed-step marching with Siddon's exact ray–voxel
   traversal (exact intersection lengths, no quadrature error) and compare the DRR
   and the timing against the marching version.
5. **Close the registration loop.** Add a similarity metric (start with NCC) between
   the DRR and a "target" DRR rendered at a known perturbed pose, and a small
   coordinate-descent search over translation — recover the known offset.

## Limitations & honesty

- **Synthetic data.** The committed phantom is generated, labeled synthetic
  everywhere, and carries **no clinical meaning**. Nothing here is validated for
  diagnosis, treatment, or positioning.
- **One fixed pose.** We render a single lateral cone-beam view; real registration
  varies a full 6-DOF pose and renders 50–200 DRRs per iteration. The optimizer,
  similarity metric, and pose parameterization are described in THEORY but not
  implemented (see Exercise 5).
- **Plain interpolation, not texture hardware.** We do the tri-linear blend in
  device code for clarity; production engines use a 3-D texture (Exercise 1). The
  GPU timing here is therefore *conservative*.
- **Fixed-step quadrature, monoenergetic μ.** Real beams are polyenergetic (beam
  hardening) and DRR engines may use Siddon or splatting; our single-`μ_water`,
  single-energy model is a teaching simplification.
- **Timing is a teaching artifact, not a benchmark** (CLAUDE.md §12). It depends on
  the GPU, the panel size, and the ray length, and the small demo undersells the
  GPU's real advantage.
