# 4.31 — Virtual Colonoscopy & CT Colonography

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟢 Beginner · Established** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.31`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

**CT colonography** (virtual colonoscopy) acquires a CT of the air-distended colon
and lets a radiologist "fly through" its hollow interior on screen, looking for
**polyps**. This project builds the heart of that experience: a GPU **volume
ray-caster** that renders one fly-through frame from *inside* the colon. Each
output pixel casts an independent ray into a 3-D CT volume, finds where the ray
crosses the air→wall boundary, and shades that surface with a headlamp light — the
textbook **per-pixel gather** GPU pattern. The input is a tiny **synthetic** volume
(an air-filled bent tube with one planted polyp), so the demo runs offline and the
polyp is a *known answer* you can watch the renderer recover.

## What this computes & why the GPU helps

The colon's lumen is air (low CT density) and its wall is soft tissue (higher
density), separated by a sharp jump. We render the wall as the **iso-surface**
where the trilinearly-interpolated density crosses a threshold: march a ray from
the camera, stop at the first crossing, estimate the surface normal from the
**density gradient**, and apply **Blinn-Phong** shading. A polyp is a convex bump
into the lumen, so it catches the light and reads brighter than the flat wall.

**The parallel bottleneck:** the rendering itself. A clinical fly-through renders a
**512³ volume at 60 frames/second** — billions of trilinear samples per second.
Every pixel's ray is independent, so the GPU runs thousands of them at once and its
texture units do the interpolation almost for free. This is why real-time virtual
colonoscopy is a GPU application. See [THEORY.md](THEORY.md) §1, §4.

## The algorithm in brief

- **Per-pixel ray generation** from the virtual-endoscope camera (eye, basis, FOV).
- **First-hit iso-surface ray-march**: step along the ray, trilinearly sample the
  volume, stop at the first air→wall density crossing; refine with one secant step.
- **Gradient normal**: central differences of the density field give the surface
  normal (the "gradient-magnitude" surface estimate).
- **Blinn-Phong shading** with a headlamp (light = view direction), as in clinical
  CTC rendering.
- **Verify**: a serial CPU reference renders the same frame; the GPU must match it.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/virtual-colonoscopy-ct-colonography.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/virtual-colonoscopy-ct-colonography.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\virtual-colonoscopy-ct-colonography.sln /p:Configuration=Release /p:Platform=x64
```

This project links **only the CUDA runtime** (`cudart`); no extra CUDA math library
is needed — the ray-caster is hand-written CUDA C++.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the rendered-frame stats
plus a small **ASCII preview** of the fly-through, shows the GPU-vs-CPU agreement
check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/colon_volume_sample.txt` — a tiny **synthetic**
  32×32×48 CT volume (air-filled bent tube + one planted polyp) so the demo runs
  with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print how to obtain real
  CTC volumes (the script never bypasses registration). `scripts/make_synthetic.py`
  regenerates / enlarges the synthetic volume.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: TCIA CT Colonography dataset
(<https://wiki.cancerimagingarchive.net/display/Public/CT+Colonography>);
MICCAI 2018 colon challenge; ACR Lung-RADS CT dataset; NLST CTC subsets.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
4.31 -- Virtual Colonoscopy & CT Colonography
CT colonography fly-through: volume 32x32x48 -> frame 48x48 (iso=0.50)
wall-hit pixels = 2296 / 2304 (0.997)
mean intensity = 0.6041
max intensity = 1.0000 at (px,py)=(27,8)
polyp-region mean brightness = 0.7331
ascii preview (24x12, '@'=bright wall, ' '=dark lumen/background):
  |%######*****************|
  |#####******++++++++++***|
  |####****+++++#@@%*++++++|
  |###****++++#*###*+++++++|
  |##****++==-+#@@%#+==++++|
  ...
RESULT: PASS (GPU matches CPU within tol=1.0e-03)
```

The round lumen (bright `#%` walls around a dark center, where rays travel far down
the tube) is visible in the ASCII preview, with the **polyp** as the bright
`#@@%` cluster in the upper-center. The program renders on the **GPU**
(`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and asserts
they agree within `1e-3` (actual error ≈ `5e-7`). The `polyp-region mean
brightness ≈ 0.73` (vs ≈ 0.36 without the polyp) is the recovered known answer.
Timings vary and print to stderr, so they are not part of the diffed output.

## Code tour

Read in this order:

1. [`src/volume_render.h`](src/volume_render.h) — **start here**: the shared
   `__host__ __device__` per-ray math (trilinear sample, gradient, Phong,
   `cast_ray`). This is the real content; CPU and GPU both call it.
2. [`src/reference_cpu.h`](src/reference_cpu.h) — the `Scene`/`Camera` types and
   `pixel_ray()` (ray generation), plus the project overview.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the volume loader (+ camera
   placement) and the serial reference renderer.
4. [`src/kernels.cuh`](src/kernels.cuh) / [`src/kernels.cu`](src/kernels.cu) — the
   2-D ray-casting kernel and its host wrapper (upload → launch → copy back).
5. [`src/main.cu`](src/main.cu) — loads the scene, runs CPU + GPU, verifies,
   reports the deterministic stats + ASCII preview.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, the CUDA-event timer, and I/O.

## Prior art & further reading

- **3D Slicer** (<https://github.com/Slicer/Slicer>) — open-source imaging platform;
  study its `VolumeRendering` and colon-segmentation modules for the production
  rendering and segmentation pipeline.
- **VTK** (<https://vtk.org/>) — the GPU volume ray-casting engine many viewers
  build on (`vtkGPUVolumeRayCastMapper` is the reference for this project's stage).
- **MONAI** (<https://github.com/Project-MONAI/MONAI>) — nnU-Net colon/lumen
  segmentation and 3-D CNN polyp detection (the pipeline stages we omit).
- **VisIt** (<https://visit-dav.github.io/visit-website/>) — GPU visualization for
  very large CT volumes.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Per-output-pixel gather with interpolation** (PATTERNS.md §1) — the same family
as `4.01` CT backprojection. A 2-D thread grid maps one thread to one pixel; each
thread marches one independent ray, doing trilinear volume lookups (a hand-written
3-D "texture" sample) and writing one pixel. No shared memory, no atomics. The
per-ray math is shared verbatim with the CPU reference via a `__host__ __device__`
header (PATTERNS.md §2), making the GPU-vs-CPU check exact to FP32 rounding.

## Exercises

1. **Use the texture unit.** Replace `sample_volume()` with a CUDA 3-D texture and
   `tex3D<float>()`. Measure the speed-up — and see why the GPU image no longer
   matches the CPU reference bit-for-bit (THEORY §6: 9-bit interpolation weights).
2. **Animate the fly-through.** Loop the camera `eye` down the +z axis a few voxels
   per frame and dump a sequence of frames; you now have a (tiny) fly-through.
3. **Empty-space skipping.** Precompute a coarse "is there any wall in this block?"
   grid and skip marches through pure air. How much does it cut the step count?
4. **A second polyp / a fold.** Edit `make_synthetic.py` to add another bump or a
   haustral fold, and confirm the renderer (and your eye) can tell them apart.
5. **Shape index.** Compute the curvedness / shape-index of the hit surface from the
   gradient — the classic hand-crafted polyp-candidate feature (catalog stage 5).

## Limitations & honesty

- **Synthetic data.** The volume is a hand-built tube + sphere, **not** a patient
  CT, and densities are unitless 0..1, **not** Hounsfield Units. It is labeled
  synthetic everywhere and makes **no clinical claim**.
- **Reduced scope.** We implement only the *rendering* stage (stage 4) of the
  five-stage CTC pipeline; segmentation, cleansing, centerline extraction, and CNN
  polyp detection are described in THEORY §7 but not built.
- **Hand-written interpolation, global memory.** We trade the texture unit's speed
  and caching for an exact CPU/GPU match and readable code; a production renderer
  uses a CUDA 3-D texture (see Exercise 1).
- **Single frame, first-hit only.** No transfer functions, no semi-transparent
  compositing, no supine/prone registration, no early-ray termination — all of
  which a clinical viewer has.
- **Timing is a teaching artifact, not a benchmark.** The GPU's edge grows with
  frame size, volume size, and frame count; on this tiny scene it is only
  illustrative (CLAUDE.md §12).
