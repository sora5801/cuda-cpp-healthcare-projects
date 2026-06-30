# 4.6 — Ultrasound Beamforming

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟢 Beginner · Established** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.6`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

An ultrasound probe is a row of tiny transducer **elements**; after a pulse is
fired into the body, each element records a 1-D time signal of the returning
echoes (the **RF data**). This project reconstructs a B-mode image from that raw
RF data using **Delay-and-Sum (DAS) beamforming**: for every pixel in the image
we compute the round-trip travel time from the transmit source, to the pixel,
back to each element, look up (interpolate) each element's signal at that delay,
and sum. Where a real scatterer sits, the looked-up samples are the same echo in
phase and reinforce (bright); elsewhere they have random phase and cancel
(dark). That coherent sum *is* the focused image. Each pixel is independent, so
the GPU gives every pixel its own thread.

## What this computes & why the GPU helps

Delay-and-sum (DAS) beamforming reconstructs B-mode images by computing
time-delayed sums of per-element receive signals for every pixel in the image
grid. For a 128-element linear array, a 512×512 image, and 4,000 scan lines per
second, DAS requires ~10¹⁰–10¹¹ multiply-accumulate operations per second — far
beyond real-time CPU capability. GPU parallelism maps each output pixel to a
CUDA thread, computes focal delays from element geometry, interpolates raw RF
data, and sums across elements; a single RTX-class GPU achieves interactive
frame rates for volumetric beamforming.

**The parallel bottleneck:** the **per-pixel delay-and-sum gather**. Each of the
`nx·nz` pixels reads one interpolated sample from each of the `n_elements`
elements — `O(nx·nz·n_elements)` independent multiply-accumulates with no data
dependencies between pixels. This is the work parallelized across GPU threads
(one thread per pixel); the geometry/delay math is cheap, so the kernel is
gather/bandwidth-bound, exactly the shape GPUs excel at.

## The algorithm in brief

- **Delay-and-sum (DAS)** — the focusing law: `image(P) = Σ_e rf_e( τ_e(P) )`,
  where `τ_e(P)` is the round-trip time of flight from the transmit source to
  pixel `P` to element `e`.
- **Fractional-delay interpolation** — `τ_e(P)` rarely lands on an integer
  sample, so we linearly interpolate the two bracketing RF samples.
- **Envelope detection** — B-mode brightness is `|coherent sum|` (here, the
  magnitude of the signed DAS output).
- Related methods named in the catalog (coherence factor, DMAS, minimum-variance,
  SAFT, f-k migration) are discussed in [THEORY.md](THEORY.md) "Where this sits
  in the real world".

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/ultrasound-beamforming.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/ultrasound-beamforming.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\ultrasound-beamforming.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/rf_sample.txt`, prints the
recovered focal spot, shows the GPU-vs-CPU agreement check, and prints a timing
line.

## Data

- **Sample (committed):** `data/sample/rf_sample.txt` — synthetic RF echoes from
  **one point scatterer at (4.0, 20.0) mm**, so the result is exactly checkable
  (the focal spot must land there). Runs offline with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print real sources
  (PICMUS, Field II, k-Wave, MUST).
- **Provenance & license:** see [data/README.md](data/README.md). Synthetic,
  labeled synthetic.

Catalog dataset notes: PICMUS (Plane-Wave Imaging Challenge in Medical
Ultrasound) — RF data for beamforming evaluation; UltraSound SegLab dataset; IQ
ultrasound datasets from open research groups.

## Expected output

Success looks like `demo/expected_output.txt`. The program computes the image on
both the **GPU** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`)
and asserts they agree within `1e-3` — that agreement, plus the recovered focal
spot at `(3.9, 20.1) mm` (one pixel from the true `(4.0, 20.0) mm`), is the
correctness guarantee.

## Code tour

Read in this order:

1. [`src/beamform.h`](src/beamform.h) — the shared `__host__ __device__` DAS
   physics (`das_contribution`, `das_pixel`). **Start here:** this single formula
   is what makes the CPU and GPU results identical.
2. [`src/main.cu`](src/main.cu) — loads RF data, runs CPU + GPU, verifies,
   reports the focal spot.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping
   idea (one thread per pixel).
4. [`src/kernels.cu`](src/kernels.cu) — the `das_kernel` and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the RF loader + the trusted
   serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **MUST** (MATLAB Ultrasound Toolbox, <https://www.biomecardio.com/MUST/>) —
  reference DAS + GPU wrappers. Study its `das()`/`dasmtx()` for how production
  forms the delay matrix.
- **Field II** (<https://field-ii.dk/>) — the standard CPU ultrasound simulator;
  generates realistic RF data to feed a GPU DAS like this one.
- **k-Wave CUDA** (<https://github.com/klepo/k-Wave-Fluid-CUDA>) — CUDA
  time-domain full-wave acoustic propagation (a more physical forward model than
  our point-scatterer one).
- GPU-accelerated US beamforming repos on GitHub (search "CUDA ultrasound
  beamforming").

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Per-pixel gather with interpolation** (PATTERNS.md §1, exemplified by `4.01` CT
backprojection): one thread per output pixel, each thread loops over all inputs
(there: projection angles; here: transducer elements), interpolates one sample
per input, and accumulates. The geometry struct is passed **by value** so it
rides in kernel-parameter/constant space (every thread reads it with no global
traffic); only the bulky RF array goes through global memory. The per-element
physics lives in one `__host__ __device__` header so the CPU and GPU run
identical math (PATTERNS.md §2). The catalog also mentions cuBLAS for the
element sum and texture fetches for interpolation — discussed as alternatives in
[THEORY.md](THEORY.md).

## Exercises

1. **Lateral resolution vs. aperture.** Beamform using only the central 16
   elements instead of all 64 (zero the rest). The focal spot should widen — the
   main-lobe width scales like `λ·z / aperture`. Measure it.
2. **Add a coherence factor (CF).** Compute `CF = |Σ s_e|² / (N·Σ|s_e|²)` per
   pixel and multiply the image by it. Incoherent clutter is suppressed; this is
   one extra reduction per pixel, still embarrassingly parallel.
3. **Texture-memory interpolation.** Bind the RF data to a `cudaTextureObject_t`
   and replace the manual linear interpolation with a `tex1D`/`tex2D` fetch
   (hardware does the interpolation for free). Compare speed and accuracy.
4. **Two more scatterers.** Regenerate with `make_synthetic.py --extra` and
   confirm three focal spots appear at the documented locations.
5. **FP64 vs FP32.** Switch `das_pixel` to `double` and watch `max_abs_err`
   shrink toward machine precision — a concrete lesson in FMA/rounding.

## Limitations & honesty

- **Synthetic data, point-scatterer forward model.** The committed RF is
  simulated (`make_synthetic.py`): ideal point scatterers, a single virtual
  transmit from the array centre, linear propagation, no diffraction, no
  aberration, no multiple scattering, no electronic noise. Real RF is far messier.
- **Teaching-scope beamformer.** We implement plain DAS with linear
  interpolation; production systems add apodization, dynamic receive focusing,
  coherence/adaptive weighting, and clutter filtering (all sketched in THEORY).
- **Arbitrary units, not calibrated.** The image is not in dB or any clinical
  scale and must never be used for diagnosis or any medical decision.
- **`1e-3` tolerance, not bit-identity.** CPU and GPU run the same formula but
  the GPU may fuse `a*b+c` into one FMA where the host emits two rounded ops;
  over a sum of up to a few hundred elements that is a tiny absolute difference
  (here `~1.5e-4`). See [THEORY.md](THEORY.md) "How we verify correctness".
