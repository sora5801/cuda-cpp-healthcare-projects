# 4.13 — Photoacoustic Image Reconstruction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.13`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Photoacoustic imaging (PAI) shines a nanosecond laser pulse into tissue; light-absorbing
structures (blood, melanin, injected contrast) heat up, expand, and ring like tiny bells,
emitting ultrasound. A ring of sensors records the resulting pressure-vs-time traces, and we
**reconstruct** a picture of *where the absorbers were* from those traces. This project
implements the workhorse reconstruction — **delay-and-sum (DAS) backprojection** — as a CUDA
kernel that assigns **one thread per image pixel**. It runs the same computation on the CPU as a
trusted reference, verifies the two agree, and reports the recovered source locations on a small,
fully **synthetic** dataset with a known answer.

## What this computes & why the GPU helps

Photoacoustic imaging (PAI) generates ultrasound waves by pulsed laser absorption in tissue; images are reconstructed from time-series pressure data on a sensor surface. Delay-and-sum backprojection is analogous to ultrasound but in 3D; for 1,024 sensors and a 256³ volume, ~68 billion delay-and-sum operations are required per image — tractable only on GPU. Model-based iterative reconstruction solves the wave equation numerically (k-space pseudospectral method via cuFFT), enabling quantitative PAI with accurate acoustic attenuation and heterogeneous speed-of-sound modelling. Real-time 3D PA imaging for interventional guidance requires GPU throughput of multiple frames/second.

**The parallel bottleneck:** the backprojection **gather**. Every output pixel independently
reads one interpolated sample from *every* sensor trace and sums them: cost `img² × n_sensors`
in 2-D (`img³ × n_sensors` in 3-D). That is the ~68-billion-operation figure above, and it is the
step that dominates runtime. Because pixels never depend on each other, the gather is
embarrassingly parallel — the ideal GPU workload. We map a 2-D thread grid onto the 2-D image so
thread `(px, py)` reconstructs pixel `(px, py)`; on this sample the GPU kernel is ~10× faster than
the serial CPU loop, and the gap widens with image size and sensor count.

## The algorithm in brief

- **Delay-and-sum (DAS) backprojection** (implemented here): for each pixel `x`, sum
  `g_s(|x − p_s|/c)` over all sensors `s`, then normalize by the sensor count.
- **Linear interpolation** of each sensor trace at the (fractional) arrival sample.
- Related methods discussed in [THEORY.md](THEORY.md): universal back-projection (adds a
  time-derivative + solid-angle weight), time-reversal, k-space pseudospectral wave propagation
  (k-Wave, via cuFFT), iterative model-based PA reconstruction, compressed-sensing PAI, and deep
  end-to-end reconstruction.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/photoacoustic-image-reconstruction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/photoacoustic-image-reconstruction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\photoacoustic-image-reconstruction.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/pa_sample.txt`, prints the reconstructed peak and
a profile, shows the GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/pa_sample.txt` — a tiny, offline **synthetic** photoacoustic
  acquisition (64 ring sensors × 512 time samples) with three planted point absorbers, so the demo
  runs with zero downloads and has a **known** answer.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (print instructions/links; they never bypass
  any registration). `scripts/make_synthetic.py` regenerates the sample deterministically.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: k-Wave simulation datasets (generated locally); USCT (Ultrasound Computed Tomography) benchmark data (verify URL); in vivo photoacoustic datasets from Nature Communications publications (open access); PASCAA challenge datasets (verify URL at photoacoustics.org).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
4.13 -- Photoacoustic Image Reconstruction
2-D delay-and-sum backprojection
64 sensors x 512 samples -> 96x96 image (c=1500.0 m/s, dt=5.000e-08 s)
peak value = 33.2748 at pixel (px,py)=(47,47) = (x,y)=(-0.0001,-0.0001) m
center-row profile (8 samples): 0.1328 2.0825 1.8289 3.5928 4.2744 1.7030 1.8482 0.1791
RESULT: PASS (GPU matches CPU within tol=1.0e-03)
```

The reconstructed image peaks at pixel `(47,47)`, i.e. world `(≈0, ≈0)` m — exactly where the
**strongest** planted absorber sits, so the reconstruction recovered the ground truth. The program
computes the image on both the **GPU** (`src/kernels.cu`) and a **CPU reference**
(`src/reference_cpu.cpp`) and asserts they agree within `1e-3` — the correctness guarantee. The two
differ only at the `~3e-4` level because the GPU fuses multiply-adds (FMA) while the host compiler
rounds twice; that is documented and physically negligible (peak ≈ 33). stdout is deterministic and
diffed; timing goes to stderr.

## Code tour

Read in this order:

1. [`src/pa_core.h`](src/pa_core.h) — the **shared** `__host__ __device__` physics: `pa_pixel_das`
   (the delay-and-sum formula) and `pa_sample_trace` (trace interpolation). Both CPU and GPU call
   these, so they compute the *same* math. Start here.
2. [`src/main.cu`](src/main.cu) — loads the acquisition, runs CPU + GPU, verifies, reports.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the pixel→thread mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the `das_kernel` (2-D grid) and its host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the loader + trusted serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, host I/O helpers.

## Prior art & further reading

k-Wave (http://www.k-wave.org/, CUDA C++ version at https://github.com/klepo/k-Wave-Fluid-CUDA) — industry-standard PA/US simulation and reconstruction toolbox; OpenMSOT (open multi-spectral optoacoustic tomography framework, verify URL); k-Wave MATLAB + CUDA backend for fast GPU wave simulation; PyTomography (https://github.com/lukepolson/pytomography) — Python GPU tomographic reconstruction including photoacoustic.

- **k-Wave** — study its forward simulation (k-space pseudospectral solver) to see how the synthetic
  traces we hand-generate here would be produced physically, and its time-reversal reconstruction as
  the "gold" alternative to DAS.
- **k-Wave-Fluid-CUDA** — read how the FFT-based propagation maps to cuFFT and multi-GPU planes.
- **PyTomography** — a clean, readable GPU tomography codebase; compare its backprojection to ours.

Study these to learn the production approach; **do not copy code wholesale** — reimplement
didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Per-output-pixel **gather** with interpolation — the same pattern as CT filtered backprojection
(flagship `4.01`), and the row in [docs/PATTERNS.md](../../../docs/PATTERNS.md) §1 for
"per-output-pixel/voxel gather". One thread per pixel loops over all sensors, computes a
travel-time delay `|x − p_s|/c`, and linearly interpolates each sensor's time-series there. No
shared memory or atomics are needed (pixels are independent). We also use the
`__host__ __device__` **shared-core** idiom (PATTERNS.md §2) so the CPU reference and GPU kernel run
identical per-pixel math. The catalog also lists cuFFT-based k-space propagation, texture-memory
interpolation, and a shared-memory sensor LUT — advanced optimizations described in THEORY.md but
deliberately left as exercises to keep the teaching kernel legible.

## Exercises

1. **Texture-memory interpolation.** Bind each sensor trace to a 1-D CUDA texture and let the
   texture unit do the linear interpolation in hardware (this is how production DAS runs). Compare
   speed and the max error against the current version.
2. **The bipolar reality.** Regenerate the data with a *bipolar* pulse (the time-derivative of the
   Gaussian — edit `make_synthetic.py`). The raw DAS image now has negative side-lobes; add an
   **envelope** step (Hilbert transform magnitude) so the peak-at-source check works again.
3. **3-D voxels.** Extend `PAProblem` and `pa_pixel_das` to a `z` coordinate and a 3-D thread grid;
   confirm the loop is unchanged apart from one more distance term.
4. **Speed-of-sound error.** Reconstruct with a `c` that is 3 % wrong and watch the point spread
   blur — this is why quantitative PAI needs speed-of-sound modelling.
5. **Sparse aperture.** Drop to 16 sensors and observe the streak artifacts; relate them to the
   `img² × n_sensors` cost and to compressed-sensing PAI.

## Limitations & honesty

- **Synthetic data.** The sample is generated by a simple analytic forward model (a Gaussian pulse
  per absorber arriving at `distance/c`), **not** a full acoustic simulation. It is labeled synthetic
  everywhere. Real traces come from k-Wave or a scanner.
- **Unipolar pulse.** We use a positive-only pulse so "brightest pixel = source" is unambiguous. A
  real PA pulse is bipolar, which makes raw DAS bipolar; production takes an envelope or uses
  universal back-projection (THEORY.md §Where-this-sits).
- **Homogeneous, non-attenuating, 2-D.** We assume a single constant speed of sound, no acoustic
  attenuation, point (omnidirectional) sensors, and a 2-D slice. Quantitative PAI models
  heterogeneous `c(x)`, frequency-dependent attenuation, finite sensor apertures, and full 3-D.
- **Not a benchmark, not clinical.** Timings are a teaching artifact; the images are arbitrary-unit
  demonstrations, never a diagnostic result.
