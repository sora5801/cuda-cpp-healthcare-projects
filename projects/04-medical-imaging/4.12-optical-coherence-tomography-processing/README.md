# 4.12 — Optical Coherence Tomography Processing

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.12`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Optical Coherence Tomography (OCT) is the "optical ultrasound" of ophthalmology:
it images the layered micro-structure of the retina at micron resolution. A
spectral-domain OCT (SD-OCT) instrument does not measure depth directly — it
measures an interference **spectrum** per axial line (an "A-scan"), and the depth
profile is recovered by a Fourier transform of that spectrum. This project
reconstructs a whole **B-scan** (a stack of A-scans → a 2-D cross-section) on the
GPU: a custom CUDA kernel removes the background, windows the spectrum, and applies
**dispersion compensation** (an OCT-specific phase correction), then a single
**batched cuFFT** call transforms every A-scan at once, and a final kernel forms
the log-magnitude image. A naive-DFT CPU reference reconstructs the same B-scan so
we can verify the GPU exactly.

## What this computes & why the GPU helps

Spectral-domain OCT acquires spectra per A-scan (axial line); reconstruction
requires dispersion compensation, interpolation from wavelength to wavenumber
space, and a 1-D FFT per A-scan. A single B-scan of 2,048 A-scans × 2,048 spectral
pixels requires 2,048 FFTs of length 2,048 — trivially parallelizable in GPU
batches. Real-time 3-D OCT volumes for surgical guidance require ~100 B-scans/s
(~4 × 10⁸ FFT points/s), achievable only on a GPU. Downstream retinal-layer
segmentation and fluid detection add CNN inference (TensorRT U-Nets), which we
describe but do not implement here.

**The parallel bottleneck:** the **per-A-scan FFT**. Every A-scan's transform is
independent of every other A-scan's, so the whole B-scan's thousands of FFTs are
embarrassingly parallel — exactly the "batched 1-D FFT" shape cuFFT is built for
(`cufftPlan1d(..., CUFFT_C2C, n_ascan)` does them all in one call). The custom
per-sample preprocessing (dispersion compensation) is likewise one independent
thread per spectral sample.

## The algorithm in brief

- **DC / background removal** — subtract each A-scan's mean (the strong
  non-interferometric offset).
- **Windowing** — multiply by a Hann window to suppress FFT side lobes (spectral
  leakage).
- **Dispersion compensation** — multiply by `exp(-i·φ(k))`, `φ(k)=a₂(k-k₀)²+a₃(k-k₀)³`,
  cancelling the sample/reference-arm dispersion mismatch that otherwise smears
  every depth peak (the OCT-specific step; **the custom kernel**).
- **Batched 1-D FFT** — one FFT per A-scan, entire B-scan in one **cuFFT** call.
- **Log-magnitude** — `|A(z)|²`, normalised per A-scan → the displayed B-scan.

The full catalog scope also lists k-space resampling (NUFFT), 3-D graph-cut and
U-Net layer segmentation, and Doppler velocity mapping; this is the **reduced-scope
teaching version** (the spectral reconstruction core), with the rest covered in
[THEORY.md](THEORY.md) → "Where this sits in the real world" (CLAUDE.md §13).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)). This project links
**cuFFT** (`cufft.lib`, already wired into the `.vcxproj` — BUILD_GUIDE §7b).

1. Open `build/optical-coherence-tomography-processing.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/optical-coherence-tomography-processing.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\optical-coherence-tomography-processing.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if the optional CMake build is used)
```

The demo builds if needed, runs on `data/sample/oct_bscan.txt`, prints the
recovered peak depths and an ASCII B-scan, shows the GPU-vs-CPU agreement check,
and prints a timing line (on stderr).

## Data

- **Sample (committed):** `data/sample/oct_bscan.txt` — a tiny **synthetic** B-scan
  (32 A-scans × 256 spectral samples, injected dispersion) so the demo runs with
  zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (documented, idempotent;
  they point at public OCT image datasets and download nothing).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: OCTDL (<https://www.nature.com/articles/s41597-024-03182-7>) — 2,064 labeled OCT B-scans; Duke DME OCT dataset (<https://people.duke.edu/~sf59/Chiu_BOE_2012_dataset.htm>) — 110 annotated volumes; OCTA-500 (<https://arxiv.org/abs/2012.07261>) — OCT angiography volumes with labels. These are **reconstructed images** (for segmentation/classification), not the raw spectra this project reconstructs from.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program reconstructs the B-scan on the **GPU** (`src/kernels.cu`, custom kernels +
cuFFT) and on a **CPU reference** (`src/reference_cpu.cpp`, naive DFT), then checks
agreement two ways:

1. **Exact** — the per-A-scan **peak-depth index** (integer argmax of the depth
   profile) matches CPU↔GPU bit-for-bit (order-independent, deterministic).
2. **Within tolerance** — the normalised images agree to `atol = 2e-4` (cuFFT is
   single precision and reorders additions vs. the double-precision naive DFT; on
   the sample the worst difference is ~`1e-7`).

The recovered peak depths trace the synthetic surface arc (`8 → 14 → 8`), and the
ASCII B-scan shows the curved bright layer — a visible demonstration that
dispersion compensation sharpened the reflectors.

## Code tour

Read in this order:

1. [`src/oct_core.h`](src/oct_core.h) — the **shared** `__host__ __device__`
   per-sample math (window, dispersion phase, complex helpers) used identically by
   CPU and GPU. Start here: it is the numerical heart.
2. [`src/main.cu`](src/main.cu) — loads the B-scan, runs CPU + GPU, verifies
   (exact peak depths + image tolerance), prints the deterministic report.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the two-pattern idea
   (custom kernel wrapping a library call).
4. [`src/kernels.cu`](src/kernels.cu) — `dc_kernel` → `preprocess_kernel` →
   **`cufftExecC2C`** (batched) → `power_norm_kernel`, with the cuFFT call fully
   explained (no black box).
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted naive-DFT baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, host I/O.

## Prior art & further reading

- **OCT-Marker** (<https://github.com/neurodial/OCT-Marker>) — annotation tool for
  OCT B-scans; study its data model for how B-scans and layer labels are stored.
- **Iowa Reference Algorithms** (<https://www.iibi.uiowa.edu/content/shared-software-Iowa-reference-algorithms>)
  — graph-based retinal-layer segmentation; the classic 3-D graph-cut approach the
  catalog references for the downstream step.
- **k-Wave CUDA** (<https://github.com/klepo/k-Wave-Fluid-CUDA>) — GPU acoustic
  simulation, relevant to photoacoustic-OCT extensions.
- **NVIDIA cuFFT samples** — real-time OCT reconstruction demos and the canonical
  batched-FFT usage this project mirrors.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Batched 1-D FFT via cuFFT**, wrapped by **custom kernels**. `cufftPlan1d(…,
CUFFT_C2C, n_ascan)` + one `cufftExecC2C` transforms every A-scan of the B-scan in
a single call (the library step, documented not black-boxed). A custom
`preprocess_kernel` (one thread per spectral sample) does the OCT-specific DC
removal + windowing + **dispersion phase correction** the library cannot, and a
custom `power_norm_kernel` (one thread per A-scan) forms and normalises the
magnitude image. See [docs/PATTERNS.md](../../../docs/PATTERNS.md) §5 ("using a
CUDA library without a black box") and the cuFFT flagship `8.03`. The full pipeline
would add cuDNN/TensorRT for U-Net segmentation and CUDA streams to overlap
acquisition with reconstruction.

## Exercises

1. **Turn dispersion compensation off** — run with `a2=a3=0` (edit the sample
   header or regenerate with `--a2 0 --a3 0`) and watch the depth peaks broaden.
   Quantify the axial-resolution loss (peak full-width at half-maximum).
2. **Sweep the block size** — try 128 / 256 / 512 threads/block in `kernels.cu`
   and compare the stderr timing. Where does occupancy stop helping?
3. **Add a second window** — implement a Hamming or Blackman window alongside Hann
   in `oct_core.h`; compare side-lobe suppression vs. main-lobe width.
4. **Scale up** — regenerate with `--n-ascan 512 --n-spec 2048` (real B-scan size)
   and watch the GPU/CPU gap widen; the CPU naive DFT becomes the bottleneck.
5. **k-space resampling** — real spectrometers sample uniformly in wavelength λ,
   but the FFT needs uniform wavenumber k = 2π/λ. Add a linear-interpolation
   resampling kernel before the FFT (the NUFFT step named in the catalog).

## Limitations & honesty

- **Synthetic data.** The sample is generated (reflectors at known depths +
  injected dispersion + noise), labeled synthetic everywhere, and of **no
  diagnostic meaning**. It is engineered so the reconstruction has an obvious,
  verifiable answer.
- **Reduced scope.** This implements the spectral **reconstruction** core
  (DC/window/dispersion + batched FFT + log-magnitude). It does **not** do
  wavelength→wavenumber resampling (assumes already-uniform-in-k spectra), layer
  segmentation, fluid detection, or Doppler — those are described in THEORY, not
  coded.
- **Precision.** cuFFT runs in single precision; the CPU reference in double. We
  verify the **exact** integer peak depths and the image to a small, documented
  tolerance — we do **not** claim bit-identical floating-point images.
- **Timing is a teaching artifact,** not a benchmark claim (CLAUDE.md §12): the
  sample is tiny; the GPU's real advantage appears at clinical B-scan sizes.
- **Not for clinical use.** Educational only.
