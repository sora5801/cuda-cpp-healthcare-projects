# 4.10 — Super-Resolution Microscopy Reconstruction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.10`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Reconstruct a **super-resolution** image from a STORM/PALM movie by localizing
single fluorescent molecules. Each raw frame shows a sparse scatter of blurry
blobs (blinking fluorophores); this project **detects** each blob, **fits** its
sub-pixel centre by refining a Gaussian-weighted centroid over a 7×7 patch, and
**renders** all the localizations into a finely-upsampled image. Because every
blob is fit independently, the workload is embarrassingly parallel: one GPU thread
per candidate emitter, then an atomic scatter to build the image. This is the
flagship pattern **independent jobs + atomic reduction** (docs/PATTERNS.md §1),
made deterministic with fixed-point accumulation.

## What this computes & why the GPU helps

STORM/PALM single-molecule localization microscopy (SMLM) beats the ~250 nm
diffraction limit by imaging only a sparse random subset of fluorophores per
frame, so their blobs are separated and each one's true centre can be pinned to
~10–20 nm. A real acquisition is **10⁴–10⁵ frames** at 256²–512² pixels, each with
hundreds of blinking emitters — **tens of millions of independent PSF fits**. Each
fit reads only its own 7×7 patch and writes only its own (x,y): perfectly
parallel, one thread (or warp) per candidate spot.

**The parallel bottleneck:** the per-emitter **localization fit** (the 7×7 patch
detect + Gaussian-weighted-centroid refinement) is the dominant cost and is fully
data-parallel — that is what the GPU parallelizes. The final **render** is a
scatter of each emitter's intensity into a shared image, handled with `atomicAdd`.

## The algorithm in brief

- **Detect:** a candidate emitter is a strict 3×3 local-maximum pixel above a
  threshold (`smlm_is_local_max`).
- **Localize:** seed at the intensity-weighted centroid of the background-subtracted
  patch, then run `FIT_ITERS` fixed passes of **Gaussian-weighted centroid
  refinement** (`smlm_localize`) → sub-pixel (x, y), integrated photons, PSF width.
- **Render:** map each (x, y) to an 8×-upsampled bin and `atomicAdd` its
  fixed-point photons there (`render_kernel`).

The full production localizer is iterative 2D-Gaussian **maximum likelihood**; we
use the robust weighted-centroid estimator because it is deterministic and gives
bit-identical CPU/GPU results — see [THEORY.md](THEORY.md) §3, §6, §7.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/super-resolution-microscopy-reconstruction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/super-resolution-microscopy-reconstruction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\super-resolution-microscopy-reconstruction.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/smlm_stack.txt`, prints the
reconstruction digest, shows the GPU-vs-CPU agreement check, and prints a timing
line on stderr.

## Data

- **Sample (committed):** `data/sample/smlm_stack.txt` — a **synthetic** 60-frame
  40×40 STORM movie (two crossing sub-pixel lines), offline, ~660 KB.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print the real sources
  (EPFL SMLM Challenge, BioImage Archive, OME-TIFF) and the text format to export
  into; nothing is fetched automatically.
- **Provenance & license:** see [data/README.md](data/README.md). Bigger synthetic
  movie: `python scripts/make_synthetic.py --frames 200 --width 64 --height 64`.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): **187
emitters** localized across 60 frames, rendered into a **320×320** image (8×
upsampled) with **123 illuminated bins**. The program runs the whole pipeline on
both the **GPU** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`)
and asserts they agree **exactly** — same localization count, identical fixed-point
image (checksum and every pixel), and mean statistics matching to `0`. The fits
share one `__host__ __device__` core (`src/smlm.h`) and the render sums fixed-point
integers, so the two paths are bit-identical (see [THEORY.md](THEORY.md) §6).

## Code tour

Read in this order:

1. [`src/smlm.h`](src/smlm.h) — the shared **detect + fit + fixed-point** core
   (`__host__ __device__`), the heart of the project.
2. [`src/main.cu`](src/main.cu) — loads the movie, runs CPU + GPU, verifies, reports.
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`.cpp`](src/reference_cpu.cpp) —
   the loader, the trusted serial pipeline, and the shared render/summarize.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
5. [`src/kernels.cu`](src/kernels.cu) — the localize + atomic-render kernels + host loop.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **DECODE** (<https://github.com/TuragaLab/DECODE>) — deep-learning GPU SMLM
  localizer; orders of magnitude faster than MLE. Study how a network replaces the
  hand-fit per emitter.
- **ThunderSTORM** (FIJI plugin) — the reference open-source localizer; offers the
  *same* weighted-centroid fast method we teach here, plus MLE/least-squares.
- **NanoJ-SRRF** (<https://github.com/HenriquesLab/NanoJ-SRRF>) — GPU SRRF, an
  alternative to localization based on radial fluctuations (see THEORY §7).
- **fairSIM** (<https://github.com/fairSIM/fairSIM>) — GPU structured-illumination
  reconstruction (the FFT/OTF branch mentioned in the catalog).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent per-emitter jobs** — one GPU thread per interior pixel runs the same
detect+fit as the CPU on its own 7×7 patch — plus an **atomic scatter-reduction**
that renders localizations into a super-resolution image with `atomicAdd`.
**Fixed-point integers** make the atomic adds commute → the image is
order-independent, deterministic, and CPU-matching (docs/PATTERNS.md §3). The
localization list is kept in canonical (frame, row, col) order so CPU and GPU are
comparable element-for-element.

## Exercises

1. **Shared-memory / warp-per-emitter fit.** Stage each candidate's 7×7 patch in
   shared memory and assign one *warp* per candidate (the catalog's suggested
   mapping) instead of one thread per pixel. Compare occupancy and time.
2. **Real MLE fit.** Replace the weighted-centroid refinement with an iterative
   least-squares / maximum-likelihood 2D-Gaussian fit (amplitude, x, y, sigma,
   background) and study why it needs a *physical* tolerance to still match the CPU.
3. **Sub-pixel binning with anti-aliasing.** Splat each localization as a small
   Gaussian (weighted across neighbouring render bins) instead of a hard bin — the
   standard SMLM rendering — and see the reconstruction smooth out.
4. **Privatized render.** Give each block a small shared-memory histogram tile and
   flush it once, to cut global-atomic traffic on dense regions.
5. **GPU stream compaction.** Replace the host-side gather of valid localizations
   with a device `CUB::DeviceSelect::Flagged`, keeping the whole pipeline on the GPU.

## Limitations & honesty

- **Reduced-scope teaching version** (CLAUDE.md §13). We implement the SMLM
  localization + render pipeline with a robust *weighted-centroid* fit, not the
  full MLE fit, and none of SRRF/SOFI/SIM/DECODE from the catalog (those are
  described in THEORY §7).
- The **fit is chosen for determinism** (fixed iterations, shared `__host__
  __device__` math) so CPU==GPU exactly; a real MLE fit would diverge by ~1 ULP
  between compilers and need a physical tolerance.
- Detection is a simple local-max threshold; real data need denoising, multi-emitter
  fitting for overlaps, and drift correction.
- The data is **synthetic** and well-separated so the demo recovers the two lines;
  it carries no clinical or scientific claim about any specimen.
