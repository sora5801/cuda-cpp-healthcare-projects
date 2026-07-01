# 4.14 — Digital Breast Tomosynthesis

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.14`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Digital Breast Tomosynthesis (DBT) is "3-D mammography": instead of one flat X-ray,
the tube sweeps a **narrow arc** (~±15–25°) taking a handful of low-dose
projections, from which thin in-focus slices through the compressed breast are
reconstructed. Because the angular range is so **limited**, the classical analytic
inverse (Filtered BackProjection, project 4.01) is unstable, so DBT uses
**iterative algebraic reconstruction**. This project implements **SART**
(Simultaneous Algebraic Reconstruction Technique) on the GPU for a synthetic 2-D
breast slice with two planted "lesions", verifies it bit-for-bit against a serial
CPU reference, and uses it to teach the two embarrassingly-parallel gathers —
**forward projection** and **backprojection** — that sit at the heart of every
iterative CT/DBT reconstructor.

## What this computes & why the GPU helps

Digital breast tomosynthesis (DBT) acquires 9–25 low-dose projections over a limited angular range (~15–50°), then reconstructs thin slabs through compressed breast tissue. The limited-angle geometry makes analytical FBP unstable, so iterative methods (OS-EM, SART, ASD-POCS) with total-variation regularization dominate for artifact reduction. The breast is a low-contrast, soft-tissue object where noise and blur from the limited angle severely reduce lesion conspicuity, making statistical reconstruction critical. A single DBT volume (~800 × 700 × 60 slices at 85 µm) represents ~30 GB of raw projection data; GPU acceleration reduces OS-EM reconstruction from hours to under a minute. Deep learning methods (U-Net denoising on FBP outputs) additionally require GPU for inference.

**The parallel bottleneck:** every SART iteration does two huge, data-parallel
gathers — a **forward projection** (one line integral per detector ray, sampling
the image along the ray) and a **backprojection** (one correction per pixel/voxel,
gathering the residual from every angle). For a clinical volume that is billions of
ray–voxel interactions *per iteration*, over many iterations. Each output element
is independent, so we assign one GPU thread per ray (forward) and one thread per
pixel (backproject) — no atomics, no locks. That is exactly what a GPU eats for
breakfast, and why iterative DBT went from hours (CPU) to seconds (GPU).

## The algorithm in brief

- **Forward projection** `A·x`: simulate the projections the current image
  estimate *would* produce (numeric line integrals along each ray).
- **Residual** `b − A·x`: how far those simulated projections are from the measured
  data `b`.
- **Backprojection + relaxed update** `x ← x + λ·Aᵀ(b − A·x)/‖·‖`: spread the
  residual back into image space, normalize, scale by the relaxation factor λ, add,
  and clamp attenuation to ≥ 0.
- Repeat for a fixed number of iterations. Under a **limited angle** this iterative
  correction dramatically out-performs analytic FBP (which the catalog also lists:
  OS-EM, ASD-POCS/TV, MBIR are the production siblings of SART).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation (including why limited angle is ill-posed and how SART relates to OS-EM).

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/digital-breast-tomosynthesis.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/digital-breast-tomosynthesis.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\digital-breast-tomosynthesis.sln /p:Configuration=Release /p:Platform=x64
```

This project links only the CUDA runtime (`cudart_static.lib`) — no extra CUDA
libraries — so it builds out of the box on any machine with the ratified toolchain.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/dbt_sample.txt`, prints the
deterministic reconstruction samples, shows the GPU-vs-CPU agreement check, and
prints a timing line.

## Data

- **Sample (committed):** `data/sample/dbt_sample.txt` — a tiny **synthetic**
  compressed-breast phantom (soft-tissue ellipse + two dense lesion discs)
  forward-projected over a ±25° wedge, so the demo runs offline with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (documented, idempotent;
  prints links and never bypasses credentials).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: OPTIMAM Mammography Image Database (OMI-DB, access via ICR UK); CBIS-DDSM (https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM) — 2,620 mammograms via TCIA; VinDr-Mammo (https://physionet.org/content/vindr-mammo/1.0.0/); BCS-DBT (Duke DBT challenge dataset, https://bcs-dbt.grand-challenge.org/).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
4.14 -- Digital Breast Tomosynthesis
Limited-angle SART: 15 projections over +/-25.0 deg, 96 detectors -> 64x64 image
SART: 20 iterations, relaxation lambda = 0.30
center pixel value = 0.0717
peak value = 0.3479 at (px,py)=(23,31)
central row profile (8 samples): 0.0000 0.0000 0.0187 0.0000 0.0685 0.0642 0.0000 0.0000
RESULT: PASS (GPU matches CPU within tol=1.0e-03)
```

The **peak** lands on planted lesion 1 (world x ≈ −0.28·W, central row) — the
iterative method recovered a known dense structure from a narrow-angle wedge. The
program computes the reconstruction on both the **GPU** (`src/kernels.cu`) and a
**CPU reference** (`src/reference_cpu.cpp`) and asserts they agree within
`tol = 1.0e-3` (observed `max_abs_err ≈ 2e-7`, printed on stderr). That agreement
is the correctness guarantee.

## Code tour

Read in this order:

1. [`src/dbt_geometry.h`](src/dbt_geometry.h) — the shared `__host__ __device__`
   per-ray math (bilinear sampling + the forward line integral) that guarantees
   CPU/GPU parity. **Start here** — everything else calls it.
2. [`src/main.cu`](src/main.cu) — loads the problem, runs CPU + GPU SART, verifies,
   reports (deterministic stdout / timing on stderr).
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial SART
   (loader, forward project, residual, backproject-update loop).
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + thread-mapping idea.
5. [`src/kernels.cu`](src/kernels.cu) — the two projection kernels + the SART
   host driver (image kept device-resident across all iterations).
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, host I/O.

## Prior art & further reading

ASTRA Toolbox (https://github.com/astra-toolbox/astra-toolbox) — GPU forward/back-projection for arbitrary cone-beam geometry; RTK (https://github.com/RTKConsortium/RTK) — FDK and iterative DBT-capable; TIGRE (https://github.com/CERN/TIGRE) — DBT-compatible geometry; OpenDBT — research-focused DBT reconstruction framework.

- **ASTRA / TIGRE** — study how a production forward/back-projector handles
  *arbitrary* fan/cone geometry (ours is a simplified parallel-beam wedge) and how
  they exploit texture units for hardware interpolation.
- **RTK** — the iterative pipeline (SART/OS-EM, regularizers) at real scale.
- **SART's origin** — Andersen & Kak (1984), *Simultaneous Algebraic
  Reconstruction Technique (SART)*, Ultrasonic Imaging.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent gathers, no atomics** (docs/PATTERNS.md gather pattern, exemplified
by flagship 4.01 CT backprojection). Two kernels:

- **Forward projection** — 1-D grid, one thread per detector ray; the thread
  marches along its ray, bilinearly sampling the current image estimate and
  summing → the simulated line integral.
- **Backprojection + update** — 2-D grid, one thread per output pixel; the thread
  gathers the residual from every angle, averages, applies the relaxed SART
  correction, clamps ≥ 0.

Both are pure gathers (each output written by exactly one thread), so the result is
**deterministic** across runs and matches the CPU to float rounding — see
docs/PATTERNS.md §3. The image and scratch buffers stay **device-resident** across
all SART iterations, so there is zero per-iteration PCIe traffic.

> _The catalog also envisions cuFFT (ramp filter), texture-memory interpolation,
> constant-memory geometry, and an ADMM/TV inner loop. This teaching build keeps
> the numerics explicit and library-free so the reconstruction math is fully
> legible; those production accelerations are described in THEORY.md and left as
> exercises._

## Exercises

1. **Total-variation regularization (ASD-POCS).** Add a TV-minimization step
   between SART sweeps and watch limited-angle streaks shrink. Compare lesion
   contrast before/after.
2. **OS-EM instead of SART.** Split the angles into ordered subsets and update
   after each subset; measure convergence per iteration vs. plain SART.
3. **Texture-memory forward projection.** Bind the image to a CUDA texture and let
   the hardware do the bilinear interpolation (`tex2D`); compare speed and accuracy
   to the software `bilinear_sample()`.
4. **Sweep the angular wedge.** Regenerate data at ±10°, ±25°, ±45° and full 180°;
   plot how reconstruction quality (peak recovery, artifact level) improves with
   angular range — the core DBT-vs-CT trade-off.
5. **FP64 vs FP32.** Switch the accumulators to double and quantify how much the
   GPU/CPU agreement tightens over more iterations.

## Limitations & honesty

- **Synthetic data.** The phantom, its two lesions, and the projections are
  software constructs in **arbitrary attenuation units** — not calibrated
  Hounsfield/μ values, not a real breast, **not a diagnostic image**.
- **Simplified geometry.** We use **2-D parallel-beam** rays over a symmetric
  wedge. Real DBT is **3-D fan/cone-beam** with a divergent source, a moving
  detector, an oblique compression paddle, and per-slice focal planes. THEORY.md
  describes the extension.
- **Simplified reconstruction.** Plain SART with a simple column normalization and
  a non-negativity clamp — no scatter/beam-hardening correction, no statistical
  (Poisson) noise model, no TV/MBIR regularizer, no PSF/MTF modeling. Under the
  limited angle it therefore **under-recovers** attenuation (visible as low, muted
  reconstructed values) — an honest signature of the ill-posed geometry, not a bug.
- **Timing is a teaching artifact, never a benchmark claim** (CLAUDE.md §12). The
  reported speed-up depends entirely on problem size, block config, and the card.
