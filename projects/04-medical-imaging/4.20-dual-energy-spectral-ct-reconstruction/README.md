# 4.20 — Dual-Energy / Spectral CT Reconstruction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.20`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A dual-energy CT scanner shoots each ray through the patient twice — once with a
low-energy X-ray spectrum (~80 kVp) and once with a high-energy one (~140 kVp).
Because different materials attenuate the two spectra differently, those two
measurements are enough to separate the tissue into **two basis materials**
(here: water-equivalent soft tissue and iodine-equivalent contrast/bone). This
project does the core of that separation — **projection-domain material
decomposition** — by solving a small **2×2 nonlinear system per sinogram bin**
with **Newton's method**. A real scan has on the order of 10⁸ bins and every bin
is completely independent, so we give **one GPU thread per bin**. It is a compact,
self-verifying example of turning a physics inverse-problem into an embarrassingly
parallel GPU kernel.

## What this computes & why the GPU helps

Dual-energy CT (DECT) acquires sinograms at two X-ray spectra (e.g., 80 kV and 140 kV) to enable material decomposition (separating water vs. iodine basis materials, or bone vs. soft tissue). Material decomposition in projection space requires solving a 2×2 nonlinear system per sinogram bin (~10⁸ bins), each requiring Newton iteration — trivially parallel across bins on GPU. Photon-counting CT (PCCT) extends this to 4–8 energy bins, increasing the system size to 8×8 and multiplying GPU compute by 4× but enabling K-edge imaging of contrast agents. Image-domain decomposition avoids projection-space issues but requires iterative reconstruction at each energy.

**The parallel bottleneck:** the per-bin **Newton iteration** on the polychromatic
forward model. Each bin evaluates a spectrum-weighted sum of exponentials
(`exp`/`log` over ~24 energy samples) a handful of times, then inverts a 2×2
Jacobian. That is a few thousand FLOPs of transcendental math per bin, repeated
over ~10⁸ bins — arithmetic-heavy, memory-light, and perfectly independent, i.e.
exactly what a GPU eats for breakfast. We map bin `i` to thread `i` (grid-stride
loop) and keep the shared scanner physics in constant memory. See
[THEORY.md](THEORY.md) → "GPU mapping".

## The algorithm in brief

Projection-domain material decomposition (Newton iteration per sinogram bin), image-domain material decomposition, basis-material iterative CT (ADMM), virtual monoenergetic imaging, K-edge subtraction, photon-counting spectral reconstruction, GPU splitting-based DECT ADMM.

This teaching project implements the first and most fundamental of these
(**projection-domain Newton decomposition**) end-to-end, plus **virtual
monoenergetic imaging (VMI)** as the downstream payoff. The others (ADMM,
photon-counting K-edge) are described in [THEORY.md](THEORY.md) → "Where this sits
in the real world".

- **Forward model** `f_e(t1,t2)` — the polychromatic Beer-Lambert law: measured
  log-attenuation as a nonlinear function of the two basis-material path lengths.
- **Newton's method** — solve `f_lo(t1,t2)=m_lo`, `f_hi(t1,t2)=m_hi` per bin,
  using an analytic 2×2 Jacobian inverted in closed form.
- **Linearised seed** — a cheap starting guess so Newton converges in ~5 steps.
- **Virtual monoenergetic imaging** — synthesize a single-energy attenuation from
  the recovered `(t1,t2)`.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/dual-energy-spectral-ct-reconstruction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/dual-energy-spectral-ct-reconstruction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\dual-energy-spectral-ct-reconstruction.sln /p:Configuration=Release /p:Platform=x64
```

Both `Debug|x64` and `Release|x64` build with zero warnings on the ratified
toolchain.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/dect_sinogram_sample.txt`, prints
the per-bin decomposition, shows the GPU-vs-CPU agreement check plus the recovery
error against the known synthetic truth, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/dect_sinogram_sample.txt` — 24 synthetic
  dual-energy sinogram bins with **known** ground-truth path lengths, so the demo
  runs offline and can report a real recovery error.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print pointers to real
  DECT/PCCT datasets (they never bypass registration).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: AAPM Spectral CT challenge datasets (verify URL at aapm.org); MARS photon-counting CT datasets (https://www.marsbioimaging.com/); TCIA DECT collections; simulated DECT from published XCAT phantom.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): a table
of the first 8 bins' recovered `(t1_water, t2_iodine)` and Newton iteration count,
one virtual-monoenergetic value, the max recovery error vs the known truth, and a
`RESULT: PASS` line. The program computes the decomposition on both the **GPU**
(`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and asserts
they agree within `1e-9 cm`. Because both call the **identical
`__host__ __device__` Newton core** (`src/dect.h`), they agree to ~machine
precision (~7e-15 cm) — that exact agreement is the correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
2. [`src/dect.h`](src/dect.h) — **the physics**: the polychromatic forward model,
   the analytic Jacobian, and one Newton step, as shared `__host__ __device__`
   functions. This is the heart of the project.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the one-thread-per-bin kernel and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the scanner-physics builder,
   the data loader, and the trusted serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

ASTRA (https://github.com/astra-toolbox/astra-toolbox) — multi-energy projection/backprojection primitives; TIGRE (https://github.com/CERN/TIGRE) — spectral CT reconstruction; ODL (https://github.com/odlgroup/odl) — material decomposition operators; splitting-based GPU DECT paper code (https://arxiv.org/abs/1905.00934 — verify repo link in paper).

- **ASTRA Toolbox** — GPU projection/backprojection primitives you would pair with
  this decomposition step to build a full image-domain pipeline.
- **TIGRE** — a GPU CBCT/spectral reconstruction toolbox; study how it structures
  multi-energy reconstruction.
- **ODL** — clean operator abstractions for material-decomposition inverse problems.
- **Alvarez & Macovski (1976)** — the original basis-material decomposition paper;
  the intellectual root of everything here.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent per-item jobs + constant-memory shared parameters** (see
`docs/PATTERNS.md §1`, exemplified by flagship `1.12`), combined with the **shared
`__host__ __device__` core** idiom (`§2`) for exact CPU/GPU parity. Concretely: a
custom kernel runs one thread per sinogram bin and solves the 2×2 Newton system in
registers; the scanner physics (both spectra + both attenuation curves) lives in
`__constant__` memory so the constant cache broadcasts it warp-wide. (The catalog
also lists cuFFT for a spectral filter and cuBLAS for joint iterative
reconstruction — those belong to the fuller image-domain/ADMM variant discussed in
THEORY, not this focused projection-domain teaching version.)

## Exercises

1. **Photon-counting extension.** Add a third and fourth energy channel (spectra)
   and a third basis material (a K-edge contrast like gadolinium). The per-bin
   solve becomes an over- or exactly-determined nonlinear least-squares — replace
   the 2×2 inverse with a Gauss-Newton step. How does the conditioning change?
2. **Add measurement noise.** Perturb `m_lo`, `m_hi` with Poisson-like noise in
   `make_synthetic.py` and watch the decomposition's variance explode where the
   two spectra are most similar (ill-conditioning). Plot recovery error vs. noise.
3. **Block-size sweep.** Try 64/128/256 threads per block and time the kernel at
   `--n 10000000`. This kernel is math/register-bound — where is the sweet spot on
   your card, and why?
4. **Virtual monoenergetic images.** Report VMI at 40 keV vs 100 keV for a
   contrast-carrying bin and confirm iodine contrast is boosted at low keV.
5. **FP32 vs FP64.** Switch the core to `float` and measure how the GPU-vs-CPU
   agreement and the recovery error degrade. This makes the ill-conditioning of
   material decomposition concrete (THEORY → "Numerical considerations").

## Limitations & honesty

- **Synthetic data.** Both the sinogram and the scanner physics are synthetic and
  labeled as such. The spectra are smooth analytic bumps (not Kramers/tube
  spectra) and the attenuation curves are simplified `~1/E³ + const` fits (not
  NIST XCOM cross-sections). They capture the *qualitative* physics that makes
  DECT work (different energy weighting, high-Z vs low-Z contrast) but are not
  quantitatively accurate. **Not for clinical use.**
- **Projection-domain only.** This project decomposes the *sinogram*; it does not
  reconstruct an image. A full pipeline would follow with filtered backprojection
  (see project 4.01) or an iterative image-domain solver (ADMM), which is where
  cuFFT/cuBLAS would enter.
- **Two materials, two spectra.** The real frontier (photon-counting CT, K-edge
  imaging) uses 4–8 energy bins and 3+ basis materials; that is described in
  THEORY but not implemented, per CLAUDE.md §13's reduced-scope teaching-version
  guidance.
- **Recovery error ~1e-6 cm** comes from storing the measurements at 6 decimals in
  the sample file; the *GPU-vs-CPU* agreement is ~1e-15 cm (machine precision).
  Both numbers are reported honestly by the demo.
