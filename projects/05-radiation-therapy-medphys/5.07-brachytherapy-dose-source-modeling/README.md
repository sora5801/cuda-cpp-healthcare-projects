# 5.7 — Brachytherapy Dose & Source Modeling

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🟢 Beginner · Established** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.7`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

This project computes a **brachytherapy dose map** using the clinical-standard
**AAPM TG-43** formalism, on the GPU. A radioactive source (here an Ir-192-like
line source) "dwells" at several positions inside the tumor; at every voxel of a
3-D grid we superpose each dwell's dose rate — an analytic product of a
geometry function, a tabulated radial dose function `g_L(r)`, and a tabulated 2-D
anisotropy function `F(r,θ)`. Every voxel is independent, so we give each one its
own GPU thread and keep the source tables in constant memory. The GPU result is
verified against a plain CPU reference that runs the *identical* math. All data
is **synthetic and educational** — not a real source and not for clinical use.

## What this computes & why the GPU helps

Brachytherapy (BT) delivers dose from radioactive sources (Ir-192 HDR, Pd-103, I-125) implanted inside or adjacent to the tumor. TG-43 formalism computes dose analytically from tabulated radial and anisotropy functions per source dwell position; for an HDR plan with 50 dwell positions in a prostate implant, GPU parallelization across (source, voxel) pairs reduces plan calculation from seconds to milliseconds. Beyond TG-43, model-based dose algorithms (MBDCA) — Acuros BT, Monte Carlo — account for tissue heterogeneity and inter-source shielding, requiring the same GPU particle-transport infrastructure as external-beam MC. Real-time BT dose visualization on TRUS/fluoroscopy feed requires GPU latency <100 ms.

**The parallel bottleneck:** the dose at each of `N_vox` voxels is an independent
sum over `N_dwell` dwell positions — `O(N_vox · N_dwell)` TG-43 evaluations. A
prostate HDR plan is ~`10⁷` voxels × ~`50` dwells ≈ `10⁹` evaluations, seconds on
a CPU. It parallelizes perfectly: **one thread per voxel**, each looping over the
(few) dwells, with the shared source tables broadcast from **constant memory**.

## The algorithm in brief

- **TG-43 dose formalism:** `Ḋ = S_K·Λ · [G_L/G_L,ref] · g_L(r) · F(r,θ)`.
- **Line-source geometry function** `G_L(r,θ)` (inverse-square corrected for the
  source's active length `L`; `L→0` recovers the point source `1/r²`).
- **Radial dose function** `g_L(r)` — 1-D linear interpolation of a measured table.
- **2-D anisotropy function** `F(r,θ)` — bilinear interpolation of a measured grid.
- **Superposition** of the per-dwell dose rates over the whole plan.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/brachytherapy-dose-source-modeling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/brachytherapy-dose-source-modeling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\brachytherapy-dose-source-modeling.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: AAPM TG-43 consensus datasets (radial/anisotropy tables — https://www.aapm.org/pubs/reports/); TCIA prostate BT CT datasets; ESTRO ACROP BT guideline test cases; BrachyView QA data (verify URL).

## Expected output

Success looks like `demo/expected_output.txt`:

```
5.7 -- Brachytherapy Dose & Source Modeling
[TG-43 analytic dose | SYNTHETIC teaching source, not clinical]
source: line L=0.35 cm  Lambda=1.1090 cGy/(h*U)  dwells=3
grid: 41 x 41 x 1 voxels @ 0.10 cm  (1681 voxels)
max dose = 98511.132812 cGy/h at voxel (20,20,0)
dose @ ~1cm transverse probe = 2.818497 cGy/h
center-row profile (8 samples): 0.766199 1.339740 3.383707 16.877193 29.011187 4.121541 1.526351 0.766199
RESULT: PASS (GPU matches CPU within rel-tol=1.0e-05)
```

The program computes the dose on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and asserts they agree within a `1e-5`
**relative** tolerance — that agreement is the correctness guarantee. Because both
paths call the *same* `tg43_physics.h` math, they match exactly (`max_rel_err = 0`).
The hottest voxel is the grid center (where the dwells sit), and the center-row
profile shows the symmetric `1/r²` falloff.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the plan, runs CPU + GPU, verifies, reports.
2. [`src/tg43_physics.h`](src/tg43_physics.h) — **the heart**: the shared
   `__host__ __device__` TG-43 formula (geometry, `g_L`, `F`) used by both sides.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the per-voxel kernel, constant-memory upload, host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the plan loader + trusted serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

BrachyDose (via EGSnrc, https://github.com/nrc-cnrc/EGSnrc) — EGSnrc BT MC user code; TOPAS-BrachyDose (https://github.com/topasmc) — Geant4-based BT MC; PyTG43 (https://github.com/GregSal/PyTG43 — verify URL) — Python TG-43 dose calculator; matRad BT module (https://github.com/e0404/matRad) — MATLAB BT dose and optimization.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent jobs + constant-memory tables** (PATTERNS.md §1, exemplar 1.12).
A custom kernel covers the output voxels with a grid of threads (one thread per
voxel); each thread's inner loop walks the source dwell positions and superposes
their TG-43 dose rate. The source's `g_L`/`F` tables and the dwell list live in
`__constant__` memory, so the reads a warp shares are broadcast from the constant
cache instead of replayed against global memory. The per-voxel physics is shared
with the CPU reference through a single `__host__ __device__` header
(`tg43_physics.h`, PATTERNS.md §2), which makes verification exact.

The catalog's note also mentions **cuRAND** (for the Monte-Carlo BT path — that is
projects 5.01/5.10, not TG-43), **texture memory** (a hardware-interpolation home
for `F(r,θ)`, deferred because its reduced precision would break bit-exact
verification), and **warp-level reduction** (only needed if you parallelize
*within* a voxel over thousands of dwells). See THEORY §4 for why each is or is
not used here.

## Exercises

1. **Point-source limit.** Set `L = 0` in `data/sample/plan_sample.txt` and
   confirm `G_L → 1/r²`. How does the on-axis dose change vs. the line source?
2. **Texture-memory anisotropy.** Move `F(r,θ)` into a 2-D `cudaTextureObject_t`
   and use hardware bilinear filtering. Measure the speedup, then measure how much
   `max_rel_err` grows — and explain why (hint: texture filter weights are 9-bit
   fixed point).
3. **FP32 core.** Add an FP32 variant of `tg43_physics.h` and compare the dose and
   the verification tolerance. Where does single precision first hurt (near the
   source? on-axis?)?
4. **Bigger plan.** Run `python scripts/make_synthetic.py --grid 201 --dwells 30`
   and watch the CPU/GPU timing gap widen — the GPU's edge grows with the grid.
5. **DVH.** Compute a dose-volume histogram over a spherical "target" and report
   the `D90` (dose covering 90% of the target volume), the core planning metric.

## Limitations & honesty

- **Synthetic data.** The source model (`Λ`, `g_L`, `F`) is a plausible *shape*,
  **not** an AAPM consensus dataset and not any real, named source. Everything is
  labeled synthetic; nothing here is clinically valid.
- **TG-43 assumes an infinite water phantom** — no bone, air, applicator, or
  inter-source shielding. Real heterogeneity needs MBDCA / Monte-Carlo
  (projects 5.01, 5.06, 5.10). TG-43 is the fast first pass those refine.
- **On-source dose is undefined.** We floor `r` to `1e-4 cm`, so the "max dose"
  voxel (right on a dwell) is an artifact, not a physical value.
- **Single shared long axis.** All dwells are assumed oriented along `+z`; real
  catheters curve and each dwell has its own tangent.
- **FP64 for teaching clarity.** A production kernel would likely use FP32 with a
  documented error budget; we chose FP64 so CPU==GPU verification is exact.
