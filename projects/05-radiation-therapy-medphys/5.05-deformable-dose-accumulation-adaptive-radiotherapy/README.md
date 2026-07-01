# 5.5 — Deformable Dose Accumulation & Adaptive Radiotherapy

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.5`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Over a radiotherapy course the patient's anatomy changes day to day — the tumour shrinks, the bladder fills,
the lung breathes. **Adaptive radiotherapy (ART)** re-images the patient each fraction and adjusts. To know the
**total** dose actually delivered to a moving target you cannot just add each day's dose in place: you must
first **deformably register** today's anatomy to the planning anatomy (recovering a per-voxel displacement
field, the DVF), then **warp each fraction's dose through that DVF** into a common frame and sum — *summation
of deformed doses*. This project is a **reduced-scope 2-D teaching version** of that pipeline: GPU Thirion
Demons for the registration, a custom trilinear (here bilinear) warp for the dose, an integer-atomic dose-volume
histogram (DVH), and a CPU reference that the GPU is checked against exactly.

## What this computes & why the GPU helps

Adaptive radiotherapy (ART) adjusts the treatment plan during a course of fractions based on daily imaging (CBCT), requiring: (1) daily GPU CBCT reconstruction, (2) deformable image registration (DIR) between planning CT and daily image, (3) deformable warping of the dose distribution via the DVF to accumulate physically meaningful total dose. DIR and dose warping on a 512³ volume require iterative GPU Demons/B-spline followed by trilinear interpolation of the 3D DVF — each voxel's dose is mapped to its deformed position. Online ART workflows (MR-Linac) must complete all steps in <5 min, achievable only with GPU. Uncertainty in DIR propagates to dose uncertainty, motivating ensemble DIR and probabilistic dose accumulation on GPU.

**The parallel bottleneck:** the two dominant costs are both **per-voxel and embarrassingly parallel**.
(1) DIR runs ~100+ iterations, and *each* iteration touches every voxel three times (force + two Gaussian
passes) — O(iters · N · radius) work. (2) The dose warp/accumulate/histogram is O(N) per fraction. On a real
512³ volume that is ~10⁸ voxels × hundreds of iterations; the online-ART <5-minute budget is only reachable by
giving each voxel its own GPU thread. This demo runs a 64×64 version so it finishes instantly while exercising
exactly those kernels.

## The algorithm in brief

- **Thirion's Demons DIR** (`daily_img` → `plan_img`): per iteration, `force` (optical-flow update from the
  intensity mismatch and fixed-image gradient) → `Gaussian smooth-x` → `Gaussian smooth-y` (a diffusion
  regularizer), producing the DVF `u(x)`.
- **Deformable dose warp** (the catalog's "custom CUDA trilinear warp"): for each planning voxel, gather the
  delivered dose at the deformed position `x + u(x)` by bilinear interpolation.
- **Summation of deformed doses**: add each warped fraction into a running total in the planning frame.
- **Dose-volume histogram (DVH)**: bin every voxel's accumulated dose using **integer atomic adds**, so the
  parallel reduction is deterministic and matches the CPU exactly.
- **Rigid-vs-deformable contrast**: the demo also computes the naive no-DIR accumulation to show why the DVF
  matters.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/deformable-dose-accumulation-adaptive-radiotherapy.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/deformable-dose-accumulation-adaptive-radiotherapy.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\deformable-dose-accumulation-adaptive-radiotherapy.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/art_case.txt`, prints the accumulated-dose result and DVH,
shows the GPU-vs-CPU agreement check (DVF, dose, DVH), and prints a timing line.

## Data

- **Sample (committed):** `data/sample/art_case.txt` — four tiny 64×64 grids (planning image, daily image,
  planning dose, daily dose), fully synthetic, so the demo runs with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (print pointers to DIR-Lab / TG-132 / TCIA / CREATIS;
  they never bypass any registration).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: TCIA CT-on-rails / CBCT datasets; DIR-Lab 4D-CT lung dataset (https://www.dir-lab.com/); AAPM TG-132 DIR test cases; CREATIS deformable lung phantom (https://www.creatis.insa-lyon.fr/).

## Expected output

Success looks like `demo/expected_output.txt` (also shown in [demo/README.md](demo/README.md)). The program
computes the whole pipeline on both the **GPU** (`src/kernels.cu`) and a **CPU reference**
(`src/reference_cpu.cpp`) and asserts they agree: the DVF within 1e-3 px, the accumulated dose within 1e-9 Gy,
and the integer DVH **exactly**. That three-way agreement is the correctness guarantee.

## Code tour

Read in this order:

1. [`src/demons.h`](src/demons.h) — the shared `__host__ __device__` DIR physics (warp, gradient, Thirion
   force, separable Gaussian). Start here.
2. [`src/dose.h`](src/dose.h) — the shared dose-warp gather and DVH binning (the new physics of 5.5).
3. [`src/main.cu`](src/main.cu) — loads the case, runs CPU + GPU pipelines, verifies (3 checks), reports.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
5. [`src/kernels.cu`](src/kernels.cu) — the kernels (DIR + warp/accumulate/DVH) and the two host wrappers.
6. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline.
7. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

Plastimatch (https://plastimatch.org/) — GPU B-spline DIR + dose warping, DICOM-RT; VoxelMorph (https://github.com/voxelmorph/voxelmorph) — DL DIR for daily CBCT to CT; CERR (https://github.com/cerr/CERR) — deformable dose accumulation pipeline; pyRadPlan (https://github.com/e0404/pyRadPlan) — adaptive plan re-optimization.

Study these to learn the production approach; **do not copy code wholesale** — reimplement didactically and
credit the source (CLAUDE.md §2). In particular, Plastimatch and CERR implement exactly the DIR → dose-warp →
accumulate pipeline this project miniaturizes; AAPM **TG-132** is the clinical QA reference for it.

## CUDA pattern used here

Two patterns chained (PATTERNS.md §1): **per-voxel gather + stencil** (DIR: one thread per voxel; force is a
gather, the separable Gaussian is a double-buffered stencil) and **per-voxel gather + atomic reduce** (dose:
the trilinear/bilinear warp is a race-free gather; the DVH is an integer-atomic histogram). The per-voxel
physics is shared between CPU and GPU through `demons.h` / `dose.h` (`__host__ __device__` idiom), so the GPU
matches the CPU exactly. This teaching version implements the Gaussian smoothing as a **hand-written separable
stencil** rather than the cuFFT convolution the catalog mentions, and the DVH as **integer atomics** rather
than a float histogram — both deliberate choices that keep the result deterministic and legible (see THEORY §5
for why, and how a production tool differs).

## Exercises

1. **Make the motion matter.** Regenerate with `python scripts/make_synthetic.py --shift 12.0 --dose-sigma 6`
   and watch the `deformable − rigid` hot-spot gap grow. Explain why a steeper dose gradient amplifies the
   effect of the DVF.
2. **Shared-memory Gaussian.** The `gauss_x/y` kernels re-read each neighbour from global memory. Tile a block
   of the field into `__shared__` memory with a halo and reuse it across the block (cf. project 7.10). Measure
   the speed-up.
3. **Energy/mass-transfer accumulation.** The interpolation method here does not conserve energy under
   compression. Implement the alternative *push*/energy-transfer warp (deposit each source voxel's energy into
   its deformed neighbours with `atomicAdd`) and compare total accumulated energy to the interpolation method.
4. **Inverse-consistency check.** Register `plan → daily` as well and verify `u_forward(x) ≈ −u_inverse(x+u)`;
   report the residual as a DIR-quality metric (this is a real ART QA step, TG-132).
5. **Ensemble DIR uncertainty.** Run the DIR several times with slightly perturbed `sigma`, warp the dose with
   each DVF, and report the per-voxel dose standard deviation — a probabilistic accumulated dose (catalog).

## Limitations & honesty

- **Reduced scope, on purpose.** This is a **2-D** pipeline on a tiny **synthetic** case, not clinical software.
  Real ART is 3-D (256³–512³), reads DICOM-RT, and handles multi-modality (CBCT↔CT) intensity differences that
  plain SSD-Demons does not.
- **Demons, not diffeomorphic/B-spline.** The catalog lists diffeomorphic Demons and B-spline FFD; we use the
  classic (non-diffeomorphic) Demons, which can in principle fold the field. The Gaussian regularizer keeps our
  smooth synthetic case well-behaved; a production solver enforces invertibility (see THEORY §7).
- **Interpolation accumulation only.** We implement summation-of-deformed-doses by trilinear interpolation. The
  energy/mass-transfer method (which conserves energy under tissue compression/expansion) is described in THEORY
  §7 and left as Exercise 3.
- **Synthetic data.** Every grid in `data/sample/` is generated, labeled synthetic, and carries **no** clinical
  meaning. Nothing here may inform a real treatment decision.
