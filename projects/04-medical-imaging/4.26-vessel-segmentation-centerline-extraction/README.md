# 4.26 — Vessel Segmentation & Centerline Extraction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.26`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Blood vessels in a 3-D CT-angiography (CTA) volume look like bright **tubes** on a
darker background. This project implements the **Frangi vesselness filter** — the
classic way to turn "how tube-like is the neighbourhood of each voxel?" into a
score in `[0, 1]` — and runs it on the GPU with **one thread per voxel**. For each
voxel we build the local **Hessian** (the 3×3 matrix of second derivatives),
compute its three **eigenvalues**, and combine them into a vesselness score:
high on vessels, ~0 in flat tissue. Thresholding the score gives a first-pass
segmentation, and its peak along a vessel is a centerline seed. The GPU result is
verified against a plain-C++ reference that runs the **identical** per-voxel math,
so the two agree **exactly** on the demo.

## What this computes & why the GPU helps

Vascular tree segmentation in CTA detects tubular structures as small as 1–2 mm in
noisy 3-D volumes. The workhorse is the **Hessian-based vesselness filter**
(Frangi 1998): it computes a **symmetric 3×3 eigenvalue decomposition per voxel**
— roughly **10⁶–10⁸ eigendecompositions** for a clinical CTA. Downstream steps
(3-D U-Net segmentation, fast-marching centerlines) also lean on the GPU.

**The parallel bottleneck this project targets:** the per-voxel eigendecomposition.
Every voxel's Hessian and eigenvalues depend only on its local 3×3×3
neighbourhood, so the voxels are **fully independent** — a textbook "map" (one GPU
thread computes one voxel, no communication, no atomics). This is where the GPU's
throughput on ~10⁷ independent small linear-algebra problems dominates a serial CPU
loop; the rest of a clinical pipeline (network inference, path finding) is out of
scope for this teaching version (see THEORY §7).

## The algorithm in brief

- **Gaussian pre-smoothing** at scale `σ` (separable 1-D passes) so the filter
  responds to vessels of ~that radius.
- **Hessian by finite differences** — the 6 unique second derivatives per voxel.
- **Closed-form symmetric 3×3 eigenvalues** (Cardano / trigonometric formula) —
  deterministic, no iteration, so CPU and GPU match to ~1e-9.
- **Frangi vesselness** from the eigenvalue ratios `R_A`, `R_B` and the
  structureness `S`, with a sign gate for bright-on-dark vessels.
- **Segmentation + centerline seed** — threshold the score; report the peak voxel.

Catalog also lists multi-scale vesselness, 3-D U-Net / V-Net, and fast-marching
centerlines; those are discussed as the real-world pipeline in
[THEORY.md](THEORY.md), which has the full science → math → algorithm →
GPU-mapping derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/vessel-segmentation-centerline-extraction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/vessel-segmentation-centerline-extraction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\vessel-segmentation-centerline-extraction.sln /p:Configuration=Release /p:Platform=x64
```

This project links only the CUDA runtime (`cudart_static.lib`) — no extra CUDA
libraries, since the eigendecomposition is hand-rolled on purpose (that is the
lesson).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/vessel_volume.txt`, prints the
peak-vesselness voxel + segmented voxel count + the across-vessel response profile,
shows the GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/vessel_volume.txt` — a tiny 24×16×16
  synthetic volume with one embedded bright vessel, so the demo runs offline with
  zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print links/instructions
  (the real sets need registration; the scripts never bypass it).
- **Provenance & license:** see [data/README.md](data/README.md).

Real datasets: ASOCA (<https://asoca.grand-challenge.org/>), ImageCAS
(<https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT>),
3D-IRCADb-01 (<https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/>).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The key
line is the **across-vessel profile** — a clean single-peak ridge
(`… 0.16 0.63 0.63 0.16 …`) that is zero away from the tube, i.e. the filter has
localized the vessel. The program computes the vesselness on both the **GPU**
(`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and asserts
they agree; here the max difference is **`0.000e+00`** (they run identical math),
so `RESULT: PASS`.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the volume, smooths it, runs CPU + GPU,
   verifies, and prints the deterministic report.
2. [`src/frangi.h`](src/frangi.h) — **the heart**: the shared `__host__ __device__`
   per-voxel math (finite-difference Hessian, closed-form 3×3 eigenvalues, Frangi
   score). Read this to understand *what* is being computed.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the serial pipeline (loader,
   separable Gaussian smooth, per-voxel loop, deterministic summary).
4. [`src/kernels.cuh`](src/kernels.cuh) → [`src/kernels.cu`](src/kernels.cu) — the
   GPU twin: one thread per voxel calling the same `frangi.h` functions.
5. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, host I/O.

## Prior art & further reading

- **VMTK** (Vascular Modeling Toolkit, <https://github.com/vmtk/vmtk>) — the
  reference open-source toolkit for centerline extraction and vascular meshing.
  Study how it turns a segmentation into a **graph** of centerlines with radii.
- **SlicerVMTK** (<https://github.com/vmtk/SlicerExtension-VMTK>) — VMTK inside
  3D Slicer; good for seeing the interactive clinical workflow.
- **MONAI** (<https://github.com/Project-MONAI/MONAI>) — 3-D vessel-segmentation
  networks (U-Net/V-Net); study how learned segmentation complements the classic
  Hessian filter (the filter is a strong hand-crafted prior / preprocessing step).
- **nnDetection** (<https://github.com/MIC-DKFZ/nnDetection>) — GPU object
  detection framing for tubular structures.

Study these for the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

The **map** pattern (docs/PATTERNS.md §1, exemplified by the stencil/per-element
flagships): one GPU thread per voxel, a 3-D grid of 3-D blocks over the volume, no
atomics or shared memory in the teaching version. The per-voxel physics lives in a
single shared `__host__ __device__` header (`frangi.h`, PATTERNS.md §2) so the CPU
reference and the GPU kernel are byte-for-byte identical. The catalog also mentions
Jacobi iteration, cuDNN, and Thrust priority queues; this project uses the
**closed-form** eigensolver (deterministic, exact) instead of Jacobi and leaves the
learned/graph steps to THEORY §7.

## Exercises

1. **Move the smoothing onto the GPU.** Today the separable Gaussian runs on the
   host; write three 1-D convolution kernels (with shared-memory halos) and keep
   the whole pipeline on the device.
2. **Multi-scale vesselness.** Run the filter at several `σ` values and take the
   per-voxel **max** response — the standard way to catch vessels of different
   radii. Add a `--scales` list to `make_synthetic.py`'s companion.
3. **Shared-memory tiling.** The map kernel re-reads each neighbour up to 27×;
   stage a block's tile+halo into shared memory and measure the bandwidth win.
4. **A real converter.** Write a small Python tool that reads a NIfTI CTA volume
   (via `nibabel`), crops a region, and writes this project's text format, so the
   filter runs on ASOCA/ImageCAS data.
5. **Eigenvectors + direction.** Extend `frangi.h` to also return the eigenvector
   of the smallest-magnitude eigenvalue (the vessel direction) and trace a crude
   centerline by walking along it from the peak seed.

## Limitations & honesty

- The committed volume is **synthetic** — one straight bright tube plus noise. It
  is engineered so the result is verifiable; it is **not** a real angiogram.
- This is **single-scale** Frangi; real pipelines are multi-scale and often follow
  with a learned network. The "centerline" here is only the **peak-response voxel**
  (a seed), not the graph VMTK produces.
- The peak voxel in the demo sits at the `x = 23` volume boundary: clamp-to-edge
  borders slightly inflate the response at the tube's ends. This is an honest
  finite-difference edge effect, not a bug (see THEORY §5); a larger `nx` or edge
  masking moves the peak into the interior.
- Timings are a **teaching artifact, not a benchmark**: on this tiny volume the
  kernel is launch/copy-bound. The GPU's edge grows with clinical-size volumes.
- **Not for clinical use.** No diagnostic or therapeutic claim is made or implied.
