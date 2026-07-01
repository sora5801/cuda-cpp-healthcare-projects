# 4.32 — GPU-Accelerated Landmark Detection

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.32`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A landmark-detection network does not emit coordinates directly — it emits, for
each anatomical landmark, a whole 3D **heatmap** (a "probability blob" that peaks
at the landmark). This project implements the **decode step** that turns those
heatmaps into coordinates: a GPU kernel that, for each landmark, finds the peak
voxel (**argmax**) and refines it to a **sub-voxel** position (**soft-argmax**,
an intensity-weighted centroid). It runs the decode on both the GPU and a plain
CPU reference and proves they agree exactly, then reports how well the decoded
points recover a known synthetic ground truth. It teaches two core CUDA idioms —
a block-level **reduction** and **deterministic fixed-point atomics** — on a real
medical-imaging task.

## What this computes & why the GPU helps

Anatomical landmark detection localizes clinically relevant points (vertebral endplates, femoral head centers, dental cusps) in 3D medical images for registration initialization, measurement, and surgical planning. Deep learning heatmap regression (stacked hourglass, U-Net with Gaussian target maps) predicts a 3D heatmap per landmark; for 100 landmarks in a 512³ CT, the output tensor is 100 × 512³ ~ 13 GB requiring GPU. Reinforcement learning landmark detection (DQN, MARL — multi-agent RL) has each agent navigate the volume independently, with GPU parallelizing all agents simultaneously. GPU is essential for training on large 3D datasets and for achieving clinical inference speeds.

**The parallel bottleneck:** decoding a landmark means scanning its *entire*
heatmap volume for the peak — `O(V)` voxel reads per landmark, `V = 1.3 × 10⁸`
for a 512³ grid, times up to 100+ landmarks. This full-volume **argmax reduction**
is memory-bandwidth bound and embarrassingly parallel, both across landmarks and
across voxels. We map **one thread block per landmark**, and its 256 threads
stream the volume and reduce to the peak; a short soft-argmax window then refines
it. (Training the network that *produces* the heatmaps is the other GPU-heavy
half — out of scope here; see THEORY §7.)

## The algorithm in brief

- **Argmax** — per landmark, the integer voxel with the maximum heatmap value
  (coarse localization); deterministic tie-break by lowest row-major index.
- **Soft-argmax** — the intensity-weighted centroid over a `(2R+1)³` window around
  the peak, giving a **sub-voxel** coordinate the integer grid cannot represent.
- **Fixed-point reduction** — the centroid's weight sums are accumulated in
  integers so the parallel `atomicAdd`s are order-independent → deterministic and
  bit-exact against the CPU.
- (Context: these decode the output of a stacked-hourglass / 3D U-Net heatmap
  regressor; cascade coarse-to-fine and anatomy-guided priors sit on top in
  production — see THEORY.)

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/gpu-accelerated-landmark-detection.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/gpu-accelerated-landmark-detection.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\gpu-accelerated-landmark-detection.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: VerSe vertebral challenge (https://github.com/anjany/verse) — 374 CT scans with 26 vertebral landmarks; RSNA Vertebral Fracture Detection (https://rsna-vertebral-labeling-level-detection.grand-challenge.org/); CephaloNet cephalometric landmark dataset; MICCAI 2015 prostate challenge landmark dataset.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): one
line per landmark giving the integer peak voxel, the sub-voxel coordinate, and
the recovery error vs the planted ground truth, ending in `RESULT: PASS`. The
program decodes on both the **GPU** (`src/kernels.cu`) and a **CPU reference**
(`src/reference_cpu.cpp`) and asserts they agree — integer peaks *exactly*, sub-
voxel coordinates within `1e-9` (only the final division can differ; see THEORY
§5–6). That agreement is the correctness guarantee. Timing goes to **stderr** (a
teaching artifact, not a benchmark) and is not part of the diffed output.

## Code tour

Read in this order:

1. [`src/landmark.h`](src/landmark.h) — the shared `__host__ __device__` decode
   math (grid indexing, fixed-point weights, the centroid division) used
   identically by CPU and GPU. Start here — it defines the "physics".
2. [`src/main.cu`](src/main.cu) — loads heatmaps, runs CPU + GPU, verifies, reports.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-block-per-
   landmark, two-phase idea.
4. [`src/kernels.cu`](src/kernels.cu) — the decode kernel (argmax tree-reduction +
   fixed-point soft-argmax atomics) and the host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline
   and the sample loader.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

MONAI (https://github.com/Project-MONAI/MONAI) — landmark detection transforms; nnDetection (https://github.com/MIC-DKFZ/nnDetection) — GPU object/landmark detection framework; VertXNet vertebral landmark (search GitHub — verify URL); MARL landmark detection (https://github.com/amiralansari/marl-landmark — verify URL).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**One independent reduction per landmark:** `grid = L` blocks (one landmark
each), `block = 256` threads that cooperate on that landmark's heatmap. Phase 1
is a shared-memory **tree reduction** for the argmax; phase 2 uses **block-scoped
integer `atomicAdd`** into shared accumulators for a deterministic soft-argmax
(see `docs/PATTERNS.md`, cross of the `1.12` per-item and `11.09` atomic-reduce
patterns). No external CUDA library is needed for the decode. In the *full*
pipeline the catalog's libraries apply to the parts we omit: **cuDNN** + **Tensor
Cores** for the 3D hourglass/U-Net convolutions, **cuBLAS** for a regression
head, and GPU-resident augmentation (elastic deformation, Gaussian blur) — all
covered in [THEORY.md](THEORY.md) §7.

## Exercises

1. **Bigger volumes.** Generate a 64³ set with 26 landmarks
   (`python scripts/make_synthetic.py --nx 64 --ny 64 --nz 64 --landmarks 26`)
   and watch the GPU/CPU timing gap widen as `V` grows.
2. **Warp-shuffle reduction.** Replace the shared-memory tree reduction in
   `argmax_reduce` with `__shfl_down_sync` within each warp, then a final
   cross-warp step. Measure the difference; confirm the result is unchanged.
3. **Parabolic sub-voxel fit.** Swap soft-argmax for a 1-D parabola fit through
   the peak and its two neighbours per axis. Compare recovery error to the
   centroid — which is more accurate for a Gaussian blob, and why?
4. **Tune the window radius.** Sweep `SOFTARGMAX_RADIUS` (1, 2, 3). A larger
   window captures more of the blob but also more background — plot recovery
   error vs radius.
5. **Non-maximum suppression.** Extend the decoder to return the *top-K* peaks
   per heatmap (for multi-instance landmarks), suppressing peaks within a radius
   of a stronger one.

## Limitations & honesty

- **Reduced scope (decode only).** This project implements the inference-time
  *decode*, not the network that produces the heatmaps. Training a 3D U-Net
  (cuDNN, Tensor Cores, a data pipeline) is deliberately out of scope; THEORY §7
  describes the full system.
- **Synthetic data.** The heatmaps are Gaussian blobs at known centres, generated
  by `scripts/make_synthetic.py` — labeled synthetic everywhere. Real network
  outputs are noisier, can be multi-modal, and may miss landmarks entirely.
- **Soft-argmax bias.** The centroid over a finite window with Gaussian tails is
  slightly biased (the demo's ~0.2-voxel recovery error is honest, not a bug); a
  parabolic fit or larger radius reduces it (Exercises 3–4).
- **Timing is a teaching artifact.** On this tiny sample the GPU is *slower* than
  the CPU — launch and copy overhead dominate. The GPU's edge appears only at
  realistic volume sizes; the printed milliseconds are illustrative, never a
  benchmark claim.
- **Not for clinical use.** Educational only. No output here may inform diagnosis,
  measurement, or treatment.
