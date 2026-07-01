# 4.17 — Real-Time Intraoperative / Image-Guided Surgery

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.17`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Image-guided surgery (IGS) overlays a surgeon's **pre-operative plan** (a tumour
or organ surface extracted from the planning MRI/CT) onto the **live patient** as
they operate. For that overlay to land in the right place, the two coordinate
frames must be **registered**: we need the rigid motion — a rotation `R` and a
translation `t` — that maps the pre-op surface onto points measured *during*
surgery (from a tracked pointer, ultrasound, or depth camera). This project
computes exactly that registration with the classic **Iterative Closest Point
(ICP)** algorithm, accelerated on the GPU. It aligns a *moving* point cloud to a
*fixed* one, verifies the GPU result against a CPU reference bit-for-bit, and
reports the shrinking alignment error in millimetres. It is a self-contained,
reduced-scope teaching slice of the much larger IGS pipeline.

## What this computes & why the GPU helps

Image-guided surgery fuses pre-operative MRI/CT with intra-operative imaging
(ultrasound, CBCT, fluorescence) to track instruments and tumour margins in real
time, under a hard latency budget (< ~1 s for an image/registration update). The
full pipeline has many GPU-accelerated stages — CBCT reconstruction, deformable
registration, instrument segmentation, DRR generation. **This project builds the
surface-registration stage: rigid ICP.**

**The parallel bottleneck:** ICP's cost is dominated by the **correspondence
step** — for each of the `|P|` moving points, find its nearest neighbour among
the `|Q|` fixed points. Brute force that is `O(|P|·|Q|)` per iteration, and real
surfaces have 10⁴–10⁶ points. Every point's search is **independent**, so we give
each moving point its own GPU thread. The follow-on step that turns those pairs
into a transform is a **reduction** (accumulate a 3×3 cross-covariance matrix),
which we also do on the GPU, in integer fixed-point for determinism. The tiny
3×3 SVD that extracts `R` from the covariance runs once per iteration on the host.

## The algorithm in brief

- **Coarse pre-alignment** — translate the moving cloud so its centroid matches
  the fixed cloud's. This puts ICP inside its convergence basin (a real,
  load-bearing step — without it ICP stalls at a poor local minimum).
- **ICP iteration** (repeat to convergence):
  1. **Correspond** — nearest fixed point for every (transformed) moving point.
  2. **Reduce** — accumulate centroids and the 3×3 cross-covariance `H` over all
     pairs (integer fixed-point atomics → deterministic).
  3. **Align** — SVD `H = U S Vᵀ`; `R = V·diag(1,1,det)·Uᵀ`; `t = mean(Q) − R·mean(P)`
     (the Kabsch/Arun/Horn closed form), with a reflection guard.
  4. **Compose** the increment onto the running transform.
- **Metric** — RMS nearest-neighbour distance (mm), which falls each iteration.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/real-time-intraoperative-image-guided-surgery.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/real-time-intraoperative-image-guided-surgery.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\real-time-intraoperative-image-guided-surgery.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/surface_pair.txt`, prints the
per-iteration RMS error and the recovered transform, shows the GPU-vs-CPU
agreement check, and prints a timing line on stderr.

## Data

- **Sample (committed):** `data/sample/surface_pair.txt` — a tiny **synthetic**
  pair of 3-D surfaces (36 moving + 36 fixed points) so the demo runs with zero
  downloads. Built by `scripts/make_synthetic.py`.
- **Full datasets:** `scripts/download_data.ps1` / `.sh` print pointers to real
  IGS corpora (they never bypass any registration).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: Cholec80 laparoscopic video dataset; ReMIND2Reg 2025 brain
resection multimodal dataset; EndoVis MICCAI challenge datasets; SurgT tool-tracking
benchmark. (All require registration/challenge sign-up; the demo uses synthetic
data instead.)

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): the RMS
alignment error drops from ~3.2 mm to **~0.24 mm** (the injected noise floor) in
one ICP iteration and holds flat, and the run ends with
`RESULT: PASS (GPU transform matches CPU reference)`. The program computes the
registration on both the **GPU** (`src/kernels.cu`) and a **CPU reference**
(`src/reference_cpu.cpp`) and asserts the two recovered transforms are identical
to ~1e-9 — in fact **bit-for-bit**, because the covariance reduction is integer
fixed-point (see THEORY §verification). That agreement is the correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the two clouds, runs CPU + GPU ICP,
   verifies, and prints the deterministic report.
2. [`src/icp.h`](src/icp.h) — **the shared `__host__ __device__` core**: nearest
   neighbour, the fixed-point covariance accumulators, the 3×3 SVD, `solve_rigid`,
   and `centroid_prealign`. Read this to understand the math both paths share.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the two-pattern idea.
4. [`src/kernels.cu`](src/kernels.cu) — the correspondence/reduction kernel and
   the host ICP driver (constant-memory transform, fixed-point atomics).
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial ICP.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, I/O helpers.

## Prior art & further reading

- **PLUS** (Public software Library for UltraSound imaging) —
  <https://github.com/PlusToolkit/PlusLib>. Study its real-time US acquisition and
  reconstruction, and how it streams tracked geometry.
- **3D Slicer** — <https://github.com/Slicer/Slicer>. The reference open IGS
  platform; study **OpenIGTLink** (the protocol that carries tracker/transform
  data) and its GPU-accelerated rendering.
- **NVIDIA Clara Holoscan** — <https://github.com/nvidia-holoscan/holoscan-sdk>.
  Study its low-latency GPU streaming pipeline for surgical video.
- **RTK** — <https://github.com/RTKConsortium/RTK>. Study intra-operative CBCT
  (FDK) reconstruction (see also this repo's flagship **4.01**).
- Classic ICP papers: Besl & McKay 1992 (ICP), Arun/Horn 1987 (the SVD rigid fit),
  Rusinkiewicz & Levoy 2001 ("Efficient variants of the ICP algorithm").

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent per-item jobs + an atomic reduction** — the same pattern as flagship
**11.09** (k-means): one thread per moving point does an independent nearest-
neighbour search, then `atomicAdd`s its contribution into shared fixed-point
accumulators (the 3×3 covariance + two centroids). The running transform lives in
**constant memory** (read by every thread, never changing during a launch — the
same idea as the constant-memory query in flagship **1.12**). Determinism comes
from **integer fixed-point atomics** (docs/PATTERNS.md §2–3). The catalog also
lists cuBLAS for the align solve; here the solve is a 3×3 SVD, small enough to
hand-roll clearly on the host (a full-size version would call cuSOLVER/cuBLAS —
see THEORY §real-world and flagship 2.06 for the cuSOLVER pattern).

## Exercises

1. **k-d tree correspondence.** Replace the brute-force `nearest_index` with a
   k-d tree (or a uniform voxel grid) to cut the search from `O(|Q|)` to
   `O(log|Q|)` per query. Measure the crossover point vs. brute force as `|Q|` grows.
2. **Point-to-plane ICP.** Swap the point-to-point error for point-to-plane
   (minimize distance to the local tangent plane using per-point normals). It
   converges in far fewer iterations on smooth surfaces — implement it and compare
   the RMS curves.
3. **Outlier rejection.** Add a distance threshold (or trimmed ICP) so spurious
   correspondences (occlusion, partial overlap) do not drag the fit. Regenerate
   the sample with `--noise 1.0` and a few gross outliers to see the effect.
4. **Scale the cloud.** Run `python scripts/make_synthetic.py --grid 60` (3600
   points) and watch the GPU/CPU timing gap widen — the point where the GPU's
   `O(|P||Q|)` parallelism pays off.
5. **FP64 everywhere.** The covariance is already double; try an all-double point
   type and quantify how much the fixed-point quantization actually cost.

## Limitations & honesty

- **Reduced scope.** Real IGS is a *pipeline* (CBCT reconstruction, deformable
  registration, instrument segmentation, tracking, rendering). This project builds
  only the **rigid surface-registration** stage — the cleanest, most self-contained
  piece — and describes the rest in THEORY §real-world.
- **Rigid only.** Tissue deforms (brain shift, breathing). Production systems add
  **deformable** registration (e.g. Demons, or biomechanical FEM). ICP here assumes
  the surfaces differ by a rigid motion plus noise.
- **Brute-force correspondence.** `O(|P||Q|)` is chosen for clarity, not speed; a
  real system uses a k-d tree / GPU spatial hash (Exercise 1).
- **Synthetic data.** The sample is generated, labeled synthetic everywhere, and
  models no real patient. The recovered transform is a teaching artifact.
- **Not for clinical use.** Nothing here is validated for diagnosis, treatment, or
  navigation.
