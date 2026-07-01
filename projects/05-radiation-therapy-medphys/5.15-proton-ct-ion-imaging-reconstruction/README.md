# 5.15 — Proton CT & Ion Imaging Reconstruction

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.15`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Proton CT (pCT) images the **relative stopping power (RSP)** of tissue by sending
protons *through* a patient and measuring how much each one slows down. This
project reconstructs a 2-D RSP map from a list of individual proton histories
using **SART** (an iterative algebraic reconstruction), following each proton
along its curved **most-likely path (MLP)**. It is a deliberately small, 2-D
**teaching version** of a frontier problem: the whole pipeline — MLP geometry,
forward/backprojection, the iterative solver, and a deterministic GPU reduction —
in a few hundred heavily-commented lines you can read end to end.

## What this computes & why the GPU helps

Proton CT (pCT) measures the residual range of individual protons after
traversing a patient, converting to relative stopping power (RSP) maps directly
for treatment planning — eliminating the Hounsfield-unit-to-RSP conversion
uncertainty (~3%) baked into X-ray CT. Each proton's path through tissue is a
curved **most-likely path (MLP)**, not a straight line, because multiple Coulomb
scattering bends it. Reconstruction forward- and backprojects along those curved
paths — fundamentally different from X-ray cone-beam CT.

**The parallel bottleneck:** the reconstruction repeatedly (a) walks every
proton's MLP through the current RSP image to predict its water-equivalent path
length, and (b) scatters a correction back along that path. A clinical scan is
~**10⁸ protons**, and within each SART sweep the protons are **independent** — so
we assign **one GPU thread per proton**. That is where essentially all the work
is, and it parallelises perfectly (with an atomic reduction where paths overlap).

## The algorithm in brief

- **Most-likely path (MLP)** — a cubic-Hermite curve matching each proton's
  measured entry/exit position and angle (the small-angle limit of the Highland /
  Gaussian-scattering MLP).
- **Forward projection** — integrate RSP along the MLP → predicted WEPL.
- **SART** (Simultaneous Algebraic Reconstruction) — per sweep, form each
  proton's WEPL residual and scatter a length-weighted correction into per-voxel
  accumulators, then update `RSP += λ·num/den`.
- **Deterministic reduction** — accumulate corrections in fixed-point integers so
  the many-writer GPU tally is order-independent and matches the CPU exactly.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/proton-ct-ion-imaging-reconstruction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/proton-ct-ion-imaging-reconstruction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\proton-ct-ion-imaging-reconstruction.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/protons_sample.txt`, prints the
reconstructed RSP probes and recovery metrics, shows the GPU-vs-CPU agreement
check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/protons_sample.txt` — 1440 synthetic
  proton histories through a known RSP phantom (water disc + dense + light
  inserts) plus the ground-truth grid; runs offline with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print how to obtain or
  simulate real pCT data (TOPAS/GATE, PRaVDA/PRIMA); they download nothing and
  never bypass registration.
- **Provenance & license:** see [data/README.md](data/README.md). The sample is
  **synthetic** and labeled as such.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program reconstructs the RSP map on both the **GPU** (`src/kernels.cu`) and a
**CPU reference** (`src/reference_cpu.cpp`) and asserts they agree within
`1.0e-3` RSP units (observed error ~`1e-6`, printed on stderr) — that agreement
is the correctness guarantee. It also reports **RMSE vs. the ground-truth
phantom** and a central-row profile that peaks through the dense insert, showing
the reconstruction recovered the known object.

## Code tour

Read in this order:

1. [`src/pct_physics.h`](src/pct_physics.h) — the shared `__host__ __device__`
   physics: the `Proton`/`PctGeom` types and the MLP curve. Start here.
2. [`src/reference_cpu.h`](src/reference_cpu.h) / [`.cpp`](src/reference_cpu.cpp)
   — the problem, the loader, and the trusted serial SART baseline.
3. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-
   proton idea.
5. [`src/kernels.cu`](src/kernels.cu) — the tally/update kernels + host driver.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **Schulte et al. 2008** (*Med. Phys.*) — the maximum-likelihood proton path
  formalism; our cubic-Hermite MLP is its small-angle limit.
- **TOPAS** (<https://github.com/OpenTOPAS/OpenTOPAS>) / **GATE** — Monte-Carlo
  toolkits that simulate a pCT scan; use them to *generate* realistic list-mode
  data in this project's format.
- **FRED** (<https://www.fredonline.eu/>) — fast GPU proton transport / range
  imaging; study how a production GPU projector is structured.
- **UCSC/Baylor pCT** reconstruction codes and "proton CT GPU most likely path"
  repos — custom CUDA MLP projection kernels.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**One CUDA thread per detected proton**: each thread walks its MLP (a **gather**
from the RSP image, like CT backprojection 4.01) and scatters a correction with
**`atomicAdd` into fixed-point integer accumulators** (a deterministic
many-writer reduction, like Monte-Carlo dose 5.01 and k-means 11.09). A second
one-thread-per-voxel kernel applies the SART update. The RSP image persists on
the device across all sweeps. (The catalog also mentions cuRAND / Thrust sort /
cuBLAS; THEORY §4 explains exactly where each would go and why this reduced-scope
version does not need them.)

## Exercises

1. **Non-negativity (POCS).** After each sweep, clamp `RSP ≥ 0` in
   `update_kernel`. Watch the edge ringing (the small negative values in the
   profile) disappear. How does RMSE change?
2. **Convergence study.** Regenerate with more sweeps
   (`make_synthetic.py --iters 80`) and plot RMSE vs. iteration. Where does it
   plateau? Try relaxation `λ ∈ {0.3, 0.8, 1.0}`.
3. **Better projector.** Replace nearest-voxel binning with **bilinear**
   splatting (distribute each sample over the 4 nearest voxels). Update *both*
   `reference_cpu.cpp` and `kernels.cu` so they still match.
4. **Straight-line ablation.** Force `entry_angle = exit_angle = 0` (straight
   paths) and compare RMSE — quantify how much the MLP curvature actually helps.
5. **Texture memory.** Put the RSP image in a CUDA **texture** and use hardware
   bilinear interpolation in the forward projection (cf. 4.01). Measure the
   speedup.

## Limitations & honesty

- **Reduced scope.** This is a **2-D single slice** with a **closed-form cubic
  MLP**, a **nearest-voxel projector**, and **no constraints/regularisation** —
  a teaching version. Production pCT is 3-D, uses the full covariance MLP,
  bilinear/Siddon projectors, POCS box constraints, and TV/scattering
  regularisation (THEORY §7).
- **Synthetic data.** The committed sample is **synthetic**, generated from a
  known phantom whose WEPLs are computed with the *same* forward model the solver
  uses — so it is self-consistent by construction. Real pCT data has range
  straggling, detector noise, and secondary particles this demo omits.
- **Determinism caveat.** The fixed-point tally is exact and order-independent;
  the float forward-projection can still differ by ~1 ULP between host and
  device, so we verify to a small *physical* tolerance, not bit-equality (THEORY
  §5). Timings are teaching artifacts, not benchmarks.
- **Not clinical.** RSP values are a software demonstration; nothing here may
  inform diagnosis, treatment, or planning.
