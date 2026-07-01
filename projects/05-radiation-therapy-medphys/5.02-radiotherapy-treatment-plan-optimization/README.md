# 5.2 — Radiotherapy Treatment-Plan Optimization

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.2`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._


## Summary

This project builds a small, fully-commented **fluence-map optimizer** — the
numerical engine behind IMRT/VMAT radiotherapy planning. Given a sparse
**dose-influence matrix** `D` (how much dose each beamlet deposits in each voxel)
and a clinical objective (hit the tumor at the prescription dose, spare the
organ-at-risk), it finds the beamlet intensities `x ≥ 0` that best trade off tumor
coverage against organ sparing. The optimizer is **projected gradient descent**,
and every iteration is dominated by two **sparse matrix–vector products** (`D x`
and `Dᵀ r`) run on the GPU with **cuSPARSE**. It is the canonical "big sparse
matrix, resident on the GPU, multiplied over and over" pattern — and it teaches
both a real inverse-planning workflow and how to use a CUDA library without
treating it as a black box.

## What this computes & why the GPU helps

IMRT/VMAT plan optimization solves a large-scale constrained optimization: minimize dose to OARs subject to PTV coverage constraints, with variables being beam aperture shapes or fluence maps. The dose-influence matrix D (N_voxels × N_beamlets, typically 10⁶ × 10⁴) must be computed and stored on GPU; the iterative optimizer (gradient descent, IPOPT, L-BFGS) performs repeated sparse matrix-vector products (D·x) per iteration. GPU SpMV reduces each DMAT-vector product from seconds to milliseconds, enabling real-time adaptive re-optimization. Biological-effect optimization (TCP/NTCP) and robust optimization over uncertainty scenarios further multiply the compute by the number of scenarios (~50–100 for robust plans).

**The parallel bottleneck:** each optimizer iteration recomputes the dose
`d = D x` (forward SpMV) and the gradient `∇F = Dᵀ r` (transpose SpMV). With
`D` being `n_vox × n_beam` (~`10^6 × 10^4`, sparse), these two SpMVs are ~100% of
the runtime; everything else (the per-voxel residual, the fluence update) is cheap
element-wise math. Keeping `D` resident on the GPU in **CSR** format and running
the SpMVs with **cuSPARSE** turns each iteration from seconds to milliseconds,
which is what makes real-time *adaptive* re-planning possible. See
[THEORY.md §4](THEORY.md) for the GPU mapping.

## The algorithm in brief

- **Fluence-map optimization (FMO)** as a convex quadratic problem: minimize
  `F(x) = Σ_v w_v · pen_v(d_v)` over `x ≥ 0`, with `d = D x`.
- **Per-structure penalties:** two-sided quadratic for the PTV (tumor), one-sided
  (over-dose only) for the OAR (organ) and BODY.
- **Projected gradient descent:** `x ← max(0, x − η·Dᵀr)` each step, where the
  residual `r_v = w_v·pen_v'(d_v)`.
- **CSR sparse storage** of `D`, resident on the GPU.
- **cuSPARSE `cusparseSpMV`** for both `D x` (non-transpose) and `Dᵀ r`
  (transpose); two tiny hand-written kernels for the element-wise residual and
  projected update.

The broader field also uses L-BFGS/IPOPT, direct aperture optimization (DAO),
VMAT, robust (minimax) and biological (TCP/NTCP) optimization, and deep-learning
dose prediction — all built on the same SpMV core (see [THEORY.md §7](THEORY.md)).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/radiotherapy-treatment-plan-optimization.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/radiotherapy-treatment-plan-optimization.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\radiotherapy-treatment-plan-optimization.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: OpenKBP (knowledge-based planning) dataset (https://github.com/ababier/open-kbp) — 340 head-and-neck IMRT plans; TCIA RT datasets; PlanIQ (verify URL); AAPM TG-263 structure naming dataset; OpenTPS test datasets.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
5.2 -- Radiotherapy Treatment-Plan Optimization
Fluence-map optimization: 48 voxels (PTV 7, OAR 6, BODY 35), 16 beamlets, nnz=178
optimizer: projected gradient, 400 iters, step=0.020, Rx=60.0 Gy
final objective F(x) = 253.8840
PTV dose (Gy): mean 59.244  min 55.221  max 62.787  homogeneity 0.1277
OAR dose (Gy): mean 10.955  max 29.164  (tolerance-limited sparing)
RESULT: PASS (GPU plan matches CPU within dose tol=1.0e-02 Gy)
```

Read it as a plan-quality report: the tumor (**PTV**) is driven to **mean 59.2 Gy**
against the 60 Gy prescription with a homogeneity index ≈ 0.13, while the organ
(**OAR**) is held to ~11 Gy mean — the optimizer *learned* to aim intensity at the
tumor and pull it off the organ. The program computes the result on both the
**GPU** (`src/kernels.cu`, cuSPARSE) and a **CPU reference**
(`src/reference_cpu.cpp`) and asserts the two plans agree within a documented dose
tolerance (`1e-2 Gy`) — that agreement is the correctness guarantee. Timings and
the measured GPU-vs-CPU error print to **stderr** (they vary run to run, so the
demo shows but does not diff them). See [THEORY.md §5](THEORY.md) for why the
tolerance is a small *physical* one rather than bit-equality.

## Code tour

Read in this order:

1. [`src/fmo.h`](src/fmo.h) — the shared `__host__ __device__` core: the
   per-voxel penalty, residual, and non-negativity projection. Both CPU and GPU
   call these, so their scalar math is identical.
2. [`src/reference_cpu.h`](src/reference_cpu.h) — the CSR `Problem` struct, the
   `PlanStats` report, and the reference prototypes.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline:
   the loader, CSR SpMV / transpose-SpMV, DVH stats, and `optimize_cpu()`.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the cuSPARSE big idea.
5. [`src/kernels.cu`](src/kernels.cu) — the cuSPARSE SpMV plumbing and the two
   element-wise kernels (residual, projected update).
6. [`src/main.cu`](src/main.cu) — loads the plan, runs CPU + GPU, verifies, reports.
7. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

matRad (https://github.com/e0404/matRad) — open-source MATLAB treatment planning, photon/proton/carbon; pyRadPlan (https://github.com/e0404/pyRadPlan) — Python interoperable extension of matRad; CERR (https://github.com/cerr/CERR) — MATLAB comprehensive RT research platform with DICOM-RT; OpenTPS (https://opentps.org/) — open-source Python/GPU treatment planning system (verify URL).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Library SpMV on a GPU-resident CSR matrix** (PATTERNS.md §5, exemplified by the
cuSOLVER/cuBLAS flagships). The dose-influence matrix `D` is uploaded once in CSR
form and stays on the device; every iteration multiplies it twice with
**cuSPARSE `cusparseSpMV`** — `NON_TRANSPOSE` for `d = D x`, `TRANSPOSE` for
`g = Dᵀ r`. Two small hand-written kernels (`residual_kernel`, `update_kernel`)
do the element-wise steps, reusing the shared `__host__ __device__` math from
`fmo.h`. The broader catalog note (cuBLAS for DVHs, warp reductions, multi-GPU
for robust scenarios) points at how a production planner scales this further.

## Exercises

1. **Scale it up.** Regenerate a bigger problem with
   `python scripts/make_synthetic.py --n-vox 4096 --n-beam 256` and re-run. Watch
   the stderr timing: at what size does the GPU stop being launch-bound and start
   beating the CPU? (The tiny committed sample is dominated by setup overhead.)
2. **Add a hard OAR constraint.** The current OAR penalty is a soft one-sided
   quadratic, so the organ can drift above tolerance (note `OAR max 29.2 Gy` vs a
   25 Gy target). Add a projection that clips any beamlet whose corridor pushes an
   OAR voxel over `d_max`, or raise `w_oar`, and observe the coverage/sparing
   trade-off shift.
3. **Swap the optimizer.** Replace projected gradient descent with **projected
   L-BFGS**. The two SpMVs stay identical — only the update rule changes — and
   convergence should need far fewer iterations. (See [THEORY.md §3](THEORY.md).)
4. **Hand-roll the SpMV.** Write your own one-thread-per-row CSR SpMV kernel and
   compare it to cuSPARSE on speed and on load balance for a matrix with very
   uneven row lengths. This is the fastest way to appreciate what the library buys
   you ([THEORY.md §4.2](THEORY.md)).
5. **Add a DVH histogram.** Compute a real dose-volume histogram for the PTV and
   OAR on the GPU (a per-bin atomic histogram, or `cuBLAS`/sort-based), and print
   `D95`/`V20`-style point metrics instead of just mean/min/max.

## Limitations & honesty

- **Synthetic data.** The dose-influence matrix is a **1-D Gaussian phantom**
  generated by `scripts/make_synthetic.py` — clearly labeled synthetic
  everywhere. It is *not* a patient dose calculation. Real `D` comes from a
  validated Monte-Carlo/pencil-beam dose engine on a 3-D CT.
- **Reduced-scope model.** We use a simple convex weighted-quadratic objective
  with three structure types and a soft OAR penalty. Clinical planning adds
  dose-volume constraints, biological (TCP/NTCP) and robust (minimax) objectives,
  and MLC/aperture deliverability — several of which are non-convex.
- **First-order optimizer.** Projected gradient descent is chosen to be readable;
  production uses L-BFGS/IPOPT that converge in far fewer iterations.
- **Not bit-identical.** The GPU and CPU agree only within a small *physical*
  dose tolerance (`1e-2 Gy`), because the two SpMVs sum in different orders over
  hundreds of iterations ([THEORY.md §5](THEORY.md)). We verify to a
  physically-negligible tolerance and say so.
- **Not for clinical use.** Educational only — no output here may inform any real
  diagnosis, treatment, or plan (CLAUDE.md §8).
