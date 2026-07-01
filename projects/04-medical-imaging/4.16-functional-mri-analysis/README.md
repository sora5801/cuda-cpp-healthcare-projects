# 4.16 — Functional MRI Analysis

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.16`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Functional MRI records a 4-D movie of the brain: at every voxel we get a BOLD
(blood-oxygen-level-dependent) time-series across many scans. This project
implements the workhorse of task-fMRI analysis — the **mass-univariate General
Linear Model (GLM)**. We build a design matrix from the task timing convolved with a
hemodynamic response function (HRF), then fit ordinary least squares (OLS)
**independently at every voxel** and compute a **t-statistic** for the task
contrast, producing an activation map. Because every voxel is an independent little
regression against the *same* design matrix, the workload is embarrassingly parallel
— we give **one GPU thread per voxel**. The demo runs on a tiny **synthetic**
dataset with a known planted answer and shows the GLM recovering exactly the voxels
we activated, with the GPU result verified against a CPU reference to ~10⁻¹³.

## What this computes & why the GPU helps

fMRI BOLD signal analysis involves preprocessing pipelines (motion correction, slice-timing, smoothing, registration) and statistical modeling (general linear model, GLM) across hundreds of thousands of voxels and thousands of time points. ICA (independent component analysis) via MELODIC decomposes a T × V spatiotemporal matrix; for 1,200 TRs and 150,000 gray-matter voxels, the matrix-SVD and subsequent unmixing are natural cuBLAS workloads. Resting-state functional connectivity computes a V × V correlation matrix — for 100,000 voxels this is a 10¹⁰-element matrix — computed efficiently on GPU via batched inner products. Dynamic functional connectivity via sliding-window or HMM approaches further multiply this cost, requiring GPU for tractable runtimes.

**The parallel bottleneck:** fitting the GLM means solving `β̂ = (XᵀX)⁻¹Xᵀy` and a
t-test at **every** voxel. For a whole brain that is `V ≈ 10⁵` independent
least-squares solves, each touching a `T ≈ 10²–10³` time-series — the dominant cost
of the statistical stage. The solves share the design matrix `X` and never interact,
so they map perfectly onto a GPU: one thread per voxel, with the voxel-independent
`(XᵀX)⁻¹` broadcast from constant memory. (The heavier fMRI workloads — ICA SVD,
`V×V` connectivity — are where cuSOLVER/cuBLAS come in; see THEORY "real world".)

## The algorithm in brief

- **HRF**: SPM canonical hemodynamic response = difference of two gamma densities
  (peak ~6 s, undershoot ~16 s), evaluated in log space for stability.
- **Design matrix `X`** (`T×3`): task boxcar ⊛ HRF, linear drift, intercept.
- **Precompute once**: `XᵀX` (a `3×3` Gram matrix) and its closed-form inverse.
- **Per voxel (OLS)**: `Xᵀy` → `β̂ = (XᵀX)⁻¹Xᵀy` → residual variance `σ̂²` →
  contrast t-statistic `t = β̂₀ / sqrt(σ̂² (XᵀX)⁻¹₀₀)`.
- **GPU pattern**: one thread per voxel (independent solves), constant memory for
  the shared design + inverse (docs/PATTERNS.md §1, like the 9.02 ensemble).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/functional-mri-analysis.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/functional-mri-analysis.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\functional-mri-analysis.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/fmri_sample.txt`, prints the top
activated voxels, shows the GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/fmri_sample.txt` — a tiny **synthetic**
  48-voxel × 80-scan block-design experiment so the demo runs with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print pointers to real
  public fMRI sources (HCP, OpenNeuro, ABIDE, UK Biobank) and never bypass
  credentials.
- **Provenance & license:** see [data/README.md](data/README.md). The committed
  sample is 100% synthetic — no patient data.

Catalog dataset notes: HCP fMRI (https://db.humanconnectome.org/) — resting-state and task fMRI, 7T/3T; OpenFMRI / OpenNeuro (https://openneuro.org/) — thousands of fMRI datasets in BIDS; ABIDE autism fMRI (http://fcon_1000.projects.nitrc.org/indi/abide/); UK Biobank fMRI (https://www.ukbiobank.ac.uk/).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): the
top-6 voxels ranked by task t-statistic, all tagged `[active]`, and a
`recovered 6/6` line confirming the GLM found the planted answer, ending in
`RESULT: PASS`. The program computes t-statistics on both the **GPU**
(`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and asserts
they agree within **1e-9** (observed ~3.7e-13). Both call the *same* per-voxel
routine (`src/glm.h`), so the only divergence is the GPU's fused-multiply-add — a
real, teachable ~10⁻¹³ effect (docs/PATTERNS.md §4).

## Code tour

Read in this order:

1. [`src/glm.h`](src/glm.h) — **start here**: the shared `__host__ __device__`
   science (HRF, design columns, the OLS + t-stat `fit_voxel()`), run identically
   on CPU and GPU.
2. [`src/main.cu`](src/main.cu) — loads data, precomputes `(XᵀX)⁻¹`, runs CPU + GPU,
   verifies, and prints the deterministic activation report.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-voxel idea.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel and host wrapper (constant memory).
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline + loader.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

FSL (https://fsl.fmrib.ox.ac.uk/) — MELODIC GPU ICA, FEAT GLM, BEDPOSTX; Nilearn (https://nilearn.github.io/) — Python fMRI statistical learning with scikit-learn; BrainSpace (https://github.com/MICA-MNI/BrainSpace) — gradient analysis on GPU; fMRIPrep (https://github.com/nipreps/fmriprep) — standardized preprocessing pipeline (CUDA-accelerated ANTs registration within).

- **FSL FEAT / FILM** — the reference implementation of the fMRI GLM (with
  prewhitening); study its design-matrix construction and contrast handling.
- **Nilearn** `FirstLevelModel` — a readable Python GLM; compare its HRF and
  drift/nuisance regressors to ours.
- **SPM** — the origin of the canonical double-gamma HRF we use here.
- **fMRIPrep** — what real preprocessing looks like before any GLM is fit.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**One thread per voxel** (independent OLS solves) with the voxel-independent design
parameters and precomputed `(XᵀX)⁻¹` in **constant memory** (broadcast warp-wide),
and a grid-stride loop to cover any `V`. This is the "many identical small solves"
pattern from docs/PATTERNS.md §1 (cf. the 9.02 SEIR ensemble), using the shared
`__host__ __device__` core idiom (§2) so CPU and GPU match near-exactly. No shared
memory or atomics — pure map parallelism.

Catalog CUDA note (verbatim): cuBLAS for GLM design-matrix product (V × T × T × T^-1 × T × V batched); cuSOLVER for ICA SVD; cuRAND for permutation testing; GPU histogram for parcellation; multi-GPU via PyTorch for DL resting-state classifiers.

## Exercises

1. **Coalescing.** Switch the BOLD layout from voxel-major (`bold[v*T+t]`) to
   time-major (`bold[t*V+v]`) and re-coalesce the kernel's global reads across
   threads. Measure the kernel-time change at larger `V` (see THEORY "GPU mapping").
2. **Bigger brain.** Run `python scripts/make_synthetic.py --V 200 --T 240` and
   watch the CPU-vs-GPU timing gap open up as launch overhead stops dominating.
3. **Prewhitening.** BOLD noise is autocorrelated, so OLS t-values are inflated. Add
   an AR(1) prewhitening step (GLS) and compare the t-map to the OLS one.
4. **More regressors.** Add a high-pass cosine (DCT) drift basis instead of the
   single linear term; generalize `FMRI_K` and the `3×3` solve accordingly.
5. **Connectivity.** Compute the `V×V` resting-state correlation matrix with a
   single `cuBLAS Dsyrk`/`Dgemm` on z-scored time-series — the heavy-library
   companion to this per-voxel GLM.

## Limitations & honesty

- **Reduced-scope teaching version.** We fit a plain OLS GLM with a task regressor,
  one linear drift term, and an intercept. Real pipelines add motion/physiological
  nuisance regressors, high-pass filtering, and prewhitening, and run only *after*
  motion correction, slice-timing correction, smoothing, and registration — none of
  which we do.
- **No multiple-comparison correction.** We rank raw t-values; we do not threshold or
  control family-wise error / FDR. Do not read the top-K list as "significant."
- **Synthetic data.** The committed sample is generated with a fixed seed and a toy
  HRF; the `[active]` labels are planted ground truth. BOLD "units" are arbitrary and
  imply nothing physiological. Nothing here is validated on real subjects.
- **Not a clinical tool.** This is study material only (CLAUDE.md §8) — no diagnostic
  or therapeutic use.
