# 4.23 — Arterial Spin Labeling & Perfusion Imaging

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.23`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Arterial Spin Labeling (ASL) is a non-contrast MRI method that measures **brain
perfusion** — how much arterial blood is delivered to tissue each minute. It works
by magnetically "labeling" (inverting the spins of) water in arterial blood below
the brain, waiting a **post-labeling delay (PLD)** for that blood to flow into the
tissue, and subtracting a control image to isolate the tiny perfusion-weighted
difference signal ΔM (only ~0.5–1% of the raw signal). In **multi-delay** ASL we
acquire ΔM at several PLDs, giving each voxel an inflow *curve*. This project fits
the **Buxton kinetic model** to that curve, per voxel, to recover two physiological
quantities: **cerebral blood flow (CBF)** and **arterial transit time (ATT)**. Every
voxel is an independent nonlinear least-squares fit, so we give each voxel its own
GPU thread and solve them all in parallel.

## What this computes & why the GPU helps

Arterial spin labeling (ASL) magnetically labels water protons in arterial blood
upstream and images the resulting perfusion-weighted signal difference (label minus
control). The signal change is only 0.5–1% of background, so many pairs are averaged
for SNR; multi-delay ASL produces datasets where **kinetic-model fitting per voxel**
is the compute bottleneck. Oxford_asl/BASIL solves this as a Bayesian inverse problem
parallelized across voxels on the GPU.

**The parallel bottleneck:** the per-voxel model fit. A brain volume at 2 mm has
~10⁵–10⁶ tissue voxels; each requires an iterative nonlinear least-squares solve of
the Buxton model against its multi-delay curve. These fits are **mutually
independent** — voxel *v*'s solution needs nothing from voxel *w*. That is the ideal
GPU pattern: one thread per voxel, no communication, no shared state
(docs/PATTERNS.md rows 1 & 8). We implement the standard **Levenberg-Marquardt**
optimizer (the robust engine underneath oxford_asl's model fit) entirely in
registers per thread.

## The algorithm in brief

- **Buxton general kinetic model** (single-compartment pCASL): a closed-form
  ΔM(PLD; CBF, ATT) with three regimes (before arrival → arriving bolus → decaying
  bolus).
- **Per-voxel nonlinear least squares:** minimize Σⱼ (model(PLDⱼ) − ΔMⱼ)² over
  (CBF, ATT).
- **Levenberg-Marquardt** with analytic Jacobian, Marquardt diagonal scaling, and
  adaptive damping (accept/reject steps) — robust to the ~1000× scale mismatch
  between the CBF and ATT parameters.
- **GPU mapping:** one thread per voxel; the shared PLD schedule lives in constant
  memory (broadcast cache).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/arterial-spin-labeling-perfusion-imaging.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/arterial-spin-labeling-perfusion-imaging.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\arterial-spin-labeling-perfusion-imaging.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/asl_sample.txt`, prints the
per-voxel fit, shows the GPU-vs-CPU agreement check and the ground-truth recovery
check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/asl_sample.txt` — a tiny, offline synthetic
  study (6 voxels × 7 PLDs) with **noise-free** Buxton curves and known ground-truth
  CBF/ATT, so the fit's accuracy is checkable.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (documented, idempotent;
  they print sources and never bypass credentials).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: HCP ASL data (<https://db.humanconnectome.org/>); OpenNeuro
ASL datasets (<https://openneuro.org/> — search "ASL"); ISMRM 2015 ASL challenge
data; UK Biobank ASL pilot data.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): each
voxel's fitted CBF/ATT reproduces its ground-truth value, and the run ends with
`RESULT: PASS`. The program fits every voxel on both the **GPU** (`src/kernels.cu`)
and a **CPU reference** (`src/reference_cpu.cpp`) and asserts two things:

1. **GPU == CPU** to ≤ 1e-9 (both call the identical double-precision solver in
   `src/asl.h`; observed ~7e-15), and
2. **fit recovers ground truth** to ≤ 1e-4 (noise-free data; observed ~1e-8).

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the study, runs CPU + GPU, verifies, reports.
2. [`src/asl.h`](src/asl.h) — **the heart**: the `__host__ __device__` Buxton model,
   its analytic Jacobian, and the Levenberg-Marquardt fit (shared by CPU & GPU).
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-voxel
   idea + constant memory for the PLDs.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel and host wrapper (alloc/copy/launch).
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the loader + trusted serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **FSL BASIL** (<https://fsl.fmrib.ox.ac.uk/fsl/docs/physiological/basil.html>) —
  the reference Bayesian ASL analysis (variational Bayes over voxels); our LM fit is
  the deterministic core it wraps with priors. Study its kinetic-model definitions.
- **BART** (<https://github.com/mrirecon/bart>) — dynamic ASL compressed-sensing
  reconstruction; relevant to the *upstream* CS-MRI stage (out of scope here).
- **ExploreASL** (<https://github.com/ExploreASL/ExploreASL>) — a full multi-center
  ASL pipeline; good for understanding preprocessing (motion, registration, PVC).
- **SigPy** (<https://github.com/mikgroup/sigpy>) — GPU CS reconstruction primitives.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent per-voxel fit — one CUDA thread per voxel.** Each thread runs the full
Levenberg-Marquardt loop for its voxel in registers; the shared PLD schedule sits in
`__constant__` memory (broadcast cache). This is the "same model, many parameter
sets" pattern (docs/PATTERNS.md rows 1 & 8). We deliberately **do not** pull in
cuBLAS here: the per-voxel normal equations are only 2×2, so a hand-written
closed-form solve is faster and far more instructive than a batched library call —
THEORY.md explains where cuBLAS/cuSOLVER *would* pay off (higher-parameter models).

## Exercises

1. **Add measurement noise.** Give `make_synthetic.py` a `--noise` flag (Gaussian on
   ΔM) and watch the recovered CBF/ATT scatter; loosen `TOL_RECOVER` accordingly and
   plot fitted vs. true. This is where a Bayesian prior (BASIL) starts to help.
2. **Scale to a whole slice.** Generate `--voxels 1000000` and compare CPU vs. GPU
   time — the GPU should overtake the CPU by a wide margin. Where does the crossover
   sit on your card?
3. **Single-delay ASL.** Fix ATT to a constant and fit CBF only from one PLD (the
   clinical "single-delay" formula). Compare the CBF bias vs. the multi-delay fit.
4. **Robustify the init.** Replace the fixed initial guess with a per-voxel estimate
   (e.g. ATT from the PLD of peak signal). Does it cut the iteration count?
5. **Partial-volume correction.** Split each voxel into grey/white fractions with
   two CBFs and fit both — a small step toward the real BASIL PVC model.

## Limitations & honesty

- **Synthetic data.** The committed sample is generated (noise-free Buxton curves);
  it is labeled synthetic everywhere. Real ΔM is noisy and needs averaging.
- **Teaching model.** We fit the single-compartment pCASL model with **fixed,
  assumed-known** T1/α/λ/τ; only CBF and ATT are estimated. Production ASL adds
  dispersion, a macrovascular (arterial) component, T1 partial-volume correction,
  and Bayesian priors (BASIL) — described in THEORY §"real world".
- **No image reconstruction.** We start from already-reconstructed ΔM curves; the
  upstream k-space / compressed-sensing NUFFT stage (BART/SigPy) is out of scope.
- **Not clinical.** Outputs are a software demonstration of the kinetic fit, not a
  perfusion measurement, and must not be used for diagnosis or treatment.
