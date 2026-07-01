# 4.25 — Image Harmonization Across Scanners/Sites

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.25`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Multi-site imaging studies pool subjects scanned on **different machines** (vendors,
field strengths, protocols). Each scanner stamps a systematic **batch effect** onto
every image-derived feature — a shift that looks like biology but is really hardware,
and that confounds any downstream group analysis. This project implements **ComBat**,
the field-standard *statistical* harmonizer (NeuroComBat): it removes each scanner's
per-feature **location** (mean) and **scale** (variance) signature while **preserving**
the biological covariates you care about (age, sex, diagnosis). The GPU pattern is an
**ensemble of independent per-feature solves** — one thread harmonizes one whole
feature, the same shape as the ODE-ensemble flagships (9.02, 13.02).

> **Reduced-scope teaching version (CLAUDE.md §13).** The catalog also lists *image-level*
> deep harmonizers (CycleGAN, CALAMITI, DeepHarmony). Training those is ~100 GPU-hours on
> 256³ volumes — out of scope for a study project. We ship the statistically-tractable
> **ComBat** core, which is what most published multi-site pipelines actually run, and we
> describe the deep methods in [THEORY.md](THEORY.md) *§Where this sits in the real world*.

## What this computes & why the GPU helps

Feature-level harmonization operates on a table of **N samples × P features** where each
sample belongs to one of **B scanners**. ComBat fits a small linear model per feature,
estimates that feature's per-scanner mean/variance shift, shrinks those estimates toward
a panel-wide **empirical-Bayes** prior (so small scanners "borrow strength"), and
subtracts the shift. Every feature is processed **independently**, so the work is
embarrassingly parallel across P.

**Why the GPU helps:** feature sets get large. A *voxel-wise* or *vertex-wise*
harmonization has **P ≈ 10⁵–10⁶** features, each needing its own regression +
empirical-Bayes fit. Mapping one feature to one GPU thread turns a serial `for p in
features` loop into a single launch. **The parallelized work** is the per-feature ComBat
pipeline (fit → standardize → estimate L/S → shrink → adjust); the cheap panel-wide prior
fit runs once on the host.

## The algorithm in brief

- **Design:** build `X = [ covariates | batch-indicators ]` (full rank, no intercept — see THEORY §Numerics).
- **Fit:** per feature, OLS `β = (XᵀX)⁻¹ Xᵀy`; recover grand mean + pooled residual SD `σ`.
- **Standardize:** `z = (y − grand_mean − covariate_fit) / σ`.
- **L/S estimate:** per scanner, location `γ̂ = mean(z)`, scale `δ̂ = var(z)`.
- **Empirical-Bayes shrink:** pull `γ̂, δ̂` toward across-feature priors (`γ̄, τ², a, b`).
- **Adjust:** `z* = (z − γ*) / √δ*`, then map back: `y* = z*·σ + grand_mean + covariate_fit`.

See [THEORY.md](THEORY.md) for the model, the empirical-Bayes derivation, and the GPU mapping.

## Build

Requires **Visual Studio 2026** (v145) + **CUDA 13.3** ([docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/image-harmonization-across-scanners-sites.sln`.
2. Select **`Release|x64`** → **Build** → produces
   `build/x64/Release/image-harmonization-across-scanners-sites.exe`.

CLI: `msbuild build\image-harmonization-across-scanners-sites.sln /p:Configuration=Release /p:Platform=x64`

## Run the demo

```powershell
./demo/run_demo.ps1
```

Harmonizes the committed synthetic table on CPU + GPU, verifies the two harmonized tables
agree to ~machine precision, and reports the across-scanner mean gap **before vs after**.

## Data

- **Sample (committed):** `data/sample/harmonization_sample.txt` — **synthetic**, 24
  samples, 12 features, 3 scanners, 1 covariate (age). Clearly labeled synthetic; the
  scanner offsets and the age signal are known, so the demo is interpretable.
- **Real data:** ABIDE, ADNI, UK Biobank, IXI — all need registration/credentials and
  most forbid redistribution. `scripts/download_data.ps1`/`.sh` print the official links
  and instructions (they never bypass credentials). See [data/README.md](data/README.md).
- Regenerate / resize the synthetic set: `python scripts/make_synthetic.py --p 200 --b 4`.

## Expected output

`demo/expected_output.txt` holds the deterministic report: the dataset dimensions, the
**max across-scanner feature-mean gap before vs after** harmonization (it collapses from
`7.74` to `0.80` — the scanner signature is removed), a few harmonized values, and
`RESULT: PASS`. The GPU (`src/kernels.cu`) and CPU (`src/reference_cpu.cpp`) call the
**same `__host__ __device__` core** (`src/combat.h`) on identical double-precision inputs,
so the harmonized tables agree to `~7e-15` (tolerance `1e-9`; the tiny gap is FMA
rounding, [THEORY.md](THEORY.md) §Numerics).

## Code tour

1. [`src/main.cu`](src/main.cu) — load, build design + priors, run CPU + GPU ComBat, verify, print.
2. [`src/combat.h`](src/combat.h) — **the entire per-feature ComBat pipeline** as one shared `__host__ __device__` function.
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — loader, design builder, **empirical-Bayes prior fit**, serial reference.
4. [`src/kernels.cuh`](src/kernels.cuh) — the one-thread-per-feature kernel interface.
5. [`src/kernels.cu`](src/kernels.cu) — the kernel (delegates to `combat.h`) + the host wrapper (5 canonical CUDA steps).

## Prior art & further reading

- **NeuroComBat** (<https://github.com/Jfortin1/ComBatHarmonization>) — the reference
  statistical ComBat for neuroimaging; our math mirrors it (design, standardization, EB shrinkage).
- **CycleGAN** (<https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix>) — unpaired
  *image-level* harmonization; adaptable to MRI but heavy to train.
- **NiftyMIC** (<https://github.com/gift-surg/NiftyMIC>) — multi-contrast MRI reconstruction/harmonization.
- **CALAMITI** — disentangled multi-contrast GPU harmonization (search GitHub).

Study these for production harmonization; reimplement the pattern didactically (CLAUDE.md §2).

## CUDA pattern used here

**Ensemble of independent per-feature solves** — one GPU thread runs the whole ComBat
pipeline (a small OLS fit + empirical-Bayes shrinkage) for one feature in registers/local
memory. No shared memory, no atomics, no cross-thread reduction → **deterministic** and
byte-for-byte matched to the CPU via a shared `__host__ __device__` core (PATTERNS.md §2).

## Exercises

1. **Recover the age signal.** Regress each *harmonized* feature on age; confirm the slope
   survives harmonization (biology preserved) while the scanner gap collapsed.
2. **Turn off empirical Bayes.** Replace the shrunk `γ*, δ*` with the raw `γ̂, δ̂` (a naive
   per-scanner z-score). The gap goes to ~0 but per-feature estimates get noisier for small
   scanners — *why ComBat shrinks*.
3. **Add a rank-deficient intercept.** Re-add an all-ones column to the design and watch the
   normal-equations solve become ill-posed (CPU≠GPU); this is the lesson in THEORY §Numerics.
4. **Scale up.** `python scripts/make_synthetic.py --p 200000` and compare CPU vs GPU wall time
   as P grows — the ensemble parallelism is where the GPU pulls ahead.
5. **Longitudinal ComBat.** Extend the design with a within-subject term (LongCombat) so
   repeated scans of the same subject share a random effect.

## Limitations & honesty

- **Statistical (feature-level) harmonization only.** This is ComBat, *not* image-level
  deep harmonization; it operates on extracted features, not raw voxels. The catalog's
  CycleGAN/CALAMITI methods are described in THEORY but not implemented (out of scope).
- **Parametric empirical Bayes** with a closed-form update (no iterative EM refinement of
  the posterior). NeuroComBat optionally iterates; we do the single closed-form step.
- **Synthetic data**, generated so the scanner effect and the biological signal are known.
  It is labeled synthetic everywhere and implies **no clinical validity**.
- The harmonized across-scanner gap is small but **not exactly zero** — empirical-Bayes
  shrinkage deliberately leaves a little (that robustness is the point vs a naive z-score).
