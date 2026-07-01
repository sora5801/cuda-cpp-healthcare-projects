# 4.15 — Diffusion MRI & Tractography

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.15`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Diffusion MRI measures how freely water diffuses along many directions inside
tissue; in brain **white matter** that diffusion is directional, and its preferred
direction is the direction the nerve fibers run. This project fits a **diffusion
tensor** to the signal in every voxel (recovering the standard **FA** and **MD**
scalar maps and the principal fiber direction), then reconstructs white-matter
**pathways** by tracing streamlines through the fiber-direction field. Both stages
are *embarrassingly parallel* — each voxel and each streamline is independent — so
they map cleanly onto the GPU with **one thread per voxel** (fit) and **one thread
per seed** (tractography). Everything runs on a tiny **synthetic** phantom with a
known answer, and every GPU result is checked against a plain-C++ reference.

## What this computes & why the GPU helps

Water diffusion anisotropy encodes fiber orientation. Fitting a per-voxel
diffusion model (DTI here; DKI/NODDI in production) is **trivially parallel** — each
voxel is an independent least-squares problem — so for a whole brain (~10⁵–10⁶
voxels × dozens of directions) batch GPU fitting is dramatically faster than serial
CPU. Tractography then samples many streamlines simultaneously, each step a
**trilinear interpolation** of the direction field (exactly what GPU texture units
accelerate).

**The parallel bottleneck:** the two per-element workloads — (1) the per-voxel
tensor fit (`fit_kernel`: a fixed 7×13 matrix-vector solve + a 3×3 eigen-solve per
voxel) and (2) the per-seed streamline integration (`tract_kernel`: repeated
8-neighbour gathers). Both have zero cross-element dependencies, so the GPU runs
thousands of them at once. On this tiny teaching volume the kernels are launch-
bound (the CPU is competitive); the GPU's edge grows with volume size.

## The algorithm in brief

- **Stejskal–Tanner signal model** `S_k = S0·exp(−b_k·gᵀDg)`, linearised by `ln`.
- **Ordinary least-squares tensor fit** via a fixed pseudo-inverse `M = (BᵀB)⁻¹Bᵀ`
  (computed once, reused for all voxels): `θ = M·ln(S)`.
- **Analytic 3×3 symmetric eigen-decomposition** (Smith 1961) → eigenvalues (sorted)
  and principal eigenvector.
- **FA / MD** scalar maps from the eigenvalues.
- **Deterministic streamline tractography**: Euler integration of the principal-
  direction field with trilinear interpolation, FA and curvature stopping.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/diffusion-mri-tractography.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/diffusion-mri-tractography.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\diffusion-mri-tractography.sln /p:Configuration=Release /p:Platform=x64
```

Only `cudart_static.lib` is linked — the DTI fit and tractography are hand-written
custom kernels (no cuBLAS/cuSOLVER needed at 3×3; see THEORY §4).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/dwi_sample.txt`, prints the FA/MD
summary + seed tensor fits + streamline lengths, shows the GPU-vs-CPU agreement
checks (fit and tractography), and prints timing to stderr.

## Data

- **Sample (committed):** `data/sample/dwi_sample.txt` — a tiny **synthetic**
  16×16×4 DWI volume with a curved fiber bundle and a known ground truth, so the
  demo runs offline with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (documented, prints links
  and a conversion recipe; never bypasses credentials).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: Human Connectome Project (HCP), 3T/7T multi-shell dMRI
(<https://db.humanconnectome.org/>); ABCD Study dMRI (<https://abcdstudy.org/>);
UK Biobank dMRI (<https://www.ukbiobank.ac.uk/>). All require a data-use agreement.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): mean
FA ≈ 0.80 on the bundle, seed voxels whose principal direction points along the
arc, and five short streamlines. The program computes everything on both the
**GPU** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and
asserts they agree within tolerance (fit 1e-9, tractography 1e-3) — that agreement,
plus the recovered known FA, is the correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the DWI volume, builds the OLS operator,
   runs CPU + GPU fit and tractography, verifies both, reports.
2. [`src/dti_core.h`](src/dti_core.h) — the shared `__host__ __device__` per-voxel
   physics (log-linear fit + analytic eigen-solve + FA/MD).
3. [`src/tract_core.h`](src/tract_core.h) — the shared per-step tractography
   (trilinear direction sampling with eigenvector-sign alignment).
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
5. [`src/kernels.cu`](src/kernels.cu) — the two kernels (constant-memory operator,
   fixed-size streamline slots) and their host wrappers.
6. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline
   + the loader + the pseudo-inverse builder.
7. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, I/O helpers.

## Prior art & further reading

- **MRtrix3** (<https://github.com/MRtrix3/mrtrix3>) — gold-standard CSD, iFOD2,
  SIFT2. Study `dwi2tensor` (the OLS fit we mirror) and `tckgen` (streamline
  propagation and its stopping criteria).
- **FSL BEDPOSTX GPU** (<https://fsl.fmrib.ox.ac.uk/>) — GPU Bayesian multi-fiber
  estimation (~200× speedup); the probabilistic counterpart to our deterministic
  tracking.
- **TractSeg** (<https://github.com/MIC-DKFZ/TractSeg>) — direct CNN white-matter
  tract segmentation (the deep-learning route).
- **DIPY** (<https://github.com/dipy/dipy>) — readable Python dMRI reference;
  `TensorModel` matches this fit almost line-for-line.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

The **"independent jobs"** pattern (PATTERNS.md §1, exemplar `1.12`): a custom
kernel with one thread per voxel for the DTI fit, reading the fixed OLS operator
from **constant memory** (broadcast to every warp); and one thread per seed for
tractography, whose per-step **trilinear interpolation** is the by-hand form of a
CUDA **texture** gather. The catalog also lists cuBLAS (for the CSD variant's
spherical-harmonic GEMMs) and cuRAND (for *probabilistic* tractography) — both
discussed in THEORY §4 as the next step up from this DTI teaching version.

## Exercises

1. **Add noise.** Perturb the synthetic signals (Rician noise) in
   `make_synthetic.py` and watch FA rise in the background — then compare weighted
   vs. ordinary least squares (real fitters weight by signal, since `ln` distorts
   the noise). How large a tolerance do you now need?
2. **Bind a texture.** Replace the hand-rolled `sample_dir` trilinear blend with a
   `cudaTextureObject_t` (linear filtering, clamp addressing) over the v1 field and
   confirm the streamlines still match; measure the difference at whole-brain scale.
3. **Denser scheme.** Grow the gradient scheme from 12 to 30–64 directions (extend
   `make_gradient_scheme` and `make_synthetic.py`, bump `NDIR`); observe how the fit
   conditioning improves.
4. **Probabilistic step.** Add a per-thread cuRAND generator and perturb each step's
   direction within a cone — the seed of iFOD2. (This breaks stdout determinism, so
   verify statistics, not exact points.)
5. **Second scalar map.** Compute and report **RD** (radial diffusivity, `(λ2+λ3)/2`)
   and **AD** (axial, `λ1`) alongside FA/MD, and verify them CPU-vs-GPU.

## Limitations & honesty

- **Synthetic data**, labeled synthetic everywhere: a noise-free phantom with a
  known ground truth, engineered so the result is interpretable and verifiable. No
  real patient data; **no clinical validity**.
- **Single-tensor DTI only.** It cannot resolve *crossing fibers* within a voxel
  (a fundamental DTI limitation) — production uses CSD/multi-fiber models. See
  THEORY §7.
- **Deterministic tractography**, not probabilistic iFOD2 (we omit RNG so stdout is
  reproducible); no anatomical priors, no streamline filtering (SIFT2).
- **No preprocessing** (motion/eddy-current correction, brain extraction,
  denoising) that a real pipeline requires; single shell only.
- **Timing is a teaching artifact, not a benchmark** — the tiny sample is launch-
  and copy-bound; the GPU advantage appears at whole-brain scale.
