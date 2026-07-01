# 4.22 — Quantitative Susceptibility Mapping (QSM)

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.22`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

**Quantitative Susceptibility Mapping (QSM)** turns the *phase* of a gradient-echo
MRI scan into a map of tissue magnetic **susceptibility** χ — a physical property
that separates iron-rich deep-brain nuclei and veins (paramagnetic, χ > 0) from
calcifications and myelin (diamagnetic, χ < 0). The measured field map is the
susceptibility distribution *convolved* with the magnetic **dipole kernel**, which
is a pointwise multiply in k-space. This project solves the **inverse**: recover χ
from the field map by undoing that multiply — an *ill-posed* problem because the
dipole kernel vanishes on a "magic-angle" cone. We implement the two canonical
fixes — **TKD** (threshold-based k-space division) and **Tikhonov-regularized
least squares** (closed-form *and* iterative) — doing the 3-D Fourier transforms
on the GPU with **cuFFT** and verifying every result against a plain CPU reference.
On a tiny synthetic phantom the demo recovers four "susceptibility blobs" and
prints a byte-deterministic report.

## What this computes & why the GPU helps

QSM reconstructs tissue magnetic susceptibility (χ) from gradient-echo phase data
in a 3-D volume. The pipeline is phase unwrapping → background-field removal →
**dipole inversion** (this project). The forward model in k-space is a
multiplication by an analytically-known dipole kernel; inversion is ill-posed at
the magic angle (the cone where the kernel crosses zero). Iterative MEDI-style
minimization needs O(100) iterations of 3-D FFT + gradient updates on a 256³
volume — each ~30 ms on GPU vs seconds on CPU; deep-learning QSM (QSMnet, xQSM)
replaces the solver with a single GPU network pass.

**The parallel bottleneck:** every method is bracketed by **3-D Fourier
transforms** of the whole volume, and the iterative method runs many of them. A
direct DFT is `O(N²)`; the FFT is `O(N log N)` and is exactly what GPUs (via
cuFFT) do fastest. That transform, plus the trivially-parallel per-k-space-bin
weighting, is what we parallelize.

## The algorithm in brief

- **Forward model (to make the input):** `field = IFFT3( D(k) · FFT3(chi) )`,
  where `D(k) = 1/3 − kz²/|k|²` (B₀ ∥ z) is the dipole kernel.
- **TKD inverse:** `chi = IFFT3( (1/D_thr(k)) · FFT3(field) )` — clamp `|D|` away
  from zero, then divide (direct, one shot).
- **Tikhonov inverse:** minimize `‖D·Fchi − Ffield‖² + α‖Fchi‖²`; closed form is
  the Wiener weight `D/(D²+α)`, and we also solve it by **iterative gradient
  descent** (the structure real MEDI-style solvers use).
- The 3-D FFTs run on the GPU with **cuFFT**; the per-bin weights and the gradient
  step are tiny custom kernels sharing math with the CPU via `qsm_core.h`.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation (the dipole kernel, the magic-angle ill-posedness, and why the
iterative solve converges to the closed-form minimizer).

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)). This project links the
**cuFFT** library (already wired into the `.vcxproj` and `CMakeLists.txt`).

1. Open `build/quantitative-susceptibility-mapping-qsm.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/quantitative-susceptibility-mapping-qsm.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\quantitative-susceptibility-mapping-qsm.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if the CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, reconstructs χ on both the CPU
and the GPU, prints the recovered susceptibilities and the GPU-vs-CPU agreement
check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/field_map.txt` — a tiny **synthetic**
  16×16×8 field map, produced by applying the dipole forward model to a known
  susceptibility phantom, so the demo runs with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print instructions for the
  QSM Reconstruction Challenge 2.0, HCP 7T, and UK Biobank (each under its own
  license / registration — we do not redistribute).
- **Regenerate / enlarge the sample:** `python scripts/make_synthetic.py --nx 24 --ny 24 --nz 12`.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: QSM Reconstruction Challenge 2.0
(<https://doi.org/10.1101/2020.11.25.397695> — data on Zenodo); HCP 7T multiecho
GRE data (<https://db.humanconnectome.org/>); AHEAD (Amsterdam Ultra-high-field
Adult lifespan Database); UK Biobank (<https://www.ukbiobank.ac.uk/>).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): the
recovered χ at the four known source voxels next to the ground truth, the RMS error
of each reconstruction vs the phantom, a data-consistency residual, and
`RESULT: PASS`. The program reconstructs χ on the **GPU** (`src/kernels.cu`, cuFFT)
and a **CPU reference** (`src/reference_cpu.cpp`, direct DFT) and asserts they agree
to within `atol = 1e-6` (they actually agree to ~`1e-16`, printed on stderr), and
that the iterative solve converged to the closed-form minimizer.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the field map, runs the CPU + GPU
   reconstructions, verifies the three checks, prints the deterministic report.
2. [`src/reference_cpu.h`](src/reference_cpu.h) — the `Volume` data model, the
   k-space grid helper, and the CPU function contracts.
3. [`src/qsm_core.h`](src/qsm_core.h) — the shared `__host__ __device__` per-bin
   math: the dipole kernel, the TKD reciprocal, the Tikhonov gradient step and
   closed-form weight. **This is where the physics lives.**
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the cuFFT idea.
5. [`src/kernels.cu`](src/kernels.cu) — the cuFFT 3-D plans, the two host wrappers
   (TKD and iterative Tikhonov), and the element-wise k-space kernels.
6. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted baseline (direct
   O(N²) 3-D DFT + the three reconstructions).
7. [`src/util/`](src/util/) — shared `CUDA_CHECK`, the CUDA-event timer, I/O helpers.

## Prior art & further reading

- **MEDI toolbox** (Cornell, <http://pre.weill.cornell.edu/mri/pages/qsm.html>) —
  the reference Morphology-Enabled Dipole Inversion; study its edge-aware ℓ₁
  regularizer, which our Tikhonov term deliberately simplifies away.
- **ROMEO** (<https://github.com/korbinian90/ROMEO>) — fast phase unwrapping, the
  pipeline stage *before* dipole inversion (we assume it done).
- **QSMnet** (<https://github.com/SNU-LIST/QSMnet>) — deep-learning QSM on GPU; a
  learned inverse that replaces the whole iterative solve with one network pass.
- **STISuite** — Susceptibility Tensor Imaging + QSM MATLAB toolbox; broadens QSM
  to anisotropic (tensor) susceptibility.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Use cuFFT for the spectral transform** (PATTERNS.md §1, the same pattern as
flagship `8.03` and sibling `4.30`). cuFFT does the 3-D double-complex FFTs; the
only custom kernels are the per-k-space-bin weighting (TKD `1/D_thr`, or one
Tikhonov gradient step) — each "one GPU thread per k-space bin", the most basic
CUDA mapping. The dipole operator is diagonal in k-space, so bins are independent
(no shared memory, no atomics). The catalog also lists `cuBLAS` conjugate-gradient
solvers and custom TV gradient/divergence operators for the full MEDI problem —
described in [THEORY.md](THEORY.md) "Where this sits in the real world".

## Exercises

1. **Turn the threshold up and down.** Change `TKD_THRESHOLD` in `src/main.cu`
   (try 0.05 and 0.30). Small `t` → sharper but streakier; large `t` → smoother but
   more underestimated χ. Watch the "recovered χ at sources" line move.
2. **Break the regularizer.** Set `TIK_ALPHA = 0` and watch the reconstruction blow
   up near the magic cone (you may see huge values). Then restore it and explain
   *why* α saves you, in terms of `D/(D²+α)`.
3. **Watch convergence.** Lower `TIK_ITERS` to 5, 20, 50. Print the
   iterative→closed-form gap (already on stderr) and see it shrink — gradient
   descent approaching the Wiener solution.
4. **Bigger volume.** `python scripts/make_synthetic.py --nx 24 --ny 24 --nz 16`
   and re-time. The CPU's O(N²) DFT explodes; the GPU barely notices — that gap is
   the whole point.
5. **Add a spatial prior (research-grade).** Replace the `α‖Fchi‖²` term with a
   total-variation penalty on χ. Now the gradient couples neighbouring voxels, so
   you must FFT *inside* every iteration — reproducing the real MEDI bottleneck.

## Limitations & honesty

- **Reduced-scope teaching version.** We implement only **dipole inversion** and
  assume a clean local field map. Real QSM must first **unwrap phase** and **remove
  background fields** — each its own inverse problem, out of scope here.
- **Simplified regularizer.** We use TKD and *Tikhonov* (ℓ₂) regularization, which
  have a closed form. The state of the art is **MEDI** (edge-aware ℓ₁) and
  deep-learning QSM; those are described in THEORY.md, not implemented.
- **Synthetic data.** The sample is a **synthetic** field map from a known phantom
  (no real patient data). Recovered χ is in arbitrary O(1) units, not physical ppm.
- **Bias is real.** The reconstructions **underestimate** χ near the magic cone (see
  the demo output) — a genuine QSM trade-off from regularization, not a bug.
- **Teaching-scale, FP64.** 16×16×8 double-precision so the demo is instant and the
  math is exact; real volumes are 256³ at FP32. The k-space physics is identical.
- This is **study material, not a clinical tool.** No output here is validated for
  any diagnostic or therapeutic use.
