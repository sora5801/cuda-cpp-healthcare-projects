# 5.8 — Linac QA & Machine Performance Assessment

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🟢 Beginner · Established** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.8`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Before a linear accelerator (linac) treats a patient, a medical physicist verifies
that the dose it *actually delivers* matches the dose the treatment plan *intended*.
This project implements the two most common numbers in that check on a 2-D dose
plane: **(1) the 2-D gamma index** — a spatially-forgiving comparison of a measured
(EPID / portal-dosimetry) dose image against the planned dose, producing the clinical
**gamma pass rate**; and **(2) machine-performance metrics** — central-axis output,
beam **flatness**, **symmetry**, and field size from the measured profile. The gamma
map is the expensive part (every measured pixel searches a neighbourhood of the
reference plane), so it is computed on the GPU with **one thread per pixel** and
verified bit-for-bit against a plain CPU reference. All data here is **synthetic and
clearly labelled**; nothing is clinical.

## What this computes & why the GPU helps

Linear accelerator (linac) quality assurance measures beam output, flatness, symmetry, and MLC leaf positions from portal dosimetry images or log files. GPU acceleration is applied in three areas: (1) rapid gamma-index computation comparing measured vs. planned dose distributions (3D gamma on a 200³ dose grid requires ~10⁹ distance searches), (2) EPID (electronic portal imaging device) image-based dose reconstruction converting 2D portal images to 3D dose via a GPU MC kernel, and (3) machine learning prediction of machine failures from large log-file datasets (training on GPU). Automated daily QA with immediate GPU-based analysis enables real-time feedback before the treatment session.

**The parallel bottleneck:** the **gamma index**. For every measured pixel we must
find the *closest-agreeing* reference pixel in a combined dose/space metric — a local
neighbourhood search. On a clinical EPID frame (~1024², searching a few-mm window)
that is hundreds of millions of independent distance evaluations. Each pixel's answer
depends on nothing but a small window of read-only data, so the work maps perfectly
onto the GPU: **one thread per measured pixel**, a 2-D grid over the 2-D plane, no
atomics, no shared state. This project implements a **2-D** gamma (the daily-QA and
per-beam-IMRT-QA case); the full **3-D** volumetric gamma is the natural extension
(see THEORY §"real world"). This teaching version scopes to 2-D so the whole thing
builds, runs offline, and verifies exactly.

## The algorithm in brief

- **2-D gamma index** (Low 1998; TG-218): `γ(m) = min_r √( (ΔD/DD)² + (dist/DTA)² )`
  over reference pixels `r`; pass if `γ ≤ 1`. DD = dose-difference criterion (e.g. 3%
  of the max dose), DTA = distance-to-agreement (e.g. 3 mm).
- **Gamma pass rate** = % of evaluated pixels with `γ ≤ 1` (above a low-dose cut).
  TG-218 per-beam IMRT QA action limit: **≥ 95%** at 3%/3mm.
- **Machine metrics** from the central-axis profile: CAX output, **FWHM** field width,
  **flatness** `(D_max−D_min)/(D_max+D_min)`, **symmetry** `max|D(+x)−D(−x)|/CAX`.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/linac-qa-machine-performance-assessment.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/linac-qa-machine-performance-assessment.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\linac-qa-machine-performance-assessment.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the QA scorecard, shows the
GPU-vs-CPU agreement check (`max_abs_err = 0`), and prints a timing line to stderr.

## Data

- **Sample (committed):** `data/sample/qa_planes_sample.txt` — a tiny, offline pair of
  24×24 **synthetic** dose planes (planned + measured) so the demo runs with zero
  downloads.
- **Full dataset / real data:** `scripts/download_data.ps1` / `.sh` print pointers to
  the public reference material (documented, idempotent, and they never bypass any
  registration).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: AAPM TG-119 IMRT QA test cases; AAPM TG-218 tolerance criteria datasets; TCIA linac log datasets (verify URL); Varian/Elekta log file datasets from published QA studies; OpenMedPhys (https://github.com/jrkerns/awesome-medphys) reference datasets.

## Expected output

Success looks like `demo/expected_output.txt`. The program computes the gamma map on
both the **GPU** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`)
and asserts they agree with **`max_abs_err = 0`** — an *exact* match, because both
call the same `__host__ __device__` gamma routine (`src/gamma.h`) with identical
float operations. The scorecard for the committed sample reports a **100.00% gamma
pass rate** (a healthy machine), a worst-γ of 0.333 at a beam-edge pixel, ~0.99%
flatness, and ~1.96% symmetry — recovering the 1%-low / 2%-asymmetric error that was
deliberately baked into the synthetic measured plane.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the two dose planes, runs CPU + GPU gamma,
   verifies exact agreement, prints the QA scorecard.
2. [`src/gamma.h`](src/gamma.h) — the **shared** `__host__ __device__` per-pixel gamma
   math (the one true formula both CPU and GPU run — this is the crux).
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-pixel
   gather idea.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel and its host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline plus
   the flatness/symmetry/output metrics.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

Pylinac (https://github.com/jrkerns/pylinac) — Python linac QA automation (image analysis, log files); PRIMO MC linac simulator (https://www.primoproject.net/ — verify URL); Plastimatch (https://plastimatch.org/) — GPU-accelerated gamma index; matRad (https://github.com/e0404/matRad) — plan-vs-measurement comparison.

- **Pylinac** — the reference open-source implementation of exactly these QA analyses
  (flatness/symmetry, Winston-Lutz, gamma). Read it to see the clinical conventions we
  simplified (edge-finding, normalisation choices).
- **Plastimatch** — a production, GPU-accelerated gamma index; compare its search
  strategy and 3-D handling to our 2-D teaching kernel.
- **matRad** — a MATLAB research treatment-planning system with plan-vs-measurement
  comparison; good for seeing gamma in a full TPS context.
- **AAPM TG-218** (Miften et al., *Med Phys* 2018) — the tolerance/action-limit report
  that defines the 3%/3mm, ≥95% conventions this project uses.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Gather + per-thread min-reduction** (PATTERNS.md §1, the pattern exemplified by the
CT-backprojection flagship `4.01`): one thread owns one output (measured) pixel, reads
a local window of the read-only reference plane, and reduces it to a single minimum.
No atomics, no shared memory, no data races. The catalog also mentions texture memory
and cuBLAS for the log-file ML variant; those belong to the 3-D-gamma and
machine-learning extensions described in THEORY §"real world" and are out of scope for
this exact/deterministic teaching version.

## Exercises

1. **Break the machine.** Edit `scripts/make_synthetic.py` to inject a larger output
   error (e.g. `0.95`) or a 2-pixel shift, regenerate the sample, and watch the gamma
   pass rate drop below the TG-218 95% action limit. Which pixels fail first, and why?
2. **Tighten the tolerance.** Change the criteria to 2%/2mm (edit the header line of
   the sample, or add CLI flags). Explain why a stricter DTA raises the search radius
   and the cost.
3. **Local vs global gamma.** We normalise DD to the *global* max dose. Modify
   `gamma_value_at` to use *local* normalisation (DD as a percent of each reference
   point's dose) and compare pass rates in the penumbra.
4. **Add the second profile.** `compute_qa_metrics` only analyses the horizontal
   central-axis row. Add the vertical (cross-plane) profile and report both flatness
   values, as real QA does.
5. **Go to 3-D.** Extend the kernel and data format to a small 3-D volume (search a
   sphere instead of a square). This is the clinical VMAT-QA case; mind the
   `(2R+1)³` cost.

## Limitations & honesty

- **Synthetic data.** Both dose planes are generated analytically (open field + a
  scripted error) and are labelled synthetic everywhere. No patient or real machine
  data is used, and no output here is clinically valid.
- **2-D, not 3-D.** Clinical VMAT/IMRT QA increasingly uses a full 3-D volumetric
  gamma; we implement the 2-D case (portal-image / daily-QA plane) for tractability.
- **Global gamma, single reference orientation.** We use the standard global-gamma
  convention with a square pixel search window and do not interpolate the reference
  plane sub-pixel (a real analysis often refines the DTA with interpolation). These
  simplifications are documented in THEORY and left as exercises.
- **Simplified metrics.** Flatness/symmetry use the central row and a 50%-crossing
  edge finder; production tools use calibrated profiles, both axes, and vendor-specific
  definitions.
- **Timing is a teaching artifact, not a benchmark.** On this tiny 24² sample the GPU
  is launch/copy-bound and can be *slower* than the CPU; its advantage appears at
  clinical frame sizes (see THEORY §"real world").
