# 2.3 — Cryo-EM Single-Particle Reconstruction

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟢 Beginner · Established** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.3`
>
> _Educational only — not for clinical use (see CLAUDE.md §8). This is a
> **reduced-scope 2-D teaching model** of a 3-D research problem; see "Limitations"._

## Summary

Cryo-EM images thousands of frozen copies of a protein, each in a random, unknown
orientation and buried in noise, then reconstructs the molecule's 3-D density from
those projections. This project teaches the GPU-critical heart of that pipeline in
a tractable **2-D** setting: given many noisy 1-D projections of a synthetic
"molecule" at unknown angles, it (1) **matches** each projection to its best-fitting
reference angle — the `O(N·M)` cross-correlation sweep that dominates real cryo-EM
walltime — and (2) **back-projects** the matched profiles into a recovered 2-D
density. Two CUDA kernels do the work: a *projection-matching* kernel (one thread
per particle, reference bank in constant memory) and a *back-projection* kernel
(one thread per output pixel, a gather). Both reuse the exact same
`__host__ __device__` math as the CPU reference, so the GPU and CPU agree
bit-for-bit and the result is fully reproducible.

## What this computes & why the GPU helps

Single-particle cryo-EM reconstructs 3-D density maps from thousands to millions of
2-D projection images of vitrified protein particles in random orientations. The
pipeline involves CTF estimation, 2-D class averaging, 3-D ab-initio
reconstruction, and iterative 3-D refinement (Bayesian polishing in RELION,
non-uniform refinement in cryoSPARC). RELION-3/4 and cryoSPARC achieve 10–100× GPU
speedup over CPU; EMDB houses 50,000+ deposited maps.

**The parallel bottleneck — projection matching.** Assigning each particle an
orientation means scoring it against a bank of `M` reference projections: `N`
particles × `M` references = an `O(N·M)` cross-correlation sweep. With `N` in the
millions and `M` in the thousands this dominates the runtime. Each particle's score
is **independent of every other particle's**, so the sweep maps perfectly onto the
GPU — one thread per particle, the shared reference bank broadcast from constant
memory. The second stage (back-projection) is likewise embarrassingly parallel: one
thread per output pixel. These are the two kernels this project implements.

## The algorithm in brief

- **Forward projection (Radon transform):** a 1-D projection is the line integral of
  the 2-D density along parallel rays at angle `θ` (used to build the references).
- **Projection matching (the E-step):** assign each particle the reference angle that
  maximizes the **normalized cross-correlation** `NCC(particle, ref)` — scale- and
  offset-invariant, so it matches *shape* despite varying contrast.
- **Back-projection (the M-step):** smear each assigned profile back across the image
  along its view direction and average — the (unfiltered) inverse Radon transform.
- **Verification:** GPU vs CPU agree exactly on assignments and within `1e-4` on the
  density; we also report recovery accuracy vs. synthetic ground truth.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation, including how each piece maps to real 3-D cryo-EM.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/cryo-em-single-particle-reconstruction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/cryo-em-single-particle-reconstruction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\cryo-em-single-particle-reconstruction.sln /p:Configuration=Release /p:Platform=x64
```

Both `Release|x64` and `Debug|x64` build with zero warnings; the two configs
produce byte-identical stdout (a determinism check).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (optional CMake build)
```

The demo builds if needed, runs on `data/sample/cryoem_sample.txt`, prints the
result, shows the GPU-vs-CPU agreement check, and prints a timing line (on stderr).

## Data

- **Sample (committed):** `data/sample/cryoem_sample.txt` — a tiny, **synthetic**
  64×64 phantom with 60 reference angles and 120 noisy particles, so the demo runs
  offline with zero downloads.
- **Regenerate / scale up:** `python scripts/make_synthetic.py --n 100000`.
- **Real datasets:** `scripts/download_data.ps1` / `.sh` print pointers to EMDB,
  EMPIAR, RCSB, and cryoDRGN (no credentials are ever bypassed).
- **Provenance, layout & license:** see [data/README.md](data/README.md).

All committed data is **synthetic and labeled as such** — there is no real
specimen anywhere in this project.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
2.3 -- Cryo-EM Single-Particle Reconstruction
2D single-particle reconstruction (synthetic phantom)
geometry: image 64x64, 60 reference angles, 120 particles
E-step (projection matching, O(N*M)=7200 comparisons):
  orientation recovery accuracy = 70.8% (85/120 exact-angle hits)
M-step (back-projection into 64x64 density):
  reconstruction-vs-truth NCC = 0.8764
  density digest: centre=12.7030  q1=7.9486  q2=3.4080  q3=4.2988  q4=5.5459
RESULT: PASS (GPU matches CPU: 120/120 assignments exact, density within tol=1.0e-04)
```

The program runs the computation on both the **GPU** (`src/kernels.cu`) and a
**CPU reference** (`src/reference_cpu.cpp`) and asserts they agree — every
orientation assignment matches exactly (integer), and the density matches within
`1e-4` (observed `0.0`, because both call the same `__host__ __device__` math). The
70.8% figure is honest for σ≈15% noise at 3° sampling — 93% of particles land
within ±1 angle (see THEORY §5).

## Code tour

Read in this order:

1. [`src/reference_cpu.h`](src/reference_cpu.h) — **start here.** The data model
   *and* the shared `__host__ __device__` physics (`project_sample`, `ncc_score`,
   `backproject_pixel`) that both the CPU and GPU run. This is the heart.
2. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial loops
   (`match_cpu`, `reconstruct_cpu`) and the dataset loader.
3. [`src/kernels.cuh`](src/kernels.cuh) — the two GPU kernels' interface + the
   thread-mapping ideas (constant memory, gather).
4. [`src/kernels.cu`](src/kernels.cu) — the kernels and host wrappers; note they
   call the *same* HD functions as the CPU.
5. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, I/O helpers.

## Prior art & further reading

- **RELION** (https://github.com/3dem/relion) — open-source Bayesian/MAP cryo-EM
  reconstruction with CUDA; study its E-M orientation search and how it weights
  orientations probabilistically (our hard argmax is the simplest version of it).
- **cryoSPARC** (https://cryosparc.com) — GPU platform; learn its non-uniform
  refinement (adaptive regularization across the map).
- **cryoDRGN** (https://github.com/ml-struct-bio/cryodrgn) — a VAE over a latent
  conformation space; the answer to *heterogeneity*, which we ignore here.
- **cisTEM** (https://cistem.org) — a full GPU-accelerated suite; good for seeing
  CTF estimation and 2-D class averaging in context.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Two classic GPU patterns (docs/PATTERNS.md):

1. **Projection matching = independent jobs + constant-memory query** (like project
   1.12 Tanimoto): one thread per particle scores against the whole reference bank,
   which lives in `__constant__` memory and is broadcast warp-wide.
2. **Back-projection = per-output-pixel gather** (like project 4.01 CT
   back-projection): one thread per density pixel pulls a contribution from every
   particle's profile — no atomics, fully deterministic.

The catalog also names cuFFT for 3-D Fourier-slice reconstruction; that is the real
3-D upgrade described in THEORY §8 (we use a real-space back-projector for clarity).

## Exercises

1. **Add a ramp filter** to the back-projection (FFT the profile, multiply by `|ω|`,
   inverse FFT) and watch the reconstruction NCC rise as the blur clears — links to
   project 4.01.
2. **Search in-plane shifts too:** let each particle search over a small translation
   as well as an angle (a 2-D argmax). What does it cost in registers/occupancy?
3. **Soft assignment:** replace the hard argmax with a softmax over scores and
   back-project a *weighted* sum — the first step toward RELION's MAP refinement.
4. **Iterate (the EM loop):** re-project the current reconstruction to make fresh
   references, re-match, repeat. Does accuracy climb?
5. **Break it on purpose:** make the phantom radially symmetric and watch orientation
   recovery collapse to chance — proof that *information*, not compute, is the limit.

## Limitations & honesty

- **Reduced-scope & 2-D.** Real cryo-EM is 3-D with 3 Euler angles + 2 shifts per
  particle and a Fourier-slice (cuFFT) 3-D reconstruction. We use one angle and a
  real-space 2-D back-projector. THEORY §8 lays out exactly what is simplified.
- **No CTF, no Bayesian refinement.** We omit contrast-transfer-function correction
  and the MAP/EM iteration that RELION/cryoSPARC use; we do a single hard-assignment
  pass.
- **Synthetic data.** The phantom, references, and particles are generated by
  `scripts/make_synthetic.py` and labeled synthetic everywhere. The unfiltered
  back-projection is deliberately blurry.
- **Not a benchmark, not clinical.** Timings are a teaching artifact (tiny inputs are
  launch-bound). No output here is valid for any real structural-biology or medical
  conclusion.
