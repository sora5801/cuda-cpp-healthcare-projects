# 2.35 — Electron Paramagnetic Resonance (EPR/DEER) Constrained Modeling

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.35`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Flexible proteins — membrane transporters, GPCRs, disordered regions — refuse to
sit still, so a single crystal structure misrepresents them. **DEER** (a pulsed-EPR
experiment) measures the *distribution* of distances between two spin labels
attached to the protein, reporting on the conformational ensemble in a near-native
environment. This project takes an ensemble of candidate structures and an
experimental distance distribution `P(r)`, and answers: **which conformations, and
in what populations, are consistent with the data?** It does so in two GPU-relevant
steps — back-calculating each frame's `P(r)` by **spin-label rotamer convolution**,
then **maximum-entropy reweighting** the ensemble to match the experiment — on a
small synthetic example with a *known answer* you can verify. It is a deliberately
reduced-scope, heavily-commented teaching version of a frontier method (see
[THEORY §7](THEORY.md)).

## What this computes & why the GPU helps

DEER (Double Electron-Electron Resonance) distance measurements between spin labels constrain the conformational ensemble of flexible proteins and membrane proteins in their native membrane environment. GPU-accelerated MD restrained by DEER distance distributions enables ensemble refinement of proteins that cannot be crystallized. The GPU compute pattern parallelize over hundreds of independent MD replicas, each evaluated against DEER restraints (population-weighted distance distribution comparison). Applications include ABC transporter gating, GPCR dynamics, and IDR backbone sampling.

**The parallel bottleneck:** the **DEER back-calculation**. Turning one frame into
its distance distribution requires convolving the two spin-label rotamer clouds —
an `O(R²)` sum over all rotamer pairs — and a real ensemble has `M ≈ 10⁴–10⁵`
frames, each independent. We map **one frame per GPU thread**, so all `M`
convolutions run at once with no inter-thread communication. The downstream
maximum-entropy reweighting is cheap (it touches only the `M`-vector of weights)
and runs as shared host code; see [THEORY §4](THEORY.md) for why that split is the
right one.

## The algorithm in brief

- **Rotamer-convolution back-calculation:** for each frame, histogram the `R²`
  pairwise spin–spin distances into a normalized `P_m(r)` (MTSSL spin-label cloud).
- **Population-weighted mixture:** the model is `P_w(r) = Σ_m w_m P_m(r)`.
- **Maximum-entropy reweighting (BioEn/EROS):** minimize `χ²(w) + θ·S_KL(w)` — fit
  to the target plus a relative-entropy regularizer — by gradient descent on
  softmax log-weights.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation (including the closed-form gradient).

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/electron-paramagnetic-resonance-epr-deer-constrained-modeling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/electron-paramagnetic-resonance-epr-deer-constrained-modeling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\electron-paramagnetic-resonance-epr-deer-constrained-modeling.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: SASBDB EPR-constrained structures (verify URL); published DEER datasets for membrane transporters; EPR.cxls community datasets (verify URL); PDB structures refined with EPR data.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The key
lines:

```
DEER back-calc: GPU vs CPU per-frame P(r) match = YES
  chi^2(uniform)    = 6.138988e-02
  chi^2(reweighted) = 2.790492e-04
  P(r) peak bin: uniform r=3.25 nm | reweighted r=3.45 nm | target r=3.45 nm
  true-frame population: prior 0.2500 -> reweighted 0.9895  (16/64 frames are true matches)
RESULT: PASS (GPU back-calc matches CPU; reweighting recovers the true frames)
```

The program back-calculates `P(r)` on both the **GPU** (`src/kernels.cu`) and a
**CPU reference** (`src/reference_cpu.cpp`) — they share the same
`__host__ __device__` math (`src/deer.h`), so the histograms agree **bit-for-bit**
(the stderr `[verify]` line shows `max |P_m cpu−gpu| = 0.0`). Then reweighting
concentrates ~99% of the population on the synthetic "true" frames and the model
`P(r)` peak snaps to the experimental target — recovering the known answer.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the ensemble, runs CPU + GPU back-calc,
   reweights, verifies, reports.
2. [`src/deer.h`](src/deer.h) — the shared `__host__ __device__` physics: rotamer
   convolution → `P_m(r)`, `χ²`, and the relative-entropy regularizer.
3. [`src/deer_params.h`](src/deer_params.h) — the fixed grid + solver constants.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
5. [`src/kernels.cu`](src/kernels.cu) — the one-frame-per-thread back-calc kernel.
6. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline +
   the shared loader and maximum-entropy reweighting solver.
7. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

MMM (Multiscale Modeling of Macromolecules, https://www.epr.ethz.ch/software/mmm.html) — EPR-driven ensemble modeling; DEER-PREdict (verify URL) — DEER distance prediction from MD; EnsembleFit/BioEn (https://github.com/bio-phys/BioEN) — GPU Bayesian ensemble reweighting; OpenMM DEER restraints (https://github.com/openmm/openmm) — soft distance restraints from DEER.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent jobs (one ensemble member per thread).** The DEER back-calculation
kernel gives each MD frame its own GPU thread; that thread runs the `O(R²)` rotamer
convolution and writes its own row of the `[M × NBINS]` histogram matrix — disjoint
outputs, so **no atomics and no races**. The population reweighting (maximum
entropy) is small and runs as shared host code, fed identical histograms from
either path. This is the same "GPU the heavy embarrassingly-parallel stage, keep
the cheap glue on the host" pattern as the ensemble flagships `9.02` / `13.02`
(see `docs/PATTERNS.md` §1).

## Exercises

1. **Tune the confidence `θ`.** Change `THETA` in `src/deer_params.h` to `1e-2`
   then `1e-6` and watch the `χ²(reweighted)` and the true-frame population. Plot
   the L-curve (fit vs. entropy) — this is how BioEn *chooses* `θ`.
2. **Bin-weighted `χ²`.** Real DEER fits weight each bin by its experimental
   uncertainty. Add a per-bin `σ_b` to `chi2_to_target` and the gradient, and
   regenerate a sample with noisy `σ_b` in `make_synthetic.py`.
3. **A bigger, harder ensemble.** Run `python scripts/make_synthetic.py
   --frames 2000 --true-frac 0.05` and re-time. Where does the GPU back-calc start
   to beat the CPU? (Watch the stderr timing.)
4. **GPU the mixture.** For large `M`, the per-step `P_w = Σ_m w_m P_m` becomes the
   bottleneck. Replace it with a `cublasDgemv` (or a custom reduction kernel) and
   compare — note you must keep the result deterministic (THEORY §5).
5. **Boltzmann-weighted rotamers.** Give each rotamer a weight (energy) instead of
   `1/R` and carry it through `deer_member_histogram`. Compare the back-calculated
   `P_m(r)` width to the equal-weight version.

## Limitations & honesty

- **The data is 100% synthetic** and labelled as such everywhere (`data/README.md`,
  the file header, `make_synthetic.py`). It is engineered to have a known answer;
  it is **not** a real protein or a real DEER measurement, and nothing here is for
  clinical or research use.
- **Reduced scope.** This implements the back-calculation and reweighting, **not**
  the upstream restrained MD that generates the ensemble (that is the part that
  truly needs a GPU MD engine). See [THEORY §7](THEORY.md) for the full pipeline.
- **Simplified physics:** 24 equal-weight rotamers (vs. ~200 Boltzmann-weighted in
  MMM), an unweighted `χ²` over `P(r)` (vs. a time-domain fit with a background and
  per-bin uncertainties), and a fixed `θ` (vs. Bayesian `θ`-selection in BioEn).
- **Timing is a teaching artifact, never a benchmark.** On the tiny 64-frame sample
  the CPU is faster (the GPU launch dominates); the GPU's advantage grows with
  ensemble and rotamer-library size.
