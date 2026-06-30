# 2.22 — Electron Density Map Analysis & Model Validation

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟢 Beginner · Established** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.22`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

When a structural biologist solves a protein structure by **X-ray
crystallography** or **cryo-EM**, the experiment produces a 3-D **electron-density
map** — a cube of numbers saying "how much electron density sits at each point in
space." Before that structure can be deposited in the PDB/EMDB, the map must be
**validated**: how good is it, and does the atomic model actually fit the density?

This project computes the two workhorse validation scores on the GPU and checks
them against a transparent CPU reference:

- **RSCC** — the *real-space correlation coefficient*: a Pearson correlation
  between two maps over every voxel ("do they agree, point by point?").
- **FSC** — the *Fourier shell correlation* curve: the normalized correlation of
  the two maps' Fourier transforms, averaged over spherical shells of spatial
  frequency. **FSC is the cryo-EM gold standard for resolution** — the resolution
  is the frequency where FSC first drops below a threshold (0.143 for two
  independent half-maps).

The Fourier transform is done with **cuFFT** (a 3-D FFT), the part worth a
library; the rest is small custom CUDA.

## What this computes & why the GPU helps

Crystallographic and cryo-EM electron density maps must be validated for
model-to-map fit quality before deposition. RSCC and FSC calculations run over
**millions of voxels** (a 256³ map is 16.7M), and the FFT that FSC needs is the
expensive step. Production tools (Phenix, CCP4, GEMMI, EMAN2) GPU-accelerate
exactly this. For cryo-EM, *local* resolution methods (MonoRes, ResMap) compute a
local FSC in a sliding window across the map — many independent FFTs, a natural
GPU job.

**The parallel bottleneck:** the **3-D Fourier transform** of each map dominates.
A naive DFT is `O(N²)` in the number of voxels — hopeless at map scale. cuFFT does
it in `O(N log N)` and uses the GPU's memory bandwidth to stream the cube. The
two surrounding reductions (the per-voxel real-space sums for RSCC, and the
per-shell sums for FSC) are also embarrassingly parallel per voxel. This project
keeps the FFT on the GPU (cuFFT) and finishes the *small* reductions
deterministically on the host (see Limitations).

## The algorithm in brief

- **RSCC** = Pearson correlation of maps A and B over all voxels. GPU: a
  shared-memory block reduction of five sums (Σa, Σb, Σa², Σb², Σab); the host
  adds the per-block partials in a fixed order, then closes the Pearson formula.
- **FSC**: FFT both maps (cuFFT, forward 3-D complex transform), then for each
  reciprocal-space shell `s = round(|k|)` accumulate
  `cross = Σ Re(F₁·conj F₂)`, `p₁ = Σ|F₁|²`, `p₂ = Σ|F₂|²`, and report
  `FSC(s) = cross / √(p₁·p₂)`.
- **Resolution** = the highest-frequency shell that stays above the threshold
  (0.143 half-map, or 0.5 map-vs-model), converted to Å.
- Related crystallographic scores (Fo−Fc / 2Fo−Fc difference maps, R/R-free) are
  described in [THEORY.md](THEORY.md) "Where this sits in the real world."

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)). This project links
**cuFFT** (`cufft.lib`), already wired into the `.vcxproj` and `CMakeLists.txt`.

1. Open `build/electron-density-map-analysis-model-validation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/electron-density-map-analysis-model-validation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\electron-density-map-analysis-model-validation.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/map_sample.txt`, prints the RSCC,
the FSC curve, and the resolution estimates, shows the GPU-vs-CPU agreement
check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/map_sample.txt` — a tiny **synthetic** pair
  of 16³ density maps so the demo runs offline with zero downloads. Map B has
  extra high-frequency noise, so the FSC curve decays at high frequency and yields
  a meaningful 8.0 Å "resolution."
- **Full dataset:** `scripts/download_data.ps1` / `.sh` point at EMDB/PDB and can
  fetch one open EMDB map as an example.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: EMDB validation maps (https://www.ebi.ac.uk/emdb/); PDB structure factors (https://www.rcsb.org); IUCr validation standards datasets (verify URL); wwPDB OneDep validation pipeline (https://deposit.wwpdb.org).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program computes RSCC and the FSC curve on both the **GPU** (`src/kernels.cu`, via
cuFFT) and a **CPU reference** (`src/reference_cpu.cpp`, a naive 3-D DFT) and
asserts they agree within the documented tolerance (RSCC ≤ 1e-9, FSC ≤ 1e-4) —
that agreement is the correctness guarantee. The deterministic stdout reports the
CPU's double-precision values; the (tiny) GPU-vs-CPU error is printed on stderr.

Headline result for the sample: `RSCC = 0.787929`, FSC ≈ 1 at low frequency
decaying past shell 4, **resolution @ FSC=0.143 = 8.0 Å**.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the maps, runs CPU + GPU, verifies, reports.
2. [`src/map_core.h`](src/map_core.h) — the **shared `__host__ __device__` core**:
   the per-voxel FSC/RSCC formulas and shell indexing, used identically by CPU and GPU.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the "library, not a black box" idea.
4. [`src/kernels.cu`](src/kernels.cu) — the cuFFT call, the extract/RSCC kernels, the host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline (naive 3-D DFT).
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, I/O helpers.

## Prior art & further reading

- **Phenix** (https://phenix-online.org) — crystallography + cryo-EM refinement
  and validation; study how it reports per-residue RSCC and map-model FSC.
- **CCP4** (https://www.ccp4.ac.uk) — the crystallographic computing suite;
  `EMDA`/`Servalcat` compute FSC and difference maps.
- **GEMMI** (https://github.com/project-gemmi/gemmi) — a clean library for reading
  MRC/CCP4 maps and CIF structure factors; learn the MRC header layout from it.
- **EMAN2** (https://blake.bcm.edu/emanwiki/EMAN2) — GPU cryo-EM processing; see
  its FSC and local-resolution tools.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Use a CUDA library without it being a black box** (PATTERNS.md §1 cuFFT row,
exemplar flagship 8.03). cuFFT does the 3-D forward FFT of each map; `kernels.cu`
documents exactly what `cufftExecC2C` computes and the data layout it expects.
Small custom kernels handle what cuFFT does not: widening the spectra to double,
and a shared-memory block reduction for RSCC. The float reductions are *finished*
on the host so stdout is byte-deterministic (PATTERNS.md §3).

## Exercises

1. **Bigger maps.** Regenerate the sample at `--n 32` or `--n 48`
   (`python scripts/make_synthetic.py --n 32`) and watch the CPU naive-DFT time
   blow up while cuFFT stays cheap — the `O(N²)` vs `O(N log N)` gap, made visible.
2. **Read a real MRC map.** Write a loader for the EMDB `.map` (MRC/CCP4) format
   (1024-byte header + float32 voxels) and validate two real half-maps. GEMMI's
   header docs are a good guide.
3. **Half-bit / 3-σ criteria.** Replace the fixed 0.143/0.5 thresholds with the
   frequency-dependent *half-bit* FSC criterion (van Heel & Schatz 2005) and
   compare the resolution it reports.
4. **Switch to R2C.** cuFFT's `CUFFT_R2C` stores only the non-redundant half of a
   real signal's spectrum (Hermitian symmetry). Rework the shell binning to use it
   and roughly halve the FFT memory; verify the FSC curve is unchanged.
5. **Local resolution.** Slide a small window across the map and compute a *local*
   FSC per window (the MonoRes/ResMap idea) — one independent FFT per window, a
   strong GPU-batching exercise.

## Limitations & honesty

- **Synthetic data.** The committed maps are generated, labeled synthetic
  everywhere, and carry **no clinical or structural meaning**. They exist to make
  the computation runnable and verifiable.
- **Reduced scope.** This is a *teaching* slice of map validation. We implement
  RSCC + FSC + resolution; the catalog also lists difference maps (Fo−Fc) and
  R/R-free, which need experimental structure factors and a model — described in
  THEORY.md but not implemented here.
- **Full complex FFT.** We use a full complex-to-complex FFT (not the
  memory-saving R2C half-spectrum) so the GPU produces the *exact same* cube the
  CPU's naive DFT does, making verification line-for-line. R2C is exercise 4.
- **Reductions finished on the host.** The heavy FFT runs on the GPU, but the
  small shell/RSCC sums are completed on the CPU so stdout is byte-deterministic
  (a parallel float sum reorders and would wander in its last bits). At real map
  scale you would push more of the reduction onto the GPU with deterministic
  fixed-point accumulation — see THEORY.md §Numerical considerations.
- **Timing is a teaching artifact**, never a benchmark claim (CLAUDE.md §12). On
  the tiny 16³ sample the GPU is *slower* than the CPU because the FFT plan setup
  and H2D copies dominate; the GPU's edge appears only as the map grows.
