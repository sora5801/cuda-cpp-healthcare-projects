# 1.13 — Pharmacophore & 3D Shape Screening

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.13`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

This project ranks a library of 3D molecules by how well their **shape** matches
a query molecule. Each molecule is modeled as a cloud of overlapping spheres (one
spherical Gaussian per atom), and two molecules are "similar" when their volumes
overlap a lot once superimposed. The score — the **Shape Tanimoto** — is the 3D,
continuous cousin of the bit-fingerprint Tanimoto from project 1.12: *volume*
intersection-over-union instead of *bit-set* intersection-over-union. Every
library molecule is scored independently, so the whole screen maps cleanly onto
the GPU: **one thread per conformer**. It is a beginner-friendly, self-contained
introduction to molecular shape comparison, the constant-memory broadcast
pattern, and CPU/GPU result parity via a shared physics core.

## What this computes & why the GPU helps

Pharmacophore and shape-based screening compares 3D query features (hydrogen bond
donors/acceptors, hydrophobic regions, ionizable groups, molecular shape) against
library conformers, capturing complementarity not encoded in 2D fingerprints.
ROCS (OpenEye) uses a volumetric Gaussian overlap function (ShapeTanimoto +
ColorTanimoto) that is differentiable and GPU-friendly. Screening billions of
conformers requires GPU-parallel overlap computation across independent molecule
pairs. This is a key pre-filtering step before docking in virtual screening
pipelines.

**The parallel bottleneck:** the cost is dominated by evaluating the
Gaussian-overlap integral between the query and **every** library conformer — a
double loop over atom pairs (M query atoms × K fit atoms), repeated for N
conformers. Each conformer's score is independent of the others, so the
embarrassingly-parallel dimension is **N** (the library size, which is millions
to billions in production). We give each conformer its own GPU thread; the query
sits in **constant memory** so every thread reads it from a broadcast cache.

## The algorithm in brief

- **Atom → Gaussian:** each heavy atom becomes a spherical Gaussian whose width
  (`alpha`) is set by its van der Waals radius (Grant–Pickett model).
- **Pairwise overlap:** two Gaussians' overlap integral has a **closed form** (no
  numerical integration) — one `exp()` per atom pair.
- **Molecule overlap:** `O_AB = Σ_i Σ_j V_ij` over all atom pairs (first-order).
- **Shape Tanimoto:** `O_AB / (O_AA + O_BB − O_AB)` ∈ [0, 1].
- **Screen + rank:** compute the score for every conformer, then report the top-K.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/pharmacophore-3d-shape-screening.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/pharmacophore-3d-shape-screening.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\pharmacophore-3d-shape-screening.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the top-5 ranking, shows
the GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/conformers_sample.txt` — a tiny, **synthetic**
  query + 9 library conformers so the demo runs with zero downloads. The library is
  engineered so the top hit (`lib_00_self`) scores exactly `1.000000` — a built-in
  correctness check.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print links + conversion
  instructions for real public conformer libraries (they do not bypass any
  registration).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: ZINC20 conformer libraries (https://zinc20.docking.org);
DUD-E (https://dude.docking.org); Enamine REAL conformer sets
(https://enamine.net); Directory of Useful Decoys-Enhanced including 3D conformers.

## Expected output

Success looks like `demo/expected_output.txt`. The program computes the Shape
Tanimoto of the query against every conformer on the **GPU** (`src/kernels.cu`) and
a **CPU reference** (`src/reference_cpu.cpp`), and asserts they agree within a
documented tolerance (`1e-9`). Both sides call the **same** double-precision physics
in `src/shape_overlap.h`, so they agree to ~machine precision (`max_abs_err ≈ 3e-16`)
— that agreement is the correctness guarantee.

```
1.13 -- Pharmacophore & 3D Shape Screening
Gaussian shape screen: query (7 atoms) vs 9 library conformers
top-5 by Shape Tanimoto:
  #1  lib_00_self  ShapeTanimoto = 1.000000
  #2  lib_01_jitter  ShapeTanimoto = 0.983953
  ...
RESULT: PASS (GPU matches CPU within tol=1.0e-09)
```

## Code tour

Read in this order:

1. [`src/shape_overlap.h`](src/shape_overlap.h) — the shared `__host__ __device__`
   physics (Gaussian widths, pairwise overlap, molecule overlap, Shape Tanimoto).
   **Start here** — it is the heart of the project, run identically on CPU and GPU.
2. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel (constant-memory query) and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the data loader + the trusted serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **ROCS (OpenEye/Cadence)** — the commercial standard for GPU 3D shape screening;
  defines the ShapeTanimoto/ColorTanimoto scores we model here.
  <https://www.eyesopen.com/rocs>
- **Open3DQSAR** — open 3D-QSAR tooling (related shape/field analysis).
  <https://open3dqsar.sourceforge.io>
- **RDKit shape tools** — open Gaussian-overlap shape alignment/scoring; the
  closest open reference for this computation. <https://github.com/rdkit/rdkit>
- **Pharmer** — open pharmacophore (feature-point) search.
  <https://github.com/dkoes/pharmer>
- **Grant, Gallardo & Pickett (1996)**, *J. Comput. Chem.* — the Gaussian
  molecular-volume model this project's math is built on.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent jobs + constant-memory query** (PATTERNS.md §1, shared with `1.12`
and `12.01`): one GPU thread per library conformer; the query molecule lives in
`__constant__` memory so the broadcast cache serves it warp-wide. The per-element
physics is a single `__host__ __device__` header shared by the CPU reference and
the GPU kernel (PATTERNS.md §2), giving byte-for-byte CPU/GPU parity. (The catalog
also mentions cuBLAS-accelerated rigid-body alignment; this teaching version uses
a *rigid pre-aligned overlay* and leaves alignment optimization to THEORY §7 and
the exercises — no black box hidden.)

## Exercises

1. **Color/pharmacophore Tanimoto.** Add atom *types* (HBD, HBA, hydrophobic,
   aromatic) and a second overlap that only sums pairs of the *same* type. Combine
   ShapeTanimoto and ColorTanimoto into a "Combo" score, as ROCS does.
2. **Optimize the overlay.** Right now conformers are assumed pre-aligned. Add a
   small rigid-body alignment search (translate + rotate via quaternions) that
   *maximizes* the overlap before scoring — the real ROCS inner loop.
3. **Scale it up.** Run `make_synthetic.py` variations to build tens of thousands
   of conformers and watch the GPU's relative advantage grow (the tiny sample is
   launch-bound; see the timing note).
4. **Shared-memory tiling.** For large molecules, stage the fit conformer's atoms
   in shared memory so a block of threads can reuse them. Measure the effect.
5. **Second-order volume.** Add the inclusion–exclusion triple-overlap correction
   and quantify how much the score changes versus the first-order sum.

## Limitations & honesty

- **Synthetic data.** The committed sample is engineered geometry, **not real
  molecules** — it exists to make the ranking interpretable and the demo offline.
  Nothing here has chemical or clinical meaning.
- **Rigid, pre-aligned overlay.** We score a fixed superposition; we do **not**
  optimize the alignment (translation/rotation) that production shape screening
  performs. This is the single biggest simplification — see THEORY §7.
- **First-order overlap only.** We sum pairwise Gaussian overlaps and stop (as ROCS
  does for scoring); the exact volume's higher-order inclusion–exclusion terms are
  omitted (Exercise 5).
- **Single Gaussian per atom; one radius.** A real implementation uses element-
  specific radii and can use multi-Gaussian atom descriptions. The sample uses
  one carbon-like radius so *shape*, not chemistry, is the variable.
- **No pharmacophore "color" features.** Only molecular *shape* is scored here;
  HBD/HBA/hydrophobic feature matching is left as Exercise 1.
- **`MAX_ATOMS = 64`.** A compile-time cap so the query fits in constant memory and
  molecules are POD; larger molecules need a different data layout.
