# 1.14 — Conformer Ensemble Generation

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.14`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A flexible drug-like molecule does not have one shape — it has an *ensemble* of low-energy 3D shapes
(**conformers**) it interconverts between by rotating around its single bonds. Before you can dock a molecule
or run a 3D similarity search, you must **generate that ensemble**. This project teaches the idea on a
deliberately small molecule (a short flexible chain): it **enumerates** every conformer, **embeds** each one
in 3D from its torsion angles, scores each with a tiny **MMFF-style force field**, and then **prunes** the
ensemble down to a handful of distinct shapes by RMSD clustering. The energy of every conformer is computed
**in parallel on the GPU — one thread per conformer** — and checked against a serial CPU reference that runs
the identical physics.

## What this computes & why the GPU helps

Drug-like molecules are flexible; binding-relevant conformers must be generated before 3D screening or
docking. RDKit ETKDG embeds molecules in 3D using experimental torsion knowledge (ETKDGv3) and distance
geometry; generating thousands of conformers per molecule for a library of millions is a CPU bottleneck. GPU
acceleration is achieved by **batching conformer embedding across many molecules/conformers simultaneously.**

**The parallel bottleneck:** generating and scoring one conformer (turn torsion angles into 3D coordinates,
then evaluate the force-field energy — an O(atoms²) pairwise sum) is completely **independent** of every
other conformer. So when you need thousands of conformers (per molecule × millions of molecules), that is an
*embarrassingly parallel* workload. We map it the obvious way: **one GPU thread per conformer.** The thread
decodes its conformer index into torsion angles, builds the 3D structure entirely in registers, sums the
energy, and writes one number. Hundreds of conformers are embedded and scored in a single kernel launch.

## The algorithm in brief

- **Enumerate** conformers as a mixed-radix index over the rotatable torsions (3 rotamers × 5 torsions = 243).
- **Embed** each conformer in 3D from its (fixed bond length, fixed bond angle, variable torsion) internal
  coordinates using the **Natural Extension Reference Frame (NeRF)** construction — the same internal-to-Cartesian
  step inside distance-geometry embedders.
- **Score** each conformer with a small force field: a 3-fold **torsion potential** + a **soft-cored
  Lennard-Jones** steric-clash term.
- **Prune** the ensemble by greedy **RMSD clustering** (sort by energy, keep a conformer only if it is
  > threshold RMSD from every already-kept representative) — the standard conformer-dedup recipe.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/conformer-ensemble-generation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/conformer-ensemble-generation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\conformer-ensemble-generation.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the result, shows the GPU-vs-CPU agreement check,
and prints a timing line.

## Data

- **Sample (committed):** `data/sample/conformer_params.txt` — two run knobs (RMSD pruning threshold and how
  many representatives to print). The molecule itself is fixed in `src/conformer.h`, so the demo runs fully
  offline with zero downloads. **Synthetic / illustrative** (see [data/README.md](data/README.md)).
- **Full / real datasets:** `scripts/download_data.ps1` / `.sh` print pointers to GEOM, the CSD torsion
  library, and COD (they are large and/or license-restricted, and are **not** needed for this teaching demo).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: GEOM — 37M conformers of drug-like molecules with DFT energies
(https://github.com/learningmatter-mit/geom); CSD torsion library (https://www.ccdc.cam.ac.uk); COD —
Crystallography Open Database (https://www.crystallography.net); PDB small-molecule conformations
(https://www.rcsb.org).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): 243 conformers enumerated, pruned
to 81 distinct shapes, with the **global minimum being the all-anti extended conformer** (torsions
`+180 +180 +180 +180 +180`) — the chemically correct answer for a saturated chain. The program computes every
conformer's energy on both the **GPU** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`)
and asserts they agree within `1.0e-9` kcal/mol — that agreement is the correctness guarantee. The observed
`max_abs_err` is ~`5e-12` kcal/mol (genuine fused-multiply-add rounding; see THEORY "Numerical considerations").

## Code tour

Read in this order:

1. [`src/conformer.h`](src/conformer.h) — **start here.** The shared `__host__ __device__` physics: index →
   torsions → 3D coordinates (NeRF) → energy. The CPU and GPU both call these exact functions.
2. [`src/main.cu`](src/main.cu) — loads parameters, runs CPU + GPU, verifies, prunes, reports.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-conformer mapping.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel (mostly thread→conformer bookkeeping around `conformer_energy`).
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial energy sweep + the RMSD clustering.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, I/O helpers.

## Prior art & further reading

- **RDKit ETKDG** (https://github.com/rdkit/rdkit) — the standard open-source conformer engine. Study how
  ETKDGv3 seeds a distance-geometry embedding with *experimental* torsion preferences, then refines with
  MMFF94. Our toy here replaces the stochastic distance-geometry guess with deterministic torsion enumeration,
  but the embed-then-score-then-prune pipeline is the same.
- **TorsionalDiffusion** (https://github.com/gcorso/torsional-diffusion) — a GPU diffusion model that samples
  torsion angles directly; a glimpse of where the field is heading (ML over the same torsional degrees of
  freedom we enumerate by hand).
- **GeoMol** (https://github.com/PattanaikL/GeoMol) — GNN-based conformer prediction.
- **Frog2 / OMEGA** (OpenEye, commercial) — fast rule/rotamer-library conformer generators; our discrete
  rotamer set (anti/gauche±) is the same idea in miniature.

Study these to learn the production approach; **do not copy code wholesale** — reimplement didactically and
credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent jobs / embarrassingly parallel** (docs/PATTERNS.md §1, exemplified by flagship `1.12`): one
thread per conformer, no inter-thread communication, no shared memory, no atomics. The per-conformer physics
lives in **one shared `__host__ __device__` header** (`conformer.h`, PATTERNS §2) so the CPU reference and the
GPU kernel compute byte-identical math — making verification exact to ~machine precision. The catalog also
mentions cuSOLVER batched distance-geometry and GPU pairwise-RMSD kernels for the full-scale problem; those
are described under "Where this sits in the real world" in THEORY and left as exercises.

## Exercises

1. **GPU pairwise RMSD.** The clustering currently runs on the CPU. Write a kernel that fills the full
   N×N RMSD matrix (one thread per pair) and time it — this is the "custom CUDA kernels for pairwise RMSD"
   the catalog names. (Hint: it is the same independent-jobs pattern, now over pairs.)
2. **More flexibility.** Increase `N_ATOMS` (and therefore `N_TORSION`) in `conformer.h`. The conformer count
   grows as `3^(N_ATOMS-3)` — watch the GPU's advantage appear as the count climbs past a few thousand.
3. **Finer rotamers.** Add a 4th/5th rotamer angle (e.g. ±90°) and see how the ensemble and the pruned set
   change. Does the global minimum stay all-anti?
4. **Kabsch superposition.** Our RMSD skips alignment because all conformers share their first three atoms.
   Add an optimal-rotation (Kabsch) superposition before the RMSD and confirm the pruning is stricter.
5. **FP32 vs FP64.** Switch the physics to `float` and measure how much `max_abs_err` grows — a concrete
   lesson in why this verification uses double precision.

## Limitations & honesty

- **Reduced-scope teaching version.** A production conformer generator (RDKit ETKDG) works on an *arbitrary*
  molecular graph, seeds 3D coordinates from a **stochastic distance-geometry** embedding biased by
  experimental torsion statistics, and minimizes a full **MMFF94** force field. We instead use a single fixed
  linear chain, a **deterministic** enumeration of discrete rotamers, and a deliberately tiny force field
  (one torsion term + a soft-cored repulsion). The *pipeline* (enumerate → embed → score → prune) and the
  *GPU mapping* (one thread per conformer) are faithful; the chemistry is intentionally simplified.
- **The force field is illustrative, not parameterized.** The constants (V₃, ε, σ, the soft-core floor) are
  chosen to be physically reasonable and numerically well-behaved, not to reproduce real alkane energetics.
- **Synthetic input.** Everything here is synthetic and labelled as such; no real molecule, no clinical claim.
- **Timing is a teaching artifact.** 243 conformers is far too small for the GPU to win — the run is dominated
  by launch/copy overhead. The point is the *mapping*; the speed-up is real only at library scale.
