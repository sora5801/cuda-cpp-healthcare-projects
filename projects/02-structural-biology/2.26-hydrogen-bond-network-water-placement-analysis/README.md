# 2.26 — Hydrogen Bond Network & Water Placement Analysis

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.26`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

This project teaches **Grid Inhomogeneous Solvation Theory (GIST)** — the method
behind tools like WaterMap that tell a medicinal chemist *which water molecules in
a protein pocket are worth displacing with a drug*. It overlays a 3D grid of
**voxels** on a binding site, streams a (synthetic) molecular-dynamics trajectory
of explicit water through it, and **scatter-accumulates** each water's occupancy
and interaction energy into the voxel it occupies. From those tallies it derives
per-voxel thermodynamics — density `g`, energy `ΔE`, entropy penalty `−TΔS`, and
free energy `ΔG` — and ranks the **hydration sites**. The unhappy, high-occupancy
waters (high `ΔG`) are the displacement targets. The point of the project is the
**GPU grid-accumulation-with-atomics** pattern and the **fixed-point determinism**
trick that makes the parallel result match the CPU exactly.

## What this computes & why the GPU helps

Water molecules mediate protein-ligand interactions at binding sites; their correct placement is critical for accurate docking and scoring. GPU-accelerated MD generates explicit water trajectories from which statistical water occupancy maps (WaterMap, GIST) are computed. The Grid Inhomogeneous Solvation Theory (GIST) requires computing per-voxel thermodynamic quantities (energy, entropy) across millions of trajectory frames — a GPU-parallelizable grid accumulation problem. High-occupancy waters indicate entropically costly displacement sites; displacing them with ligand atoms typically yields affinity gains.

**The parallel bottleneck:** the **scatter** in step 2 of the algorithm. There are
`frames × waters` independent water observations (`10⁶`–`10⁹` in a real run); each
computes a voxel index and an interaction energy and **atomically adds** them into
that voxel. This dominates the runtime and is embarrassingly parallel over
observations — one GPU thread per water sample. The per-voxel reduce that follows
is cheap. See [THEORY.md](THEORY.md) §3–§4.

## The algorithm in brief

- **Voxel grid** over the binding pocket (cubic cells, axis-aligned box).
- **3D water occupancy map from MD:** count how often a water lands in each voxel.
- **Per-voxel energy:** sum each water's Lennard-Jones + Coulomb interaction with
  the solute (the energetic fingerprint of the **hydrogen-bond network**).
- **GIST / IFST thermodynamics:** density `g`, excess energy `ΔE`, translational
  entropy penalty `−TΔS = k_B T ln g`, free energy `ΔG = ΔE − TΔS`.
- **Rank hydration sites** by occupancy (then `ΔG`); under-sampled voxels dropped.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation (including the nearest-neighbour orientational entropy and water-bridge
graph that a full implementation adds, §7).

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/hydrogen-bond-network-water-placement-analysis.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/hydrogen-bond-network-water-placement-analysis.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\hydrogen-bond-network-water-placement-analysis.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: SAMPL water placement challenges (https://github.com/samplchallenges/SAMPL); explicit-solvent PDB structures (https://www.rcsb.org); benchmark sets for WaterMap validation (Schrodinger, verify URL); GIST reference calculations for T4 lysozyme and FKBP12.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
2.26 -- Hydrogen Bond Network & Water Placement Analysis
GIST grid: 10x10x10 voxels @ 0.50 A spacing  (1000 voxels)
samples: 120 frames x 8 waters = 960 water observations; 14 solute atoms
hydration sites (voxels with adequate occupancy): 38

top 8 hydration sites (ranked by occupancy; GIST dG = displaceability, kcal/mol):
  rank  voxel(ix,iy,iz)   n      g      dE     -TdS      dG
    1   ( 4, 5, 5)        120  239.52   15.16    3.25   18.41
    2   ( 6, 4, 4)        120  239.52   14.93    3.25   18.18
    ...
RESULT: PASS (GPU voxel tallies + site ranking match CPU exactly)
```

The two synthetic **ordered, caged** waters at voxels `(4,5,5)` and `(6,4,4)` —
occupied every one of the 120 frames (`g ≈ 240`) and energetically strained — are
correctly recovered at **ranks 1–2**: the planted "displace me" answer.

The program computes the GIST tallies on both the **GPU** (`src/kernels.cu`) and a
**CPU reference** (`src/reference_cpu.cpp`) and asserts they agree **exactly** (zero
mismatch). That agreement is possible — not merely "within tolerance" — because the
energy is accumulated in **fixed-point integers**, whose atomic adds commute. The
timing line (on **stderr**, not diffed) is a teaching artifact, never a benchmark.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
2. [`src/gist.h`](src/gist.h) — the **shared `__host__ __device__` core**: the
   water–solute energy, the voxel lookup, the fixed-point accumulator, and the
   per-voxel GIST thermodynamics. The single most important file — both the CPU and
   GPU paths call this identical code, which is *why* they agree exactly.
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`src/reference_cpu.cpp`](src/reference_cpu.cpp)
   — the `Dataset` loader, the trusted serial GIST baseline, and the shared
   `derive_voxels` reduce.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
5. [`src/kernels.cu`](src/kernels.cu) — the atomic scatter kernel and host wrapper.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

GISTPP (https://github.com/liedlgroup/gist-pp) — GIST water thermodynamics analysis; cpptraj GIST (https://github.com/Amber-MD/cpptraj) — AMBER trajectory analysis with GIST; MDAnalysis water analysis (https://github.com/MDAnalysis/mdanalysis) — H-bond and water bridge analysis; WaterMD (verify URL) — GPU-accelerated solvation free energy.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Grid accumulation with atomic updates** (PATTERNS.md §1, exemplified by `5.01`
Monte-Carlo dose and `11.09` k-means). One thread per `(water, frame)` sample; each
thread maps its water to a voxel and **`atomicAdd`s** the occupancy and a
**fixed-point** energy into that voxel. Fixed-point integers make the atomic adds
**commute**, so the GPU result is deterministic and equals the CPU bit-for-bit
(PATTERNS.md §3). The per-voxel reduce reuses the CPU code on the host. No CUDA
library is needed — the scatter kernel is hand-written, which is the lesson.

## Exercises

1. **Orientational entropy.** We keep only the translational term `−k_B T ln g`.
   Record each water's dipole orientation and add the orientational entropy via a
   nearest-neighbour estimate (THEORY §7). How does the site ranking change?
2. **Shared-memory atom tiling.** With many solute atoms the energy loop becomes
   bandwidth-bound. Stage the atom list in `__shared__` memory per block and
   re-measure. Where is the crossover vs. the global-memory version?
3. **Per-voxel float-atomic experiment.** Swap the fixed-point energy accumulator
   for a `float` `atomicAdd` and run twice. Watch the GPU result stop matching the
   CPU (and itself) — then explain *why* in one sentence (PATTERNS.md §3).
4. **Bigger trajectory.** Regenerate with `--frames 5000` and a `20×20×20` grid;
   confirm the two ordered sites still top the list and watch the GPU's edge over
   the CPU grow with the sample count (timing on stderr).
5. **A water-bridge graph.** Cluster the high-occupancy voxels into discrete
   hydration *sites* and detect when two sites are bridged by a common water across
   frames — the "water bridge" half of the project title.

## Limitations & honesty

- **Synthetic data.** `data/sample/water_sample.txt` is **synthetic**, generated by
  `scripts/make_synthetic.py` with a fixed seed. The two top-ranked sites are
  *planted* (caged waters) so the demo has a known answer to recover. No real
  trajectory or patient data is involved.
- **Reduced-scope energy model.** A single Lennard-Jones + Coulomb term against a
  small solute atom set — not a real force field. No water–water energy, no
  PME/Ewald long-range electrostatics, no explicit water orientation. Constants
  (`ε`, `σ`, `E_bulk`, `ρ_bulk`) are textbook teaching values, not force-field
  parameters.
- **Translational entropy only.** The orientational entropy and the explicit
  hydrogen-bond / water-bridge graph (the catalog's nearest-neighbour estimator and
  network analysis) are described in THEORY §7 but not implemented here.
- **Not for any real decision.** The numbers are illustrative. This is study
  material for the GPU pattern, not a tool for docking, scoring, or drug design.
