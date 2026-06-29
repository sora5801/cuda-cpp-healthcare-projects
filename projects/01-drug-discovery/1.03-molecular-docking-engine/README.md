# 1.3 — Molecular Docking Engine

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.3`
>
> _Educational only — not for clinical use (see CLAUDE.md §8). This is a
> **reduced-scope teaching version** of molecular docking; see THEORY.md §7._

## Summary

Molecular **docking** predicts how a small candidate molecule (a *ligand*) fits
into a protein's binding pocket. This project builds the parallel heart of a
docking engine: it precomputes the pocket as a 3-D **energy grid**, samples a large
set of rigid ligand **poses** (translations × rotations), and **scores** each pose
by summing the grid energy over the ligand's atoms — keeping the lowest-energy
(best-fitting) pose. Because every pose is scored independently, we put **one GPU
thread per pose**. The result on the committed sample recovers a *known* answer:
the ligand centroid drops exactly into a synthetic energy well, which is how you
can trust the engine at a glance.

## What this computes & why the GPU helps

Docking samples ligand conformations (translations, rotations, torsions) and scores
each with an empirical energy function. **The scoring of each pose is independent**,
creating massive data parallelism — thousands of poses per ligand, millions of
ligands per virtual-screening campaign. AutoDock-GPU reaches >1000× over single-CPU
AutoDock4 by scoring distinct poses in parallel across GPU threads.

**The parallel bottleneck** is the per-pose **grid energy evaluation**: for each of
the ligand's atoms, transform it by the pose and read the receptor energy at that
point via **trilinear interpolation** (8 grid reads/atom). That gather is the inner
loop docking spends its time in, and it is what we parallelize: one thread scores a
whole pose, and the best pose is found with a deterministic GPU **min-reduction**
that carries the winning index.

## The algorithm in brief

- **Energy grid:** the receptor pocket precomputed as `grid[z][y][x]` = energy a
  probe atom feels there (kcal/mol); sampled at continuous atom positions by
  **trilinear interpolation**.
- **Pose:** a rigid transform `p = R(a,b,c)·l + t` of each ligand-local atom `l`.
- **Score:** `S(pose) = Σ_atoms weight · grid_energy(p)`; lower = better.
- **Search:** evaluate every pose on a lattice of `n_trans³` translations ×
  `n_rot³` rotations and keep the global minimum (deterministic tie → lowest index).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/molecular-docking-engine.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/molecular-docking-engine.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\molecular-docking-engine.sln /p:Configuration=Release /p:Platform=x64
```

This project links only the CUDA runtime (`cudart_static.lib`); the reduction uses
built-in intrinsics (`atomicMin`, `__shfl_down_sync`), so no extra CUDA library is
needed.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (optional CMake build)
```

The demo builds if needed, runs on `data/sample/receptor_ligand_sample.txt`, prints
the best pose, shows the **GPU-vs-CPU agreement** check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/receptor_ligand_sample.txt` — a tiny
  **synthetic** docking problem (a Gaussian energy well + a rigid 5-atom ligand,
  9261 poses). Runs the demo with zero downloads.
- **Make/scale it:** `python scripts/make_synthetic.py [--n-trans 15 --n-rot 6]`.
- **Real datasets & how docking inputs are prepared:** `scripts/download_data.ps1`
  / `.sh` print pointers (they never bypass any registration); details and the file
  format are in [data/README.md](data/README.md).

Catalog datasets: **DUD-E** (<https://dude.docking.org>), **ChEMBL**
(<https://www.ebi.ac.uk/chembl/>), **PDBbind** (<http://www.pdbbind.org.cn>),
**CASF** (<http://www.pdbbind.org.cn/casf.php>).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program docks the same problem on the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and asserts they pick the **same winning pose
index** (exact integer) and agree on its energy within `1e-9` kcal/mol. Both call
the *same* `score_pose()` from the shared `docking_core.h`, so on the sample the
energy error is exactly **0** and the best translation **(0.5, −0.5, 0.0) Å**
recovers the synthetic well's location.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the problem, runs CPU + GPU, verifies, reports.
2. [`src/docking_core.h`](src/docking_core.h) — the shared `__host__ __device__`
   physics: trilinear grid sampling, pose transform, `score_pose` (CPU/GPU parity).
3. [`src/reference_cpu.h`](src/reference_cpu.h) — the data model (grid, ligand,
   search space) + loader/reference prototypes.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-pose idea.
5. [`src/kernels.cu`](src/kernels.cu) — the kernel (grid-stride scoring + deterministic
   index-carrying min-reduction) and host wrapper.
6. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline + loader.
7. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, I/O helpers.

## Prior art & further reading

- **AutoDock-GPU** (<https://github.com/ccsb-scripps/AutoDock-GPU>) — CUDA/OpenCL
  docking with Lamarckian-GA parallelism and texture-memory grids; the direct
  ancestor of this project's pattern.
- **Uni-Dock** (<https://github.com/dptech-corp/Uni-Dock>) — GPU batch docking,
  >2000× on a V100; learn how throughput comes from docking many ligands per launch.
- **Vina-GPU 2.1** (<https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1>) — GPU AutoDock
  Vina with RILC-BFGS; learn the Monte-Carlo + local-search strategy.
- **GNINA** (<https://github.com/gnina/gnina>) — CNN-scored docking; learn how a
  learned scorer slots into the same sampling loop.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

One thread per pose (independent jobs, grid-stride loop) · per-atom **gather with
trilinear interpolation** over the energy grid · shared `__host__ __device__` core
for exact CPU/GPU parity · **deterministic index-carrying min-reduction** (warp
`__shfl_down_sync` → shared memory → one integer `atomicMin` per block). See
`docs/PATTERNS.md` §1 (independent jobs), §2 (HD core), §3 (determinism).

## Exercises

1. **Texture memory.** Bind the grid to a `cudaTextureObject_t` and replace
   `trilinear_energy` with a `tex3D<float>` fetch — the hardware does the
   interpolation for free. Compare speed and accuracy (FP32 texture vs FP64 hand-roll).
2. **Refine the winner.** After the exhaustive search, add a few steps of local
   refinement (a tiny gradient descent on the score) around the best pose — the
   first taste of AutoDock's BFGS local search.
3. **Per-atom-type grids.** Give the ligand atoms types and load one grid per type
   (plus an electrostatics grid). Index the right grid per atom — the real
   force-field shape, with the same gather.
4. **Quaternion orientations.** Swap the three Euler angles for a quaternion sweep
   to sample rotations uniformly and avoid gimbal lock; confirm CPU/GPU parity holds.
5. **Scale & profile.** Run `make_synthetic.py --n-trans 25 --n-rot 8` (millions of
   poses) and watch the GPU's advantage over the CPU grow; profile in Nsight.

## Limitations & honesty

- The sample is **synthetic** — a Gaussian well, not a real receptor; the ligand
  and the score carry **no chemical or clinical meaning**.
- The ligand is **rigid** (no torsions), there is a **single generic** energy grid
  (no per-atom-type or electrostatics maps), and the search is **exhaustive on a
  lattice** (no genetic algorithm / BFGS). Real engines relax all three — THEORY.md
  §7 describes exactly what AutoDock-GPU / Vina-GPU add.
- The reported timing is a **teaching artifact, not a benchmark** (CLAUDE.md §12):
  at 9261 poses the GPU is launch/copy-bound; its edge appears at campaign scale.
- The grid is interpolated **by hand** (not in texture memory) on purpose, so the
  math is visible and runs byte-identically on the CPU.
