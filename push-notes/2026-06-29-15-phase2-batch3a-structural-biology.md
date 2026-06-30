# Push 2026-06-29 #15 -- phase2 batch3a structural-biology

> Push-note (CLAUDE.md section 7.1). First Phase-2 batch in **Domain 2 (Structural Biology
> & Protein Science)**: 6 Beginner projects, each worker-built and lead-verified.

## 1. Summary

The build-out crosses into its **second domain**. Six **domain-2 (structural biology)**
Beginner projects are complete, taking the collection to **48 -> 54 / 301 (17.9%)** and
domain 2 to **7/35** (the flagship `2.06` NMA was already done). This batch is a tour of
the field's headline GPU workloads: a transformer self-attention block (the heart of
AlphaFold-class inference), FFT-based rigid-body docking, cryo-EM single-particle
reconstruction, coarse-grained MD, a molecular ray tracer, and a Poisson-Boltzmann
electrostatics solver — five distinct GPU patterns in one batch. Each was built in its own
folder by one worker and independently re-verified by the lead.

## 2. What changed

Six new fully-implemented projects under `projects/02-structural-biology/`:

- [`2.01` Protein Structure Prediction Inference (AlphaFold-class)](../projects/02-structural-biology/2.01-protein-structure-prediction-inference-alphafold-class)
- [`2.02` Protein-Protein Docking](../projects/02-structural-biology/2.02-protein-protein-docking)
- [`2.03` Cryo-EM Single-Particle Reconstruction](../projects/02-structural-biology/2.03-cryo-em-single-particle-reconstruction)
- [`2.05` Coarse-Grained / MARTINI Simulation](../projects/02-structural-biology/2.05-coarse-grained-martini-simulation)
- [`2.08` GPU Molecular Visualization & Ray Tracing](../projects/02-structural-biology/2.08-gpu-molecular-visualization-ray-tracing)
- [`2.09` Solvent-Accessible Surface & Poisson-Boltzmann Electrostatics](../projects/02-structural-biology/2.09-solvent-accessible-surface-poisson-boltzmann-electrostatics)

`docs/STATUS.md` -> these 6 marked **done** (54/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **2.01 AlphaFold-class Inference** — one scaled-dot-product **self-attention head** (the
  core Evoformer/transformer op) as a CUDA kernel: one block per output row, a shared-memory
  **stable softmax** reduction, and same-order value accumulation for determinism. Shared
  HD math core -> CPU==GPU to ~5e-7. The clearest small example of "attention is a batched
  matmul + softmax" on the GPU.
- **2.02 Protein-Protein Docking** — Katchalski-Katzir / ZDOCK-style **FFT docking**: both
  proteins voxelized to a core/skin shape grid, the entire translational search scored at
  once via the **correlation theorem** with cuFFT (3-D R2C/C2R + two pointwise kernels),
  checked against a brute-force O(Ng^2) CPU correlation. Recovers the planted translation
  exactly. A second, different use of cuFFT (cf. 1.02 PME).
- **2.03 Cryo-EM Single-Particle Reconstruction** — the EM loop as two kernels: projection
  matching (E-step, one thread per particle, reference bank in constant memory — the
  catalog's O(N*M) bottleneck) and back-projection (M-step, one thread per output pixel,
  gather). Shared HD core makes assignments bit-exact (120/120) and the reconstruction exact.
- **2.05 Coarse-Grained / MARTINI MD** — two-bead-type (apolar/polar) Lennard-Jones CG-MD
  with periodic boundaries and velocity-Verlet; the classic independent N-body non-bonded
  pair-force kernel (one thread per bead, no atomics). Shared `martini.h` core -> CPU==GPU
  ~1.5e-11. A gentler cousin of the all-atom MD flagship 1.01.
- **2.08 Molecular Ray Tracer** — per-pixel gather (one thread per pixel, atoms in constant
  memory) rendering VDW spheres with deterministic Hammersley ambient occlusion + hard
  shadows; CPU and GPU share one `shade_pixel()`. Honest tolerance: ~2/40000 silhouette-edge
  pixels differ by <=2 grey levels (host/device transcendental last-bit divergence),
  documented rather than claimed bit-exact (PATTERNS.md §4).
- **2.09 Poisson-Boltzmann + SASA** — a linearized PB solver: 3-D 7-point finite-difference
  with a **red-black Gauss-Seidel** stencil (one thread per cell) plus Shrake-Rupley SASA.
  Shared per-cell HD core -> CPU==GPU to ~5.5e-17; the field is correctly antisymmetric for
  a synthetic dipole. A clean red-black stencil example (cf. flagship 6.04 / 14.02).

All six are clearly-labeled **reduced-scope teaching versions** (synthetic structures,
labeled synthetic), with the production tools (AlphaFold/OpenFold, ZDOCK/HADDOCK, RELION/
cryoSPARC, GROMACS-MARTINI, VMD/OptiX, APBS) named in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/02-structural-biology/2.02-protein-protein-docking   # (or any of the six)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

2.02 links cuFFT (`.lib` in both `<Link>` sections + `CMakeLists.txt`, BUILD_GUIDE §7b).

## 5. What to study here

Reading path: **2.05** (familiar N-body force) -> **2.09** (red-black stencil) -> **2.03**
(an EM/maximization loop as two kernels) -> **2.02** (FFT as a search accelerator) ->
**2.01** (attention = batched matmul + stable softmax) -> **2.08** (a renderer, and an honest
look at floating-point non-determinism at silhouette edges). Exercise: in **2.02**, shrink
the grid and compare the cuFFT correlation against the brute-force CPU one to see the
O(Ng^3 log Ng) vs O(Ng^6) crossover; in **2.01**, add a second attention head and confirm
the outputs stay deterministic.

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 6 project folders changed; no shared/root file; no artifacts.
- ✅ **Clean rebuild** (`/t:Rebuild`, committed fat arch list) of all 6 in both
  `Release|x64` and `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (12/12 builds).
- ✅ All 6 **demos PASS**: rebuilt exe reproduces `expected_output.txt`, GPU==CPU
  (cryo-EM assignments exact; attention 5e-7; docking best-pose identical; CG-MD 1.5e-11;
  PB 5.5e-17; ray tracer within the documented 2-px edge tolerance).
- ✅ `verify_project.py` -> **DONE** for all 6 (comment ratios **0.89–1.04**).
- **Workflow:** 6 agents, ~1.10M agent tokens, 516 tool uses.
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All six are **reduced-scope teaching versions**: a single attention head (not a full
  Evoformer), a small voxel docking grid, 2-D cryo-EM, two CG bead types, a handful of
  atoms in the ray tracer, a small PB grid. Labeled synthetic; production scale is described
  in each THEORY.md.
- 2.08 is intentionally **not bit-exact** GPU-vs-CPU (transcendental rounding at edges) — a
  teaching point about determinism limits, documented in its THEORY.md and demo tolerance.

## 8. Next push preview

Continue **domain 2** Beginners (`2.10` inverse folding, `2.19` membrane-protein sim, `2.22`
electron-density validation), then the Intermediate tier (`2.04`, `2.07`, `2.11`–`2.18`,
`2.20`, `2.21`, `2.23`–`2.29`, `2.31`, `2.33`) in ~6-project batches, then the 4 Advanced.
Same workflow, lead-verified, one push-note per batch.
