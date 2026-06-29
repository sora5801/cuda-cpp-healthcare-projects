# Push 2026-06-29 #10 -- phase2 batch2a drug-discovery

> Push-note (CLAUDE.md section 7.1). The second Phase-2 build-out batch: 6 more
> drug-discovery (domain 1) projects, each built by one worker agent to the
> Definition of Done, then independently re-verified by the lead and integrated here.

## 1. Summary

Six more **domain-1 (drug discovery)** projects are complete, taking the collection
from **20 -> 26 / 301 (8.6%)**. This batch deliberately reaches into the *harder,
more research-grade* corner of domain 1 — alchemical free energy, quantum chemistry,
graph neural networks — and shows how each still reduces to a clean, verifiable GPU
teaching pattern. Every project was built by its own worker agent in its own folder,
then the lead independently re-verified it from a clean state (boundary check, clean
Release+Debug rebuild with zero warnings, demo PASS, `verify_project.py` DONE, plus a
deep code spot-read). Two genuine numerical bugs were found and fixed during the build
(documented below) — exactly the kind of teaching moment this repo exists to capture.

## 2. What changed

Six new fully-implemented projects under `projects/01-drug-discovery/`:

- [`1.02` Particle-Mesh Ewald Electrostatics](../projects/01-drug-discovery/1.02-particle-mesh-ewald-electrostatics)
- [`1.03` Molecular Docking Engine](../projects/01-drug-discovery/1.03-molecular-docking-engine)
- [`1.05` Free-Energy Perturbation / Thermodynamic Integration](../projects/01-drug-discovery/1.05-free-energy-perturbation-thermodynamic-integration)
- [`1.07` Quantum Chemistry / DFT](../projects/01-drug-discovery/1.07-quantum-chemistry-dft)
- [`1.11` QSAR Property Prediction](../projects/01-drug-discovery/1.11-qsar-property-prediction)
- [`1.14` Conformer Ensemble Generation](../projects/01-drug-discovery/1.14-conformer-ensemble-generation)

`docs/STATUS.md` -> these 6 marked **done** (26/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **1.02 Particle-Mesh Ewald** — the long-range electrostatics workhorse of MD.
  Charges are spread to a grid with B-splines (atomic **fixed-point** accumulation
  for determinism), a **cuFFT** 3-D R2C transform takes it to reciprocal space, the
  reciprocal-energy convolution is applied there, and an inverse FFT returns. The most
  interesting thing: it is verified *three* ways — GPU-vs-CPU SPME (rel 6e-8), SPME vs
  the O(N^2) direct Ewald sum (rel 7e-12), and total-energy invariance to the Ewald
  splitting parameter beta (rel 3e-5). cuFFT is documented, not a black box.
- **1.03 Molecular Docking** — rigid-body pose scoring: one thread per candidate pose
  (grid-stride), each doing **trilinear interpolation** of a precomputed energy grid.
  The headline detail is the deterministic **index-carrying min-reduction**
  (warp-shuffle -> shared memory -> integer `atomicMin` on a packed score|index word) so
  the single best pose is found identically on CPU and GPU (energy error exactly 0).
- **1.05 Free-Energy Perturbation / TI** — alchemical free energy on a 1-D harmonic
  two-state model: one Metropolis Monte-Carlo chain per lambda-window mapped to one GPU
  thread (the *ensemble* pattern), with a **counter-based SplitMix64 RNG** so the GPU
  reproduces the CPU stream bit-for-bit (agree to 8e-14). TI gives DeltaG=0.711 vs the
  closed-form 1/2 kT ln(k_b/k_a)=0.693. **Bug found & fixed:** the RNG's float-scaling
  constant was 2x too large, biasing every uniform into [0,0.5) — written up as a
  teaching note in THEORY.md.
- **1.07 Quantum Chemistry / DFT** — restricted Hartree-Fock SCF on a minimal STO-3G
  basis. The GPU computes the O(N^4) two-electron repulsion integrals (**one thread per
  (i,j,k,l) integral**, shared `__host__ __device__` `eri_primitive` so CPU==GPU to
  ~1e-16) and **cuSOLVER `Dsygvd`** solves the per-cycle generalized eigenproblem
  `F C = S C eps`. H2 reproduces the textbook -1.11671432 Ha. Look at
  `cusolver_generalized()` — the library call is fully explained (itype/jobz/uplo, the
  math, the by-hand alternative).
- **1.11 QSAR Property Prediction** — a 2-layer Graph Convolutional Network
  (message-passing) over a batched-CSR molecular graph. One thread per atom does the
  neighbour **gather** (no atomics -> deterministic), weights live in constant memory,
  and a shared `__host__ __device__` core keeps CPU==GPU to ~6e-8. A clean intro to GNNs
  as sparse gather/scatter on the GPU.
- **1.14 Conformer Ensemble Generation** — the GPU enumerates, embeds (NeRF —
  Natural Extension Reference Frame) and scores 243 conformers of a flexible chain (one
  thread per conformer), then the CPU does greedy RMSD clustering (243 -> 81).
  **Bug found & fixed:** a bare (sigma/r)^12 Lennard-Jones term overflowed to ~1e14 and
  broke CPU/GPU agreement; a soft-core floor fixed it (agreement now 5.3e-12).

Each is a clearly-labeled **reduced-scope teaching version** (1.05 and 1.07 especially —
real FEP/DFT are research-grade); the full approach is described in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/01-drug-discovery/1.07-quantum-chemistry-dft   # (or any of the six)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

1.02 (cuFFT) and 1.07 (cuSOLVER) link a CUDA library — the `.lib` is added to both
`<Link>` sections of the `.vcxproj` and to `CMakeLists.txt` (see `docs/BUILD_GUIDE.md`
section 7b). No manual include/lib paths needed.

## 5. What to study here

Reading path: **1.11** (simplest — gather, no atomics) -> **1.03** (deterministic
min-reduction) -> **1.02** (library FFT in a real pipeline) -> **1.07** (two GPU jobs:
a hand-written N^4 kernel *and* a documented cuSOLVER eigensolve in one SCF loop).
Then compare **1.05** and **1.14**: both are "ensemble of independent simulations, one
per thread", and both have a THEORY.md note on a real floating-point bug caught by the
CPU/GPU cross-check. Exercises: (1) in 1.02, vary the grid spacing and watch the
SPME-vs-direct error change; (2) in 1.07, add a second SCF molecule (HeH+) to the sample
and confirm it still converges.

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** `git status` shows only the 6 project folders changed; no shared or
  root file touched; no build artifacts staged.
- ✅ **Clean rebuild** (`/t:Rebuild`, committed fat arch list `sm_75;sm_86;sm_89`+PTX) of
  all 6 in **both** `Release|x64` and `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** every
  time (12/12 builds).
- ✅ All 6 **demos PASS**: freshly-built exe reproduces the committed `expected_output.txt`
  and GPU==CPU (PME 3-way: rel <=2.6e-5; docking exact; FEP 8e-14 / TI 5e-2; DFT ~1e-16;
  QSAR 1e-4; conformers 5.3e-12).
- ✅ `verify_project.py` -> **DONE** for all 6 (comment ratios **0.63–1.12**, floor 0.40).
- ✅ **Deep spot-read** of 1.07: the cuSOLVER `Dsygvd` call and the ERI kernel are
  documented to standard (launch config, thread->data mapping, "no black box" library
  explanation, deterministic stdout / stderr-timing split).
- **Workflow:** 6 agents, ~1.13M agent tokens, 505 tool uses.
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All six are **reduced-scope teaching versions** (minimal STO-3G basis for DFT; 1-D
  harmonic alchemy for FEP; a small synthetic energy grid for docking; toy weights for the
  QSAR GNN). Synthetic data, labeled synthetic everywhere. The full research-grade methods
  are described in each `THEORY.md` "real world" section.
- **Process note (honest):** this batch was originally launched as one 11-project batch,
  but the account session limit was hit mid-run and killed all workers. The partial,
  unverified work was discarded (`git restore` + `git clean`) leaving the tree clean at
  20/301, and after the limit reset the batch was re-run in **smaller ~6-project
  sub-batches** to stay within one usage window. This batch (2a) is the first half;
  batch 2b (the remaining 5) follows next.

## 8. Next push preview

**Batch 2b** — the rest of the domain-1 Beginner projects: `1.6` enhanced-sampling
(metadynamics/replica exchange), `1.8` semi-empirical tight-binding, `1.9` ML interatomic
potentials, `1.24` umbrella sampling / WHAM, `1.27` MM-GBSA/MM-PBSA rescoring. Same
workflow, lead-verified, one push-note. Then on through domain-1 Intermediate and into
domain 2.
