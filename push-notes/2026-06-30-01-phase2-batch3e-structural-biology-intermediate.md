# Push 2026-06-30 #01 -- phase2 batch3e structural-biology intermediate

> Push-note (CLAUDE.md section 7.1). Fifth domain-2 batch: the last 6 Intermediate
> structural-biology projects, each worker-built and independently lead-verified.

## 1. Summary

Six more **domain-2 (structural biology) Intermediate** projects are complete, taking the
collection to **72 -> 78 / 301 (25.9%) — past the quarter mark** — and finishing the entire
**domain-2 Intermediate tier** (only the 4 Advanced projects remain in domain 2). This batch
is heavy on **solvation, polarization, and enhanced sampling**: GIST water placement,
polarizable-water SCF, REST2 replica exchange, ion permeation, cryo-ET tomogram
reconstruction, and pharmacophore screening. It exercises two patterns prominently — the
**iterative self-consistent solver** (2.27) and **grid-accumulation with atomics** (2.26,
2.29). Each was built in its own folder by one worker and re-verified by the lead.

## 2. What changed

Six new fully-implemented projects under `projects/02-structural-biology/`:

- [`2.26` Hydrogen Bond Network & Water Placement (GIST)](../projects/02-structural-biology/2.26-hydrogen-bond-network-water-placement-analysis)
- [`2.27` Polarizable Water Model GPU Dynamics](../projects/02-structural-biology/2.27-polarizable-water-model-gpu-dynamics)
- [`2.28` Replica Exchange Solute Tempering (REST2)](../projects/02-structural-biology/2.28-replica-exchange-solute-tempering-rest2-on-gpu)
- [`2.29` Ion Channel Gating & Permeation Simulation](../projects/02-structural-biology/2.29-ion-channel-gating-permeation-simulation)
- [`2.31` Cryo-EM Tilt-Series Alignment & Tomogram Reconstruction](../projects/02-structural-biology/2.31-cryo-em-tilt-series-alignment-tomogram-reconstruction)
- [`2.33` Structure-Based Pharmacophore Modeling from MD](../projects/02-structural-biology/2.33-structure-based-pharmacophore-modeling-from-md-ensembles)

`docs/STATUS.md` -> these 6 marked **done** (78/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **2.26 GIST Water Placement** — Grid Inhomogeneous Solvation Theory: a **grid-accumulation
  -with-atomics** kernel (one thread per water×frame sample) scatters occupancy + fixed-point
  energy into voxels, then derives per-voxel density/ΔE/−TΔS/ΔG and ranks displaceable
  hydration sites. CPU==GPU *exactly* via commuting integer atomics; two planted ordered
  waters recovered at ranks 1-2. (CUDA gotcha fixed: device-used constants must be
  `constexpr`, not `static const`.)
- **2.27 Polarizable Water Dynamics** — the self-consistent **induced-dipole (Jacobi SCF)**
  solver at the heart of AMOEBA/SWM4-NDP/MB-pol: iterative relaxation + N-body field, one
  thread per site, ping-pong dipole buffers, Thole damping, fixed-point atomics for
  deterministic reductions. CPU==GPU to ~2e-16 (dipoles), plus an analytic mu=alpha*E check.
  A second iterative-solver example after 2.21's CG... err, 1.21's matrix-free CG.
- **2.28 REST2** — a λ-ladder of replicas (one thread each) sampling a tilted double-well by
  Metropolis MC with the faithful REST2 effective Hamiltonian (λ·E_pp + √λ·E_pw + E_ww) and
  exchange criterion. The cold replica escapes 0/8 -> 8/8 to the global well. `--fmad=false`
  set for CPU/GPU parity; verified on robust observables (well occupancy, acceptance).
- **2.29 Ion Channel Permeation** — Brownian-dynamics multi-ion permeation: one thread per
  independent ion trajectory through a 1-D pore (Gaussian PMF barrier + applied voltage),
  with an integer occupancy histogram and forward/reverse crossing counts via atomicAdd.
  Shared Ermak-McCammon step + SplitMix64 RNG -> bit-identical CPU==GPU.
- **2.31 Cryo-ET Tomogram Reconstruction** — a tilt-series pipeline: sequential
  cross-correlation **alignment** (recovers injected drift to ~1 bin), a **cuFFT ramp filter**
  (R2C -> |f| -> C2R), and a per-pixel **weighted back-projection** gather (CPU==GPU ~1e-6).
  Ties the cryo-EM thread (2.03/2.04/2.11/2.20) to tomography + the FDK idea from flagship 4.01.
- **2.33 Pharmacophore Modeling** — a ROCS-style Gaussian-overlap "color" Tanimoto screen:
  one thread per library molecule, query in constant memory, variable-length library in a
  flat **CSR** layout, shared scoring core -> CPU==GPU exact. Planted target recovered at #1.

All six are clearly-labeled **reduced-scope teaching versions** (synthetic systems, labeled
synthetic), with production tools (SSTMap/GIST, Tinker-HP/OpenMM-AMOEBA, GROMACS-REST2,
BROWNDYE, IMOD/AreTomo, RDKit/OpenEye-ROCS) named in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/02-structural-biology/2.27-polarizable-water-model-gpu-dynamics   # (or any)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

2.31 links cuFFT; 2.28 sets `--fmad=false` (FP parity). Both noted in their `.vcxproj`/CMake.

## 5. What to study here

Reading path: **2.33** (CSR per-item scoring) -> **2.26** / **2.29** (grid/histogram
accumulation with commuting integer atomics) -> **2.31** (a three-stage cryo-ET pipeline) ->
**2.28** (REST2 effective Hamiltonian) -> **2.27** (the iterative SCF dipole solver — the
conceptual peak of the batch). Exercise: in **2.27**, change the Jacobi SCF tolerance and
watch iterations vs. accuracy; in **2.26**, move a planted ordered water and confirm it still
ranks in the top sites.

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 6 project folders changed; no shared/root file; no artifacts.
- ✅ **Clean rebuild** (`/t:Rebuild`, fat arch list) of all 6 in both `Release|x64` and
  `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (12/12 builds).
- ✅ All 6 **demos PASS**: GPU==CPU (GIST/ion-permeation exact; polarizable/pharmacophore
  1e-5..1e-9; tilt-series 1e-3; REST2 on robust observables as documented).
- ✅ `verify_project.py` -> **DONE** for all 6 (comment ratios **0.85–1.09**).
- **Workflow:** 6 agents, ~1.21M agent tokens, 551 tool uses.
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All six are **reduced-scope teaching versions**: a small GIST grid, a handful of polarizable
  sites, an 8-bead REST2 toy, a 1-D ion pore, a 2-D cryo-ET slice, a small pharmacophore
  library. Labeled synthetic; production scale described in each THEORY.md.
- 2.28 (REST2) verifies on *robust observables* rather than bit-exactness — like 1.06, the
  underlying dynamics are chaotic; documented honestly in its THEORY.md.

## 8. Next push preview

The **4 Advanced** domain-2 projects (`2.30` protein solubility / phase separation, `2.32`
folding-pathway extraction, `2.34` biomolecular condensates, `2.35` EPR/DEER) — completing
**domain 2 (35/35)** — then on to **domain 3 (genomics)**. Same workflow, lead-verified.
