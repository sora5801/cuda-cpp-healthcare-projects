# Push 2026-06-30 #00 -- phase2 batch3d structural-biology intermediate

> Push-note (CLAUDE.md section 7.1). Fourth domain-2 batch: 6 more Intermediate
> structural-biology projects, each worker-built and independently lead-verified.

## 1. Summary

Six more **domain-2 (structural biology) Intermediate** projects are complete, taking the
collection to **66 -> 72 / 301 (23.9%)** and domain 2 to **31/35** — the Intermediate tier
is nearly finished. This batch covers experimental-data-driven refinement and analysis: NMR
restrained annealing, heterogeneous cryo-EM (3D variability / PCA), protein-nucleic-acid
docking, per-residue interaction-energy decomposition, SAXS/SANS forward modeling, and
coevolutionary contact prediction. The unifying GPU theme is **"score/accumulate over an
independent grid or ensemble, then reduce"** — five of the six are atomics-free and exactly
CPU==GPU. Each was built in its own folder by one worker and re-verified by the lead.

## 2. What changed

Six new fully-implemented projects under `projects/02-structural-biology/`:

- [`2.18` NMR Structure Refinement](../projects/02-structural-biology/2.18-nmr-structure-refinement)
- [`2.20` Heterogeneous Cryo-EM Reconstruction (3D Variability)](../projects/02-structural-biology/2.20-heterogeneous-cryo-em-reconstruction-3d-variability)
- [`2.21` Protein-Nucleic-Acid Docking & Co-Folding](../projects/02-structural-biology/2.21-protein-nucleic-acid-docking-co-folding)
- [`2.23` Protein-Ligand Interaction Energy Decomposition](../projects/02-structural-biology/2.23-protein-ligand-interaction-energy-decomposition)
- [`2.24` SAXS / SANS Data-Driven Structure Modeling](../projects/02-structural-biology/2.24-saxs-sans-data-driven-structure-modeling)
- [`2.25` Coevolutionary Contact Prediction](../projects/02-structural-biology/2.25-coevolutionary-contact-prediction-msa-transformer)

`docs/STATUS.md` -> these 6 marked **done** (72/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **2.18 NMR Structure Refinement** — restrained **simulated annealing** as an ensemble of
  independent Metropolis annealers (one thread per replica, shared RNG+energy+SA core), so
  GPU matches CPU (restraint counts exact, energy ~1e-14). The best of 512 replicas satisfies
  all 19 synthetic NOE distance restraints. Same ensemble-MC shape as 2.07, applied to NMR.
- **2.20 Heterogeneous Cryo-EM (3DVA)** — principal-component analysis over N particle
  volumes via per-element GPU kernels (mean, NxN Gram matrix, PC lift, projections) +
  **cuSOLVER `Dsyevd`** for the eigendecomposition (shared math core, CPU==GPU ~1e-15). The
  synthetic sample's hidden conformational motion is recovered at |corr|=0.997. A clean
  GPU-PCA + library-eigensolver pipeline.
- **2.21 Protein-Nucleic-Acid Docking** — rigid-body pose search (one thread per pose over
  24 cube-group orientations x a translation lattice) scoring contacts + electrostatics -
  clashes. **All integer fixed-point** in a shared core, so CPU==GPU is *exact* (648/648
  poses); the planted native pose is recovered as #1.
- **2.23 Interaction Energy Decomposition** — per-residue **MM-GBSA decomposition**: one
  thread per residue accumulates Coulomb + LJ + Generalized-Born over all frames/ligand atoms
  (no atomics, deterministic). The synthetic sample embeds a known answer (ARG41 salt-bridge
  electrostatic hot spot #1, LEU88 vdW hot spot #2). Builds on the 1.27 MM-GBSA project.
- **2.24 SAXS / SANS Forward Modeling** — a custom O(N^2) **Debye-summation** kernel (one
  thread per q value, shared `saxs_core.h`, CPU==GPU 7.5e-16). Forward-models I(q) from a
  40-atom structure, recovers the Guinier Rg (13.82 vs 13.67 A), and reports a reduced-chi^2 fit.
- **2.25 Coevolutionary Contact Prediction** — pairwise **Mutual Information + Average
  Product Correction** over MSA columns (one thread per column pair, integer counts ->
  deterministic, CPU==GPU 4.4e-16). The synthetic MSA's four planted contacts rank #1-4.
  A statistical-coupling cousin of the MSA-Transformer idea.

All six are clearly-labeled **reduced-scope teaching versions** (synthetic structures/MSAs/
volumes, labeled synthetic), with production tools (Xplor-NIH/ARIA, cryoSPARC-3DVA,
HADDOCK, MMPBSA.py, CRYSOL/FoXS, EVcouplings/CCMpred) named in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/02-structural-biology/2.24-saxs-sans-data-driven-structure-modeling   # (or any)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

2.20 links cuSOLVER (`.lib` in both `<Link>` sections + `CMakeLists.txt`, BUILD_GUIDE §7b).

## 5. What to study here

Reading path: **2.24** (the cleanest O(N^2) Debye sum) -> **2.25** (pairwise MI matrix) ->
**2.23** (per-residue energy accumulation) -> **2.21** (integer pose search) -> **2.18**
(ensemble annealing) -> **2.20** (GPU PCA + cuSOLVER eigensolve). Exercise: in **2.24**,
change the synthetic structure's radius of gyration and confirm the Guinier fit tracks it; in
**2.25**, add a fifth correlated column pair to the MSA and check it appears in the top contacts.

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 6 project folders changed; no shared/root file; no artifacts.
- ✅ **Clean rebuild** (`/t:Rebuild`, fat arch list) of all 6 in both `Release|x64` and
  `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (12/12 builds).
- ✅ All 6 **demos PASS**: GPU==CPU (NA-docking exact; energy/SAXS/MI ~1e-9..1e-15;
  NMR/decomp 1e-4; 3DVA within tol).
- ✅ `verify_project.py` -> **DONE** for all 6 (comment ratios **0.71–1.11**).
- **Workflow:** 6 agents, ~1.00M agent tokens, 433 tool uses.
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All six are **reduced-scope teaching versions**: 512-replica NMR annealing, small particle
  sets for 3DVA, a coarse pose lattice, single-structure SAXS, a small synthetic MSA. Labeled
  synthetic; production scale is described in each THEORY.md.

## 8. Next push preview

Finish **domain-2 Intermediates** (`2.26` H-bond/water networks, `2.27` polarizable water,
`2.28` REST2, `2.29` ion-channel gating, `2.31` cryo-EM tilt-series alignment, `2.33`
pharmacophore from MD), then the 4 **Advanced** (`2.30`, `2.32`, `2.34`, `2.35`) to complete
**domain 2 (35/35)**. Then domain 3 (genomics). Same workflow, lead-verified.
