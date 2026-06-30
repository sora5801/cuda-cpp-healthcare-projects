# Push 2026-06-30 #02 -- phase2 batch3f domain2-complete

> Push-note (CLAUDE.md section 7.1). Sixth domain-2 batch — and a **milestone**: the 4
> Advanced projects that bring **Domain 2 (Structural Biology & Protein Science) to 35/35**.

## 1. Summary

Four Advanced projects complete **Domain 2 — Structural Biology & Protein Science, now
35/35** — the **second of 14 domains** finished end-to-end — and take the collection to
**78 -> 82 / 301 (27.2%)**. This batch is the "frontier biophysics" cluster: IDP
liquid-liquid phase separation, transition-path sampling of folding, an active-learning
condensate-design loop, and EPR/DEER-constrained modeling. Each was built by one worker in
its own folder and independently re-verified by the lead. Every domain-2 project builds clean
in `Debug|x64` + `Release|x64`, ships a CPU reference and a one-command demo that verifies
GPU==CPU, and passes `verify_project.py`.

## 2. What changed

Four new fully-implemented projects under `projects/02-structural-biology/`:

- [`2.30` Protein Solubility & Phase Separation Simulation](../projects/02-structural-biology/2.30-protein-solubility-phase-separation-simulation)
- [`2.32` Protein Folding Pathway Extraction (Transition Path Sampling)](../projects/02-structural-biology/2.32-protein-folding-pathway-extraction-transition-path-sampling)
- [`2.34` Biophysical Simulation of Biomolecular Condensates (Active Learning)](../projects/02-structural-biology/2.34-biophysical-simulation-of-biomolecular-condensates-active-learning-loop)
- [`2.35` EPR / DEER Constrained Modeling](../projects/02-structural-biology/2.35-electron-paramagnetic-resonance-epr-deer-constrained-modeling)

`docs/STATUS.md` -> these 4 marked **done** (82/301; **domain 2 = 35/35**). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **2.30 Phase Separation (HPS)** — a Hydrophobicity-Scale coarse-grained MD of IDP
  liquid-liquid phase separation: one-thread-per-bead all-pairs gather force + velocity-Verlet
  (shared `hps_model.h`, CPU==GPU ~1e-15). Six synthetic sticky chains coalesce into a single
  droplet (all 36 beads condensed). Builds directly on the CG-MD pattern (2.05, 2.19).
- **2.32 Transition Path Sampling** — 1-D Brownian dynamics on a double-well folding
  landscape, one independent **shooter** per GPU thread (per-thread RNG + integer atomic
  tallies), shared core -> exact CPU==GPU. The recovered **committor** p_B crosses 0.5 right at
  the barrier top — a textbook TPS result, on the GPU.
- **2.34 Condensate Active-Learning Loop** — per-candidate coarse-grained Brownian dynamics
  (one thread per replica) measuring radius of gyration + lag-MSD diffusion, then a
  deterministic **argmin acquisition** proposing the next sequence to test. CPU==GPU ~1.3e-15.
  A compact picture of simulation-in-the-loop design.
- **2.35 EPR / DEER Constrained Modeling** — a one-frame-per-thread DEER **rotamer-convolution
  back-calculation** kernel + shared-host **maximum-entropy (BioEn/EROS) reweighting**
  (shared `deer.h`, bit-for-bit CPU==GPU). The synthetic sample's true-frame population is
  recovered 0.25 -> 0.99 by the reweighting. Ties experimental restraints back to an ensemble.

All four are clearly-labeled **reduced-scope teaching versions** (CLAUDE.md §13) with the
production methods (HOOMD-blue/openMM-HPS, OpenPathSampling, Bayesian active learning,
DEER-based BioEn/MMM) described in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/02-structural-biology/2.32-protein-folding-pathway-extraction-transition-path-sampling
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

No new CUDA libraries this batch — all custom kernels (ensemble Brownian dynamics, all-pairs
forces, convolution back-calculation).

## 5. What to study here

Domain 2 is now a **complete worked tour** of GPU patterns in structural biology. Across the
domain: per-item scoring (2.15, 2.29, 2.33), cuFFT pipelines (2.02, 2.04, 2.11, 2.22, 2.31),
cuSOLVER eigensolves (2.06, 2.20), self-attention / FlashAttention (2.01, 2.14), the
ensemble-per-thread family (2.07, 2.18, 2.28, 2.30, 2.32, 2.34, 2.35), iterative solvers
(2.27), stencils (2.09), and grid-accumulation with atomics (2.26). Exercise: pick a pattern
from `docs/PATTERNS.md` and read its two or three domain-2 instances side by side.

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 4 project folders changed; no shared/root file; no artifacts.
- ✅ **Clean rebuild** (`/t:Rebuild`, fat arch list) of all 4 in both `Release|x64` and
  `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (8/8 builds).
- ✅ All 4 **demos PASS**: GPU==CPU (TPS tally exact; HPS/condensate/DEER ~1e-6..1e-15).
- ✅ `verify_project.py` -> **DONE** for all 4 (comment ratios **0.83–1.17**).
- ✅ **Domain-2 sweep:** 35/35 markers `done`.
- **Workflow:** 4 agents, ~0.76M agent tokens, 343 tool uses.
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All four are **reduced-scope teaching versions** with synthetic data, labeled synthetic
  (toy IDP chains, 1-D landscapes, a small candidate pool, a few spin-label frames). The
  research-grade gap is documented in each THEORY.md.

## 8. Next push preview

**Domain 3 — Genomics, Sequencing & Bioinformatics (30 projects).** Flagship `3.01`
(Smith-Waterman / Needleman-Wunsch) is already done; the build-out continues easiest-first
through the remaining 29 in ~6-project batches. Two of 14 domains complete (70 projects);
12 domains (219 projects) to go. Same workflow, lead-verified, one push-note per batch.
