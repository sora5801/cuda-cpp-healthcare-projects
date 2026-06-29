# Push 2026-06-29 #14 -- phase2 batch2e domain1-complete

> Push-note (CLAUDE.md section 7.1). Sixth Phase-2 batch — and a **milestone**: the last 5
> domain-1 projects, which bring **Domain 1 (Drug Discovery & Molecular Design) to 35/35**.

## 1. Summary

Five projects (the last Intermediate plus all 4 Advanced) complete **Domain 1 — Drug
Discovery & Molecular Design, now 35/35** — and take the collection to **43 -> 48 / 301
(15.9%)**. Domain 1 is the first of 14 domains finished end-to-end: every project builds
clean in `Debug|x64` and `Release|x64`, ships a CPU reference and a one-command demo that
verifies GPU==CPU, and passes `verify_project.py`. This batch's five span kinase
selectivity panels, alchemical hydration free energy, interaction-fingerprint clustering,
sequence-based amyloid aggregation, and hybrid ML/MM dynamics. Each was built by one worker
in its own folder and independently re-verified by the lead.

## 2. What changed

Five new fully-implemented projects under `projects/01-drug-discovery/`:

- [`1.29` Kinase Selectivity Panel Scoring](../projects/01-drug-discovery/1.29-kinase-selectivity-panel-scoring)
- [`1.32` Alchemical Hydration Free Energy (ΔGsolv)](../projects/01-drug-discovery/1.32-alchemical-hydration-free-energy-gsolv)
- [`1.33` Interaction Fingerprinting & Binding-Mode Clustering](../projects/01-drug-discovery/1.33-interaction-fingerprinting-binding-mode-clustering)
- [`1.34` Amyloid Aggregation-Propensity Prediction](../projects/01-drug-discovery/1.34-amyloid-aggregation-propensity-prediction)
- [`1.35` QM/MM + ML-Potential Hybrid MD](../projects/01-drug-discovery/1.35-qmmm-ml-potential-hybrid-md)

`docs/STATUS.md` -> these 5 marked **done** (48/301; **domain 1 = 35/35**). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **1.29 Kinase Selectivity Panel** — one thread per kinase scores the query compound's
  interaction fingerprint (IFP) against each panel member, with the query in constant
  memory and an **integer** scoring core, so the deterministic per-kinase S-score is
  CPU==GPU *exactly*. A clean "broadcast one query across a panel" pattern (cf. flagship 1.12).
- **1.32 Alchemical Hydration Free Energy** — a faithful (if reduced) **TI + BAR**
  free-energy calculation: soft-core Lennard-Jones coupling across lambda-windows sampled
  by Metropolis MC, one thread per (window, walker) chain. Shared `alchemy.h` HD-core gives
  CPU==GPU to ~1.5e-11. The soft-core LJ (no r^-12 singularity as the particle vanishes) is
  the detail to study — it is exactly the fix for the FMA-blowup seen in 1.14/1.28.
- **1.33 Interaction Fingerprinting + Clustering** — a two-stage GPU pipeline: generate
  96-bit IFPs (one thread per pose evaluates distance criteria over the residue grid), then
  consensus-bit **Tanimoto k-means** for binding-mode clustering (assign + integer-atomic
  tally + host majority vote). All integer/bit logic -> GPU==CPU bit-exact.
- **1.34 Amyloid Aggregation Propensity** — a TANGO/AGGRESCAN-style sequence scanner:
  per-residue propensity lookup + a **shared-memory-tiled sliding-window mean** (one block
  per protein, one thread per residue) + threshold segmentation into aggregation-prone
  regions, batched over proteins. Exact CPU/GPU parity. A nice shared-memory-stencil example.
- **1.35 Hybrid QM/MM + ML-Potential MD** — a Behler-Parrinello descriptor + tiny tanh MLP
  (analytic forces) for the ML region, Lennard-Jones for the MM region, mechanical
  embedding across a link atom, velocity-Verlet, run as a GPU ensemble. CPU==GPU ~4e-13.
  Ties together the NNP (1.09) and QM/MM (1.23) ideas into one hybrid force loop.

All five are clearly-labeled **reduced-scope teaching versions** (CLAUDE.md §13), with the
production methods described in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/01-drug-discovery/1.32-alchemical-hydration-free-energy-gsolv   # (or any)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

## 5. What to study here

Domain 1 is now a **complete worked tour** of GPU patterns in drug discovery. A good
end-to-end reading path across the whole domain: per-item batched scoring (1.12 -> 1.04,
1.13, 1.16, 1.29), N-body + tiling (1.01, 1.31), library FFT/eigensolve pipelines (1.02
cuFFT, 1.07/1.08 cuSOLVER), the ensemble-per-thread family (1.05, 1.06, 1.21–1.26, 1.32,
1.35), and analysis pipelines (1.17 MSM, 1.33 IFP-clustering). Exercise: pick any two
projects that share a pattern from `docs/PATTERNS.md` and diff their kernels to see how the
same GPU idea is re-skinned for a different problem.

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 5 project folders changed; no shared/root file; no artifacts.
- ✅ **Clean rebuild** (`/t:Rebuild`, committed fat arch list) of all 5 in both
  `Release|x64` and `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (10/10 builds).
- ✅ All 5 **demos PASS**: rebuilt exe reproduces `expected_output.txt`, GPU==CPU
  (kinase/IFP/amyloid integer fields exact; hydration FE 1e-9; hybrid MD 4e-13).
- ✅ `verify_project.py` -> **DONE** for all 5 (comment ratios **0.75–1.02**).
- ✅ **Domain-1 sweep:** 35/35 markers `done`.
- **Workflow:** 5 agents, ~0.89M agent tokens, 391 tool uses.
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All five are **reduced-scope teaching versions** with synthetic data, labeled synthetic.
  The four Advanced ones especially (TI+BAR hydration FE, hybrid ML/MM MD) are minimal
  stand-ins for research-grade pipelines; the gap is documented honestly in each THEORY.md.
- Domain 1 used **8 of the ~13 catalogued GPU patterns**; the remaining domains will
  exercise the rest (texture memory, warp-level scan/sort, multi-GPU) — `PATTERNS.md` will
  be extended as those appear.

## 8. Next push preview

**Domain 2 — Structural Biology & Protein Science (35 projects).** Flagship `2.06` (Normal
Mode Analysis / cuSOLVER) is already done; the build-out continues easiest-first through the
remaining 34 in ~6-project batches, same workflow, lead-verified, one push-note per batch.
This is the first of 13 remaining domains (253 projects to go).
