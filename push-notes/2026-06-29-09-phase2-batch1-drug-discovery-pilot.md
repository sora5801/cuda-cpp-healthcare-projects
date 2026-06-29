# Push 2026-06-29 #09 -- phase2 batch1 drug-discovery pilot

> Push-note (CLAUDE.md §7.1). **Phase 2 begins** — the first parallel build-out batch. 6 worker agents each
> built one drug-discovery (domain 1) project to the Definition of Done; the lead verified and integrates here.

## 1. Summary

The Phase-2 multi-agent workflow is validated: **6 domain-1 projects** built in parallel (one worker agent
each), every one independently re-verified by the lead — clean Release+Debug builds, demo PASS, GPU==CPU,
`verify_project.py` DONE. Project count **14 → 20 / 301 (6.6%)**. The pilot confirms the §10 model (one agent
↔ one folder, lead integrates) and the PATTERNS.md cookbook work in practice.

## 2. What changed

Six new fully-implemented projects under `projects/01-drug-discovery/`:
- [`1.01` Molecular Dynamics Engine](../projects/01-drug-discovery/1.01-molecular-dynamics-engine)
- [`1.04` Ultra-Large Virtual Screening](../projects/01-drug-discovery/1.04-ultra-large-virtual-screening)
- [`1.13` Pharmacophore / 3D Shape Screening](../projects/01-drug-discovery/1.13-pharmacophore-3d-shape-screening)
- [`1.16` ADMET / Toxicity Prediction](../projects/01-drug-discovery/1.16-admet-toxicity-prediction)
- [`1.30` Trajectory RMSD, Clustering & Contact Analysis](../projects/01-drug-discovery/1.30-trajectory-rmsd-clustering-contact-analysis)
- [`1.31` Solvent-Accessible Surface Area (SASA)](../projects/01-drug-discovery/1.31-solvent-accessible-surface-area-sasa-on-gpu)

`docs/STATUS.md` → these 6 marked **done** (20/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb)

- **1.01 MD** — Lennard-Jones velocity-Verlet engine: shared `__host__ __device__` physics core, a
  shared-memory-tiled all-pairs force kernel (thread per atom), deterministic host-order energy reduction.
  Energy conserved (rel drift ~2.5e-8); CPU/GPU agree ~1e-13.
- **1.04 Virtual screening** — GPU-batched per-ligand pipeline (Lipinski/Veber filter + integer surrogate
  dock score, popcount-overlap motif from 1.12), target in constant memory, grid-stride, host top-K; exact
  integer scores → tolerance 0.
- **1.13 3D shape screening** — Gaussian-volume ROCS-style Shape Tanimoto, one thread per conformer, query
  in constant memory; an exact-copy conformer scores exactly 1.0 as a built-in check (CPU/GPU ~3e-16).
- **1.16 ADMET** — multi-task logistic-regression screen (N molecules × 12 endpoints), models in constant
  memory, integer-atomic flag-count reduction; CPU/GPU ~5e-16, flag counts exact.
- **1.30 RMSD/clustering** — one thread per frame, optimal-superposition RMSD via the SVD-free **QCP** method
  + native-contact fraction Q; QCP validated against numpy SVD ground truth (max err 3e-12).
- **1.31 SASA** — Shrake-Rupley, one thread per atom with shared-memory neighbour tiling; exact integer
  exposed-point counts → GPU==CPU exactly.

Each is a clearly-labeled **reduced-scope teaching version** where the catalog problem is research-grade,
with the full approach described in its THEORY.md.

## 4. How to build & run

```powershell
cd projects/01-drug-discovery/1.01-molecular-dynamics-engine   # (or any of the six)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

## 5. What to study here

Each project's `THEORY.md` + `src/`; together they show how one domain reuses several GPU patterns
(`PATTERNS.md`): per-item batched scoring (1.04, 1.13, 1.16), N-body forces + shared-memory tiling (1.01,
1.31), and per-frame reductions (1.30). All use the shared `__host__ __device__` core idiom for CPU/GPU parity.

## 6. Verification (lead-independent, not self-reports)

- ✅ Boundaries: `git status` shows **only the 6 project folders** changed; no shared/root file touched.
- ✅ **Clean Release rebuild** (`/t:Rebuild`) of all 6: **0 errors, 0 warnings**.
- ✅ All 6 **demos PASS**: the freshly-rebuilt exe reproduces the committed `expected_output.txt`, GPU==CPU
  (MD energy dE≤1e-6; screening/SASA/ADMET-flags exact; shape/RMSD tol 1e-9).
- ✅ `verify_project.py` → **DONE** for all 6 (comment ratios 0.89–1.14).
- **Workflow:** 6 agents, ~17 min wall-clock, ~1.03M agent tokens, 455 tool uses.
- **Environment:** RTX 2080 (SUPER), CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All six are reduced-scope teaching versions (no real force fields / docking engines / trained ADMET
  models / experimental trajectories); synthetic data, labeled synthetic.
- Pilot batch size was 6 (deliberately small to validate quality); subsequent batches scale to ~10–14.

## 8. Next push preview

Continue Phase 2: the **remaining domain-1 Beginner projects** (1.2, 1.3, 1.5–1.9, 1.11, 1.14, 1.24, 1.27)
in a larger batch, then 🟡 Intermediate, then on through the domains — same workflow, lead-verified, one
push-note per batch.
