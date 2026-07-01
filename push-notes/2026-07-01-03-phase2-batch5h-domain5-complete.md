# Push 2026-07-01 #03 -- phase2 batch5h domain5-complete

> Push-note (CLAUDE.md section 7.1). Third domain-5 batch — and a **milestone**: the deferred
> `5.5` plus the 5 Advanced projects bring **Domain 5 (Radiation Therapy & Medical Physics) to
> 15/15**. Each worker-built; independently lead-verified.

## 1. Summary

The account **weekly usage limit has reset**, and this batch **completes Domain 5 — Radiation
Therapy & Medical Physics, now 15/15** — the **fifth of 14 domains** finished — taking the
collection to **151 -> 157 / 301 (52.2%)**. It picks up the `5.5` project deferred by the
weekly limit and adds the domain's five Advanced projects: a stray-dose/secondary-cancer Monte
Carlo, track-structure microdosimetry, FLASH-RT chemistry, BNCT neutron transport, and proton-CT
reconstruction. The theme is **Monte-Carlo particle transport with integer-atomic scoring** —
four of the six are MC histories scored deterministically. Each was built in its own folder by
one worker and re-verified by the lead.

## 2. What changed

Six new projects under `projects/05-radiation-therapy-medphys/`:

- [`5.05` Deformable Dose Accumulation & Adaptive Radiotherapy](../projects/05-radiation-therapy-medphys/5.05-deformable-dose-accumulation-adaptive-radiotherapy)
- [`5.10` Secondary Cancer Risk & Stray-Dose Monte Carlo](../projects/05-radiation-therapy-medphys/5.10-secondary-cancer-risk-stray-dose-monte-carlo)
- [`5.11` Microdosimetry & Track-Structure Simulation](../projects/05-radiation-therapy-medphys/5.11-microdosimetry-track-structure-simulation)
- [`5.12` FLASH Radiotherapy GPU Modeling](../projects/05-radiation-therapy-medphys/5.12-flash-radiotherapy-gpu-modeling)
- [`5.13` BNCT Dose Calculation & Optimization](../projects/05-radiation-therapy-medphys/5.13-bnct-dose-calculation-optimization)
- [`5.15` Proton CT & Ion Imaging Reconstruction](../projects/05-radiation-therapy-medphys/5.15-proton-ct-ion-imaging-reconstruction)

`docs/STATUS.md` -> 6 marked **done** (157/301; **domain 5 = 15/15**). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **5.05 Deformable Dose Accumulation** — the full reduced-scope **ART pipeline**: GPU Thirion
  **Demons** DIR -> a DVF -> bilinear/trilinear **dose warp** -> summation-of-deformed-doses ->
  integer-atomic **DVH**. Shared `demons.h`/`dose.h` -> CPU==GPU ~1e-15 (DVF/dose) and exact (DVH).
  Accumulating dose across a deforming anatomy — the heart of adaptive RT.
- **5.10 Secondary-Cancer / Stray-Dose MC** — a GPU Monte-Carlo with **survival biasing, forced
  detection, Russian roulette**, scored into fixed-point integer per-organ tallies (atomicAdd,
  exact CPU==GPU), then **BEIR-VII** secondary-cancer risk. Variance-reduction MC on the GPU.
- **5.11 Microdosimetry (Track Structure)** — one thread per particle track, integer-atomic
  scoring of DNA damage (SSB/DSB) and a **lineal-energy spectrum f(y)** (shared `ts_physics.h` ->
  bit-identical). The nanodosimetry behind RBE models.
- **5.12 FLASH Radiotherapy** — the FLASH sparing effect as an **ensemble ODE**: one thread per
  tissue voxel integrates a coupled radical/oxygen **RK4** chemistry + Alper OER damage, sweeping
  pO2 x {conventional, FLASH} to reproduce the sparing signature. CPU==GPU 1e-14. (Same ensemble
  -RK4 pattern as flagship 9.02 / project 13.02, applied to radiobiology.)
- **5.13 BNCT Dose** — Monte-Carlo two-group **neutron transport** (1-D slab): per-thread neutron
  histories, integer keV-quanta atomicAdd scoring across four BNCT dose components (boron/nitrogen
  /gamma/fast) + CBE/RBE-weighted biological dose. Exact CPU==GPU. Boron neutron capture therapy.
- **5.15 Proton CT** — 2-D proton-CT **SART** reconstruction: one thread per proton with an MLP
  (most-likely-path) forward/backprojection and int64 fixed-point atomic reduction (deterministic,
  ~9.5e-7). Recovers a known relative-stopping-power phantom (RMSE 0.094).

All six are clearly-labeled **reduced-scope teaching versions** (synthetic phantoms, labeled
synthetic), with production tools (RayStation/velocity ART, Geant4/TOPAS stray dose, TRAX/
Geant4-DNA, FLASH radiochemistry models, MCNP/Geant4 BNCT, proton-CT MLP reconstructions) named
in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/05-radiation-therapy-medphys/5.12-flash-radiotherapy-gpu-modeling   # (or any)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

No new CUDA libraries this batch — all custom kernels (Monte-Carlo histories with integer
atomics, Demons stencil, ensemble RK4, SART).

## 5. What to study here

Domain 5 is now a **complete worked tour** of GPU medical physics. Across the domain: dose
engines in three philosophies — **Monte Carlo** (flagship 5.01, 5.10, 5.11, 5.13),
**convolution** (5.04 collapsed-cone), **deterministic transport** (5.06 S_N); plus optimization
(5.02 cuSPARSE), analytic dose (5.03, 5.07 TG-43), QA (5.08, 5.09 gamma), registration/adaptive
(5.05, 5.14), and reconstruction (5.15). Exercise: compare `5.01` (MC) and `5.06` (S_N) on the
same 1-D problem — stochastic noise vs. deterministic discretization error.

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 6 project folders changed; no shared/root file; no artifacts.
- ✅ **Clean rebuild** (`/t:Rebuild`, fat arch list) of all 6 in both `Release|x64` and
  `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (12/12 builds).
- ✅ All 6 **demos PASS**: GPU==CPU (MC tallies / DVH / BNCT exact; FLASH 1e-9; ADR dose 1e-9;
  proton-CT 1e-3).
- ✅ `verify_project.py` -> **DONE** for all 6 (comment ratios **0.84–1.13**).
- ✅ **Domain-5 sweep:** 15/15 markers `done`.
- **Workflow:** 6 agents, ~1.14M agent tokens, 505 tool uses (this batch also confirmed the
  weekly usage limit had reset — the previous attempt at `5.5` had been killed by it).
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All six are **reduced-scope teaching versions**: 1-D/2-D transport, small MC history counts,
  simplified radiochemistry. Labeled synthetic; production scale described in each THEORY.md.

## 8. Next push preview

**Domain 6 — Computational Physiology & Systems Biology (27 projects).** Flagship `6.04`
(Lattice-Boltzmann blood/airflow solver) is already done; the build-out continues easiest-first
through the remaining 26 in ~6-project batches. **Five of 14 domains complete** (125 domain
projects + 14 flagships = 157). Same workflow, lead-verified, one push-note per batch.
