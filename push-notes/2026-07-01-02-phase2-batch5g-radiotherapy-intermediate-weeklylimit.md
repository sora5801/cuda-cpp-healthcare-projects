# Push 2026-07-01 #02 -- phase2 batch5g radiotherapy intermediate (weekly-limit pause)

> Push-note (CLAUDE.md section 7.1). Second domain-5 batch: 4 of 5 Intermediate
> radiation-therapy projects (the 5th deferred by an account **weekly** usage limit). Each
> worker-built and independently lead-verified.

## 1. Summary

Four **domain-5 (radiation therapy & medical physics) Intermediate** projects are complete,
taking the collection to **147 -> 151 / 301 (50.2%) — past the halfway mark** — with domain 5
now at **9/15**. This batch is the **dose-engine + adaptive** cluster: fluence-map plan
optimization, proton/heavy-ion dose, deterministic Boltzmann transport, and the MR-Linac
adaptive workflow. It introduces the collection's **fourth CUDA library — cuSPARSE** (5.2
SpMV). The intended 5th project (`5.5` deformable dose accumulation) was **killed mid-run by
the account's weekly usage limit** and is deferred (see §7); its partial work was discarded so
the tree stays clean. The 4 successful projects were lead-verified normally.

## 2. What changed

Four new projects under `projects/05-radiation-therapy-medphys/`:

- [`5.02` Radiotherapy Treatment-Plan Optimization](../projects/05-radiation-therapy-medphys/5.02-radiotherapy-treatment-plan-optimization)
- [`5.03` Proton & Heavy-Ion Therapy Dose](../projects/05-radiation-therapy-medphys/5.03-proton-heavy-ion-therapy-dose)
- [`5.06` GPU Boltzmann Transport (Deterministic Dose)](../projects/05-radiation-therapy-medphys/5.06-gpu-boltzmann-transport-deterministic-dose)
- [`5.14` GPU-Accelerated Adaptive MR-Linac Workflow](../projects/05-radiation-therapy-medphys/5.14-gpu-accelerated-adaptive-mr-linac-workflow)

Plus a `docs/BUILD_GUIDE.md` §7d note (linking cuSPARSE + its deprecation macro). `docs/STATUS.md`
-> 4 marked **done** (151/301; domain 5 = 9/15). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **5.02 Treatment-Plan Optimization** — a projected-gradient **fluence-map optimizer** whose
  two dominant sparse matrix-vector products (`D x`, `D^T r`) run on a GPU-resident **CSR**
  matrix via **cuSPARSE `cusparseSpMV`**, with a shared `fmo.h` core for byte-identical CPU/GPU
  per-voxel math. PTV mean 59.2 Gy vs 60 Gy Rx. **First cuSPARSE project.** (cusparse.h C4996
  deprecations suppressed via `DISABLE_CUSPARSE_DEPRECATED` so the zero-warning gate holds.)
- **5.03 Proton / Heavy-Ion Dose** — an analytic **pencil-beam** dose engine: shared
  Bragg-peak/lateral-Gaussian physics, one thread per voxel gathering over a constant-memory
  spot list (no atomics). CPU==GPU ~1.8e-7; Bragg peak at 11.75 cm recovers the 12 cm range.
  Proton therapy's signature depth-dose curve.
- **5.06 Deterministic Boltzmann Transport** — a 1-D **discrete-ordinates (S_N)** solver:
  Gauss-Legendre quadrature, source iteration, diamond-difference sweep (one thread per
  ordinate + a fixed-order no-atomics reduction). CPU==GPU ~5e-17. The deterministic alternative
  to Monte-Carlo dose (flagship 5.01) — Acuros-style.
- **5.14 MR-Linac Adaptive Workflow** — a reduced online-adaptive-RT loop: 2-D GPU **Demons**
  deformable registration + dose warp + GTV plan metrics (shared `mrl_registration.h` ->
  byte-identical CPU/GPU ~1e-14). Ties registration (cf. 4.08) to adaptive replanning.

All four are clearly-labeled **reduced-scope teaching versions** (synthetic phantoms/plans,
labeled synthetic), with production tools (Eclipse/RayStation FMO, RayStation/TOPAS proton,
Acuros XB, Elekta Unity/ViewRay oART) named in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/05-radiation-therapy-medphys/5.02-radiotherapy-treatment-plan-optimization  # (or any)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

5.02 links **cuSPARSE** (`cusparse.lib` + `DISABLE_CUSPARSE_DEPRECATED`; see BUILD_GUIDE §7d).
The others are pure custom kernels (pencil-beam gather, S_N sweep, Demons).

## 5. What to study here

Reading path: **5.03** (analytic pencil-beam gather) -> **5.06** (S_N transport sweep) ->
**5.14** (Demons + dose warp) -> **5.02** (cuSPARSE SpMV in an optimization loop). Read the
three dose engines together — **5.01** (Monte Carlo, stochastic), **5.04** (collapsed-cone,
convolution), **5.06** (S_N, deterministic transport) — three philosophies for the same physics.
Exercise: in **5.03**, change the beam energy and watch the Bragg peak shift; in **5.06**, raise
the S_N order and watch the flux converge.

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 4 project folders + `docs/BUILD_GUIDE.md` changed; no artifacts.
- ✅ **Clean rebuild** (`/t:Rebuild`, fat arch list) of all 4 in both `Release|x64` and
  `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (8/8 builds), incl. the cuSPARSE link in 5.02.
- ✅ All 4 **demos PASS**: GPU==CPU (Boltzmann 1e-11; proton 1e-4; MR-Linac 1e-6; FMO 1e-2 Gy).
- ✅ `verify_project.py` -> **DONE** for all 4 (comment ratios **0.81–1.03**).
- **Workflow:** 5 agents launched, 4 completed + 1 (5.5) died on the weekly limit; the 4 good
  ones verified. ~0.92M agent tokens, 396 tool uses.
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs — and a PAUSE

- **Account weekly usage limit reached.** Unlike the ~5-hour session windows encountered
  throughout Phase 2 (each recovered by relaunching after the window reset), this is the
  **weekly** cap. Worker-agent batches cannot be spawned until it resets. **`5.5` (deformable
  dose accumulation) and the 5 domain-5 Advanced projects (`5.10`–`5.13`, `5.15`) are deferred**
  until then. Lead verification/integration (local MSBuild + python, no agents) is unaffected,
  which is how this batch's 4 survivors were still verified and pushed.
- All four are **reduced-scope teaching versions** (synthetic data, labeled synthetic).

## 8. Next push preview

**On hold pending the weekly-limit reset.** When quota returns: finish **domain 5** with `5.5` +
the Advanced tier (`5.10` secondary-cancer-risk MC, `5.11` microdosimetry, `5.12` FLASH-RT,
`5.13` BNCT, `5.15` proton CT) to reach **15/15**, then continue to **domain 6 (physiology /
systems biology)**. Same workflow, lead-verified, one push-note per batch. Collection is at the
**halfway mark (151/301, four of 14 domains complete)**.
