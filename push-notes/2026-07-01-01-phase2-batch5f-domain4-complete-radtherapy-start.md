# Push 2026-07-01 #01 -- phase2 batch5f domain4-complete + radiotherapy start

> Push-note (CLAUDE.md section 7.1). A **cross-domain** batch: the last 2 medical-imaging
> projects (completing **Domain 4, 33/33**) plus the first 4 radiotherapy projects (starting
> **Domain 5**). Each worker-built; independently lead-verified.

## 1. Summary

This batch **completes Domain 4 — Medical Imaging & Image Reconstruction (33/33)** — the
**fourth of 14 domains** finished — and opens **Domain 5 (Radiation Therapy & Medical
Physics)** with its 4 Beginner projects, taking the collection to **141 -> 147 / 301 (48.8%)**.
To use a usage-window efficiently, the two remaining domain-4 items (`4.32`, `4.33`) were
bundled with the four domain-5 Beginners (`5.4`, `5.7`, `5.8`, `5.9`) into one 6-project batch.
The lead caught a warning the worker's self-report missed (see §6).

## 2. What changed

Two new projects under `projects/04-medical-imaging/` (completing domain 4):

- [`4.32` GPU-Accelerated Landmark Detection](../projects/04-medical-imaging/4.32-gpu-accelerated-landmark-detection)
- [`4.33` Real-Time MRI Reconstruction](../projects/04-medical-imaging/4.33-real-time-mri-reconstruction)

Four new projects under `projects/05-radiation-therapy-medphys/` (starting domain 5):

- [`5.04` Collapsed-Cone / Superposition-Convolution Dose](../projects/05-radiation-therapy-medphys/5.04-collapsed-cone-superposition-convolution-dose)
- [`5.07` Brachytherapy Dose & Source Modeling](../projects/05-radiation-therapy-medphys/5.07-brachytherapy-dose-source-modeling)
- [`5.08` Linac QA & Machine Performance Assessment](../projects/05-radiation-therapy-medphys/5.08-linac-qa-machine-performance-assessment)
- [`5.09` Gamma-Index Dose Comparison](../projects/05-radiation-therapy-medphys/5.09-gamma-index-dose-comparison)

`docs/STATUS.md` -> these 6 marked **done** (147/301; **domain 4 = 33/33**, domain 5 = 5/15).
`CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **4.32 Landmark Detection** — heatmap-regression **decode**: one block per landmark does a
  shared-memory argmax tree-reduction (coarse peak) then a deterministic fixed-point-atomic
  **soft-argmax** (sub-voxel refinement), shared HD math -> exact CPU==GPU, ~0.2-voxel recovery.
- **4.33 Real-Time MRI** — golden-angle **radial gridding NUFFT** (density comp -> Kaiser-Bessel
  scatter with fixed-point integer atomics -> **cuFFT** inverse FFT -> deapodize) over a sliding
  window of frames. CPU==GPU 1.65e-11; last-frame correlation 0.96 with the moving phantom.
- **5.04 Collapsed-Cone Dose** — a 2-D superposition-convolution photon dose engine: Stage-1
  **TERMA** Siddon ray-trace (thread per beam column) + Stage-2 collapsed-cone superposition
  (thread per source-voxel, deterministic integer-atomic scatter). Exact CPU==GPU. The clinical
  dose-calc workhorse alongside Monte Carlo (flagship 5.01).
- **5.07 Brachytherapy (TG-43)** — a full **TG-43** dose calculator: per-voxel threads, inner
  loop over dwell positions, source g_L(r)/F(r,theta) anisotropy tables in constant memory
  (shared `tg43_physics.h` -> max_rel_err=0). The AAPM brachytherapy dose formalism.
- **5.08 Linac QA** — a 2-D **gamma-index** QA workflow (one thread per measured pixel, shared
  gamma core -> exact CPU==GPU) plus flatness/symmetry/output metrics; recovers an injected
  1%-low / 2%-asymmetry beam error.
- **5.09 Gamma-Index Comparison** — the IMRT/VMAT **2-D gamma index**: one thread per reference
  voxel does a distance-limited gather + per-thread exact-float-min (shared `gamma_core.h` ->
  bit-identical). A synthetic dose pair with a hot spot yields a 99.7% pass-rate.

All six are clearly-labeled **reduced-scope teaching versions** (synthetic phantoms/dose planes,
labeled synthetic), with production tools (nnDetection, BART/gpuNUFFT, Pinnacle/RayStation
collapsed-cone, Oncentra TG-43, DoseLab/SNC gamma) named in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/05-radiation-therapy-medphys/5.04-collapsed-cone-superposition-convolution-dose  # (or any)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

4.33 links **cuFFT**. The others are pure custom kernels (soft-argmax, Siddon ray-trace,
collapsed-cone scatter, TG-43 tables, gamma gather).

## 5. What to study here

Reading path: **5.09** / **5.08** (gamma gather + min — the QA core) -> **5.07** (TG-43 table
lookup) -> **5.04** (TERMA + collapsed-cone superposition) -> **4.32** (soft-argmax decode) ->
**4.33** (radial NUFFT + cuFFT). Read the two dose engines (5.04 collapsed-cone, 5.01 Monte
Carlo flagship) side by side — deterministic convolution vs. stochastic transport. Exercise: in
**5.09**, tighten the gamma criteria (2%/2mm -> 1%/1mm) and watch the pass-rate drop.

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 6 project folders (2 in `projects/04`, 4 in `projects/05`) changed;
  no shared/root file committed.
- ⚠️ **Lead caught two things the workers missed:**
  1. A stray `pocket_final.json` (a worker's synthetic-data scratch written to the repo root) —
     never committed (project `git add` is path-scoped), deleted during integration.
  2. **`4.33` emitted 6 warnings** (`#177-D: function "rms" declared but never referenced`) on a
     clean rebuild, despite the worker self-reporting "0 warnings". The lead **deleted the dead
     `rms()` function** (CLAUDE.md §6.1.8 no-dead-code + §9 zero-warnings), re-verified 4.33 to
     0 warnings / demo PASS / DONE, and only then integrated. A concrete example of why the lead
     independently rebuilds instead of trusting worker reports.
- ✅ **Clean rebuild** (`/t:Rebuild`, fat arch list) of all 6 in both `Release|x64` and
  `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (12/12 builds, post-fix), incl. cuFFT in 4.33.
- ✅ All 6 **demos PASS**: GPU==CPU (landmark/collapsed-cone/QA/gamma exact; brachytherapy 1e-5;
  real-time MRI 1.65e-11).
- ✅ `verify_project.py` -> **DONE** for all 6 (comment ratios **0.89–1.06**).
- ✅ **Domain-4 sweep:** 33/33 markers `done`.
- **Workflow:** 6 agents, ~1.14M agent tokens, 481 tool uses.
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All six are **reduced-scope teaching versions**: 2-D dose engines, small phantoms, synthetic
  QA planes. Labeled synthetic; production scale described in each THEORY.md.

## 8. Next push preview

Continue **domain 5** — Intermediates (`5.2` plan optimization, `5.3` proton/heavy-ion dose,
`5.5` deformable dose accumulation, `5.6` deterministic Boltzmann transport, `5.14` MR-Linac)
then Advanced (`5.10`–`5.13`, `5.15`) — to complete **domain 5 (15/15)**. Then domain 6
(physiology). Same workflow, lead-verified.
