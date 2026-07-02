# Push 2026-07-01 #07 -- phase2 batch6d physiology intermediate

> Push-note (CLAUDE.md section 7.1). Fourth domain-6 batch: 6 Intermediate physiology projects,
> each worker-built and independently lead-verified.

## 1. Summary

Six more **domain-6 (physiology & systems biology) Intermediate** projects are complete,
taking the collection to **175 -> 181 / 301 (60.1%) — past 60%** — and domain 6 to **25/27**
(only `6.26`, `6.27` remain). This batch is the **organ-perfusion & closed-loop-control**
cluster: defibrillation threshold, coronary autoregulation, tissue oxygen transport, bone
remodeling, an artificial-pancreas trial, and liver/kidney perfusion. It exercises a wide
numerical range — an ensemble cable sweep, a **Conjugate-Gradient sparse solve**, a Green's
-function gather, stencils, and controller-in-the-loop ODEs. Each was built in its own folder by
one worker and re-verified by the lead.

## 2. What changed

Six new projects under `projects/06-physiology-systems-biology/`:

- [`6.19` Defibrillation & High-Voltage Shock Simulation](../projects/06-physiology-systems-biology/6.19-defibrillation-high-voltage-shock-simulation)
- [`6.20` Coronary Autoregulation & Microvascular Perfusion](../projects/06-physiology-systems-biology/6.20-coronary-autoregulation-microvascular-perfusion)
- [`6.21` Microcirculation & Oxygen Transport](../projects/06-physiology-systems-biology/6.21-microcirculation-oxygen-transport)
- [`6.22` Bone Remodeling Simulation](../projects/06-physiology-systems-biology/6.22-bone-remodeling-simulation)
- [`6.23` Glucose-Insulin Dynamics & Artificial Pancreas](../projects/06-physiology-systems-biology/6.23-glucose-insulin-dynamics-artificial-pancreas)
- [`6.25` Liver & Kidney Perfusion Modeling](../projects/06-physiology-systems-biology/6.25-liver-kidney-perfusion-modeling)

`docs/STATUS.md` -> these 6 marked **done** (181/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **6.19 Defibrillation** — a 1-D monodomain FitzHugh-Nagumo cable **DFT sweep**: one thread
  runs a full cable per shock amplitude (ensemble-of-trajectories), shared `defib.h` -> ~1e-17.
  Finds the defibrillation threshold (0.150).
- **6.20 Coronary Autoregulation** — a sparse-SPD network-Poiseuille flow solve via
  **Conjugate Gradient with a hand-rolled CSR-SpMV** (one thread per node), an autoregulation
  outer loop (r^4 conductance + Fahraeus-Lindqvist + radius feedback), and a virtual **FFR**
  read-out. GPU==CPU ~5e-14 mmHg. An iterative sparse solver in a physiology loop.
- **6.21 Microcirculation O₂** — a **Green's-function (Secomb-Hsu)** tissue oxygen solver: one
  thread per grid point gathers each capillary's 1/r field (shared physics: Green's function +
  Hill saturation + Michaelis-Menten consumption). Finds a 3.47% hypoxic pocket. CPU==GPU 1.4e-14.
- **6.22 Bone Remodeling** — a 2-D voxel **mechanostat** (Frost/Huiskes SED dead-band) driven by
  a density-weighted Jacobi-diffusion stimulus proxy (stencil + ping-pong, 1.1e-16). Wolff's law
  as a stencil.
- **6.23 Artificial Pancreas** — a closed-loop in-silico trial: **Bergman** 3-state glucose
  -insulin ODE + gastric-emptying meal + a discrete **PID** controller, RK4, ensemble over virtual
  patients (~1e-13). Controller-in-the-loop simulation.
- **6.25 Liver/Kidney Perfusion** — an ensemble-ODE liver lobule of independent sinusoids, each a
  1-D convection-reaction ODE with zonal **Michaelis-Menten** clearance (RK4, one thread per
  sinusoid). CPU==GPU 2e-16; mean extraction matches the analytic first-order limit.

All six are clearly-labeled **reduced-scope teaching versions** (synthetic vasculatures/cohorts,
labeled synthetic), with production tools (openCARP defib, 1-D vascular network solvers,
Secomb GreensV, bone-adaptation FE codes, UVA/Padova simulator, PBPK perfusion) named in each
`THEORY.md`.

## 4. How to build & run

```powershell
cd projects/06-physiology-systems-biology/6.20-coronary-autoregulation-microvascular-perfusion  # (or any)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

No new CUDA libraries this batch — all custom kernels (ensemble cables, hand-rolled CG/CSR-SpMV,
Green's-function gather, stencils, RK4 ODEs).

## 5. What to study here

Reading path: **6.19** (ensemble cable sweep) -> **6.25** / **6.23** (ensemble ODEs — perfusion
and closed-loop control) -> **6.22** (mechanostat stencil) -> **6.21** (Green's-function gather)
-> **6.20** (the standout: a CG sparse solve with hand-rolled CSR-SpMV — read it against 5.02's
cuSPARSE SpMV to compare hand-rolled vs library). Exercise: in **6.23**, detune the PID gains and
watch glycemic control degrade; in **6.20**, change a stenosis and watch the virtual FFR drop.

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 6 project folders changed; no shared/root file; no artifacts.
- ✅ **Clean rebuild** (`/t:Rebuild`, fat arch list) of all 6 in both `Release|x64` and
  `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (12/12 builds).
- ✅ All 6 **demos PASS**: GPU==CPU (defib 1e-17; O₂/bone/liver 1e-9..1e-16; coronary 5e-14;
  glucose 1e-4).
- ✅ `verify_project.py` -> **DONE** for all 6 (comment ratios **0.74–1.20**).
- **Workflow:** 6 agents, ~1.08M agent tokens, 456 tool uses.
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All six are **reduced-scope teaching versions**: 1-D/2-D or lumped models, small vascular
  networks, synthetic patient cohorts. Labeled synthetic; production scale described in each THEORY.md.

## 8. Next push preview

The **last 2 domain-6 projects** (`6.26` virtual population / sensitivity, `6.27` parameter
estimation & data assimilation) — completing **domain 6 (27/27)** — bundled with the first
domain-7 (medical AI) projects in one cross-domain batch. **Six of 14 domains** nearly done.
Same workflow, lead-verified.
