# Push 2026-07-01 #05 -- phase2 batch6b physiology intermediate

> Push-note (CLAUDE.md section 7.1). Second domain-6 batch: 6 Intermediate physiology projects,
> each worker-built and independently lead-verified.

## 1. Summary

Six more **domain-6 (physiology & systems biology) Intermediate** projects are complete,
taking the collection to **163 -> 169 / 301 (56.1%)** and domain 6 to **13/27**. This batch
spans the organ systems: a whole-heart digital twin, blood-flow CFD, lung particle deposition,
two neuron simulators (biophysical Hodgkin-Huxley and point-neuron LIF), and tumor growth. The
recurring GPU shapes are the **ensemble-ODE** (0-D twin), the **stencil + ping-pong** (CFD,
tumor), and **one-thread-per-agent** (particles, neurons). Each was built in its own folder by
one worker and re-verified by the lead.

## 2. What changed

Six new projects under `projects/06-physiology-systems-biology/`:

- [`6.02` Whole-Heart Digital Twin](../projects/06-physiology-systems-biology/6.02-whole-heart-digital-twin)
- [`6.03` Hemodynamics / Blood-Flow CFD](../projects/06-physiology-systems-biology/6.03-hemodynamics-blood-flow-cfd)
- [`6.05` Respiratory / Lung Airflow & Particle Deposition](../projects/06-physiology-systems-biology/6.05-respiratory-lung-airflow-particle-deposition)
- [`6.06` Neuronal Network Simulation (Biophysical)](../projects/06-physiology-systems-biology/6.06-neuronal-network-simulation-biophysical)
- [`6.07` Spiking Neural Network (Point-Neuron)](../projects/06-physiology-systems-biology/6.07-spiking-neural-network-point-neuron-simulation)
- [`6.08` Tumor Growth & Treatment-Response Modeling](../projects/06-physiology-systems-biology/6.08-tumor-growth-treatment-response-modeling)

`docs/STATUS.md` -> these 6 marked **done** (169/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **6.02 Whole-Heart Digital Twin** — a 0-D twin coupling FitzHugh-Nagumo EP + **time-varying
  elastance** mechanics + a **3-element Windkessel** afterload, run as a GPU ensemble (one thread
  per virtual heart, RK4) with a contractility-sweep **twin-fit** to a target stroke volume.
  CPU==GPU 5.7e-14. Personalized cardiac modeling as an ensemble ODE.
- **6.03 Hemodynamics CFD** — a 2-D incompressible **Navier-Stokes** channel solver via
  **Chorin's fractional step** (predictor + Jacobi pressure Poisson + corrector) with
  **Carreau-Yasuda** non-Newtonian blood viscosity + wall-shear-stress output. Stencil +
  ping-pong; velocity within ~4.8% of analytic Poiseuille. A CFD complement to the LBM flagship 6.04.
- **6.05 Lung Particle Deposition** — a **Lagrangian aerosol** simulator: one thread per particle
  (grid-stride, per-thread RNG), integer-atomic deposition counters -> exact CPU==GPU tallies.
  Inhaled-drug/particle deposition modeling.
- **6.06 Biophysical Neurons** — a ring of **multi-compartment Hodgkin-Huxley** neurons with
  **Rush-Larsen** gate integration, a hand-rolled **Hines/Thomas tridiagonal** cable solver, and
  event-driven synapses. One thread per neuron; exact spike-count match; a travelling spike wave.
- **6.07 Spiking Neural Network** — a **Brunel balanced LIF** network: exponential-Euler state
  update + integer-atomic synaptic scatter (shared `lif.h` -> exact spike counts CPU==GPU). The
  point-neuron counterpart to 6.06.
- **6.08 Tumor Growth** — a **Fisher-KPP** reaction-diffusion tumor model (2-D stencil,
  ping-pong) with **linear-quadratic** radiotherapy cell-kill; treated vs. control scenarios show
  ~28% burden reduction. CPU==GPU 2.2e-16.

All six are clearly-labeled **reduced-scope teaching versions** (0-D/2-D, small networks),
labeled synthetic, with production tools (openCARP/CircAdapt, SimVascular/OpenFOAM, CFPD lung
codes, NEURON, NEST/Brian2, tumor-growth PDE frameworks) named in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/06-physiology-systems-biology/6.06-neuronal-network-simulation-biophysical   # (or any)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

No new CUDA libraries this batch — all custom kernels (ensemble RK4, Chorin stencil, Lagrangian
particles, HH cable solver, LIF network, Fisher-KPP stencil).

## 5. What to study here

Reading path: **6.05** (one thread per particle) -> **6.07** (LIF network) -> **6.06** (HH +
tridiagonal cable — the deep one) -> **6.08** / **6.03** (reaction-diffusion & CFD stencils) ->
**6.02** (0-D ensemble twin + parameter fit). Read **6.06** and **6.07** together — the same
network, at two levels of biophysical detail. Exercise: in **6.03**, raise the Reynolds number
and watch the profile change; in **6.02**, sweep a different parameter and re-fit the twin.

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 6 project folders changed; no shared/root file; no build-output paths.
- ✅ **Clean rebuild** (`/t:Rebuild`, fat arch list) of all 6 in both `Release|x64` and
  `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (12/12 builds).
- ✅ All 6 **demos PASS**: GPU==CPU (particle/neuron spike counts exact; tumor 2.2e-16; CFD/twin 1e-9).
- ✅ `verify_project.py` -> **DONE** for all 6 (comment ratios **0.79–1.01**).
- **Workflow:** 6 agents, ~1.01M agent tokens, 451 tool uses.
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All six are **reduced-scope teaching versions**: 0-D/2-D models, small neuron rings/networks,
  a coarse CFD grid. Labeled synthetic; production scale described in each THEORY.md.

## 8. Next push preview

Continue domain-6 Intermediates — systems models (`6.9` agent-based tissue, `6.10` ODE/SDE
networks, `6.13` GRN inference, `6.14` multi-scale), more cardiac (`6.16` electromechanics,
`6.17` Purkinje, `6.19` defibrillation), circulation (`6.20`, `6.21`), and organ models
(`6.22`, `6.23`, `6.25`–`6.27`) in ~6-project batches to complete **domain 6 (27/27)**. Same
workflow, lead-verified.
