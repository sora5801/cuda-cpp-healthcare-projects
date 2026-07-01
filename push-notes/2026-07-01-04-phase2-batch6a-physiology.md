# Push 2026-07-01 #04 -- phase2 batch6a physiology

> Push-note (CLAUDE.md section 7.1). First Phase-2 batch in **Domain 6 (Computational
> Physiology & Systems Biology)**: 6 projects (the 5 Beginners + the first Intermediate),
> each worker-built and lead-verified.

## 1. Summary

The build-out crosses into its **sixth domain**. Six **domain-6 (physiology & systems
biology)** projects are complete, taking the collection to **157 -> 163 / 301 (54.2%)** and
domain 6 to **7/27** (the flagship `6.04` Lattice-Boltzmann solver was already done). This
batch spans the domain's core computational styles: exact stochastic simulation (Gillespie),
constraint-based optimization (FBA), ensemble ODEs (PK/PD), a library-accelerated forward
problem (ECG via cuBLAS), and two reaction-diffusion stencils (Turing patterns, cardiac
electrophysiology). Each was built in its own folder by one worker and re-verified by the lead.

## 2. What changed

Six new projects under `projects/06-physiology-systems-biology/`:

- [`6.01` Cardiac Electrophysiology Simulation](../projects/06-physiology-systems-biology/6.01-cardiac-electrophysiology-simulation)
- [`6.11` Stochastic (Gillespie) Biochemical Simulation](../projects/06-physiology-systems-biology/6.11-stochastic-gillespie-biochemical-simulation)
- [`6.12` Metabolic Flux / Constraint-Based Modeling](../projects/06-physiology-systems-biology/6.12-metabolic-flux-constraint-based-modeling)
- [`6.15` PK/PD & PBPK Modeling](../projects/06-physiology-systems-biology/6.15-pk-pd-pbpk-modeling)
- [`6.18` ECG Forward Problem & Body-Surface Potentials](../projects/06-physiology-systems-biology/6.18-ecg-forward-problem-body-surface-potential-mapping)
- [`6.24` Reaction-Diffusion Morphogenesis (Turing Patterns)](../projects/06-physiology-systems-biology/6.24-reaction-diffusion-morphogenesis-turing-patterns)

`docs/STATUS.md` -> these 6 marked **done** (163/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **6.01 Cardiac Electrophysiology** — a 2-D **monodomain** reaction-diffusion solver
  (FitzHugh-Nagumo cell ODE + explicit 5-point Laplacian, operator-split), stencil + ping-pong
  (shared `cardiac_cell.h` -> CPU==GPU 1.11e-16). Shows a propagating action-potential wavefront.
  The entry point to cardiac modeling.
- **6.11 Gillespie SSA** — the exact **Stochastic Simulation Algorithm** (direct method) as an
  ensemble: one thread per trajectory, shared SplitMix64 RNG + mass-action core. Exact CPU==GPU
  on integer outputs; recovers the analytic Poisson stationary mean (k_prod/k_deg=20). Exact
  stochastic chemical kinetics on the GPU.
- **6.12 Metabolic Flux (FBA)** — a **Flux Balance Analysis** gene-essentiality screen: a shared
  bounded-variable **simplex** LP solver (Bland's rule), one LP per thread over all
  single-reaction knockouts. Bit-for-bit CPU==GPU. Constraint-based metabolic modeling — an LP
  per GPU thread.
- **6.15 PK/PD & PBPK** — a coupled 1-compartment oral PK + **indirect-response PD** turnover
  model, RK4-integrated as an ensemble over 4096 virtual patients (shared `pkpd.h` -> 3.55e-14).
  Mean AUC ~ dose/CL. The ensemble-ODE pattern (cf. flagship 13.02), for pharmacology.
- **6.18 ECG Forward Problem** — build the **lead-field / transfer matrix** A (one thread per
  entry, shared dipole Green's function) then **cuBLAS DGEMM** applies Phi = A*X over all time
  frames (CPU==GPU ~1.8e-15). Body-surface potential mapping. Third cuBLAS project.
- **6.24 Turing Patterns** — a 2-D **Gierer-Meinhardt** activator-inhibitor reaction-diffusion
  stencil (ping-pong, shared `turing.h` -> 4e-12), plus an analytic **dispersion-relation** check
  that independently predicts the pattern-forming regime and wavelength. Morphogenesis.

All six are clearly-labeled **reduced-scope teaching versions** (synthetic networks/phantoms,
labeled synthetic), with production tools (openCARP/Chaste, StochKit/Cain, COBRApy, Simcyp/
mrgsolve, SCIRun/ECGSIM, morphogenesis codes) named in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/06-physiology-systems-biology/6.01-cardiac-electrophysiology-simulation   # (or any)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

6.18 links **cuBLAS**. The others are pure custom kernels (SSA, simplex LP, RK4 ensemble,
reaction-diffusion stencils).

## 5. What to study here

Reading path: **6.11** (exact SSA) -> **6.15** (RK4 ensemble ODE) -> **6.24** / **6.01**
(reaction-diffusion stencils — morphogenesis and cardiac, same pattern) -> **6.12** (an LP per
thread) -> **6.18** (matrix build + cuBLAS DGEMM). Read **6.01** and **6.24** alongside the
flagships 14.02 (Gray-Scott RD) and 6.04 (Lattice-Boltzmann) — the stencil family recurs across
physiology. Exercise: in **6.24**, change the diffusion ratio and watch spots-vs-stripes; in
**6.01**, raise the diffusion coefficient and watch conduction velocity increase.

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 6 project folders changed; no shared/root file; no artifacts.
- ✅ **Clean rebuild** (`/t:Rebuild`, fat arch list) of all 6 in both `Release|x64` and
  `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (12/12 builds), incl. the cuBLAS link in 6.18.
- ✅ All 6 **demos PASS**: GPU==CPU (Gillespie exact; ECG 1.8e-15; cardiac 1e-16; FBA 1e-9;
  PK/PD & Turing 1e-6).
- ✅ `verify_project.py` -> **DONE** for all 6 (comment ratios **0.86–1.17**).
- **Workflow:** 6 agents, ~1.04M agent tokens, 466 tool uses.
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All six are **reduced-scope teaching versions**: 2-D/toy models, FitzHugh-Nagumo instead of a
  detailed ionic model, a small metabolic network. Labeled synthetic; production scale described
  in each THEORY.md.

## 8. Next push preview

Continue domain-6 Intermediates — cardiac (`6.2` whole-heart, `6.16` electromechanics, `6.17`
Purkinje, `6.19` defibrillation), circulation (`6.3` hemodynamics, `6.20`, `6.21`), neuro
(`6.6`, `6.7`), and systems models (`6.8`–`6.10`, `6.13`, `6.14`, `6.22`, `6.23`, `6.25`–`6.27`)
in ~6-project batches to complete **domain 6 (27/27)**. Same workflow, lead-verified.
