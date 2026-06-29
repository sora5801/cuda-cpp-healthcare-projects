# Push 2026-06-29 #13 -- phase2 batch2d drug-discovery advanced-md

> Push-note (CLAUDE.md section 7.1). Fifth Phase-2 batch: 6 more domain-1 Intermediate
> projects — the advanced-MD / enhanced-sampling cluster — each lead-verified.

## 1. Summary

Six **domain-1 (drug discovery)** Intermediate projects are complete, taking the
collection to **37 -> 43 / 301 (14.3%)**. This batch is the *advanced molecular-dynamics*
cluster of domain 1: a polarizable (AMOEBA) induced-dipole solver, constant-pH MD,
QM/MM MD, Gaussian-Accelerated MD, Steered MD with Jarzynski reweighting, and covalent
docking. The recurring lesson across the batch is **non-equilibrium / enhanced-sampling
free-energy estimation done as a GPU ensemble** (one independent trajectory or solve per
thread) — and the floating-point discipline needed to keep that ensemble reproducible.
Every project was built in its own folder by one worker, then the lead independently
re-verified it (boundary check, clean Release+Debug rebuild with zero warnings, demo PASS,
`verify_project.py` DONE).

## 2. What changed

Six new fully-implemented projects under `projects/01-drug-discovery/`:

- [`1.21` Polarizable / AMOEBA Force Field MD](../projects/01-drug-discovery/1.21-polarizable-amoeba-force-field-md)
- [`1.22` Constant-pH Molecular Dynamics](../projects/01-drug-discovery/1.22-constant-ph-molecular-dynamics)
- [`1.23` QM/MM Molecular Dynamics](../projects/01-drug-discovery/1.23-qm-mm-molecular-dynamics)
- [`1.25` Gaussian-Accelerated MD (GaMD)](../projects/01-drug-discovery/1.25-gaussian-accelerated-md-gamd)
- [`1.26` Steered Molecular Dynamics (SMD)](../projects/01-drug-discovery/1.26-steered-molecular-dynamics-smd)
- [`1.28` Covalent Docking](../projects/01-drug-discovery/1.28-covalent-docking)

`docs/STATUS.md` -> these 6 marked **done** (43/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **1.21 Polarizable / AMOEBA MD** — the induced-dipole self-consistent field of a
  polarizable force field is a linear solve `A mu = b`; this implements a **matrix-free
  conjugate-gradient** solver, run as an ensemble (one thread per system). CPU==GPU to
  ~5.6e-17. The thing to study is *matrix-free* CG — the operator `A*mu` is applied without
  ever forming `A`, the key to scaling polarization to large systems.
- **1.22 Constant-pH MD** — ensemble Metropolis Monte-Carlo titration: one thread per
  (pH, replica) chain flips protonation states against a shared `__host__ __device__`
  energy/RNG core, tallying with **integer atomicAdd** so the GPU histogram matches the CPU
  exactly (0/45 mismatches). Produces titration curves and a coupling-shifted pKa.
- **1.23 QM/MM MD** — proton transfer on a 2-state quantum surface with electrostatic
  embedding: an analytic 2x2 ground-state eigen-solve gives the QM energy/force each step,
  velocity-Verlet propagates, all as a GPU ensemble (one thread per trajectory). CPU==GPU
  to ~7e-12. A minimal but honest picture of the QM/MM force loop.
- **1.25 Gaussian-Accelerated MD** — GaMD-boosted Langevin walkers on a double well (one
  thread per walker, counter-based RNG, deterministic **fixed-point integer-atomic**
  histogram), with **2nd-order cumulant reweighting** to recover the true barrier from the
  biased sampling. Exact (tol 0) CPU/GPU parity. Teaches boost-potential reweighting.
- **1.26 Steered MD** — 8192 independent constant-velocity pulls (one thread per
  trajectory, shared SplitMix64 + overdamped Langevin), then **Jarzynski's equality**
  turns the non-equilibrium work distribution into a free energy (estimate -11.32 vs true
  -12.0 kJ/mol). A vivid demonstration of why you need *many* trajectories.
- **1.28 Covalent Docking** — one thread per ligand torsion conformation (46,656 on a
  36^3 grid) scoring a harmonic covalent-bond constraint plus LJ/Coulomb pocket energy
  (shared HD core, CPU==GPU ~2e-15). **Found & fixed:** the synthetic pocket geometry was
  producing r^-12 clash energies that amplified FMA rounding into a large CPU/GPU mismatch;
  the pocket was redesigned and the lesson documented in THEORY.md.

All six are clearly-labeled **reduced-scope teaching versions** (synthetic systems, labeled
synthetic), with the production methods (Tinker/OpenMM-AMOEBA, CpHMD, real QM/MM, AMBER
GaMD, NAMD SMD, covalent-docking engines) described in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/01-drug-discovery/1.26-steered-molecular-dynamics-smd   # (or any of the six)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

No new CUDA libraries this batch — all custom kernels (ensemble-per-thread, constant
memory, integer/fixed-point atomics, matrix-free CG).

## 5. What to study here

Reading path: **1.21** (matrix-free CG — a linear *solve* per thread) -> **1.22** (MC
titration with exact integer tallies) -> **1.23** (a QM force embedded in an MD loop) ->
**1.25** -> **1.26** (the two faces of enhanced sampling: reweight a *biased* simulation,
vs. average *non-equilibrium* work) -> **1.28** (a clean docking search + a real FMA-rounding
war story). Exercise: in **1.26**, cut the number of SMD trajectories from 8192 to 64 and
watch the Jarzynski estimate degrade — the exponential average is dominated by rare low-work
pulls, so it needs a large ensemble. In **1.25**, change the boost level and confirm the
reweighted barrier stays put.

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 6 project folders changed; no shared/root file; no artifacts.
- ✅ **Clean rebuild** (`/t:Rebuild`, committed fat arch list) of all 6 in both
  `Release|x64` and `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (12/12 builds).
- ✅ All 6 **demos PASS**: rebuilt exe reproduces `expected_output.txt`, GPU==CPU
  (constant-pH/GaMD tallies exact; AMOEBA 5.6e-17; QM/MM 7e-12; SMD work 1e-12; covalent 2e-15).
- ✅ `verify_project.py` -> **DONE** for all 6 (comment ratios **0.96–1.15**).
- **Workflow:** 6 agents, ~1.09M agent tokens, 499 tool uses.
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All six are **reduced-scope teaching versions**: a single-system AMOEBA SCF, a toy
  titration model, a 2-state QM surface, 1-D double-well GaMD/SMD, a rigid synthetic
  covalent pocket. Labeled synthetic; real methods need full force fields, real QM, and
  long trajectories.
- 1.26's Jarzynski estimate is intentionally imperfect at the committed ensemble size — a
  teaching point about the variance of exponential averages, documented in its THEORY.md.

## 8. Next push preview

Finish **domain 1**: `1.29` (kinase selectivity panel) is the last Intermediate, plus the
4 **Advanced** projects `1.32`–`1.35`. After that domain 1 (drug discovery) is **complete
(35/35)** and Phase 2 moves to **domain 2 (structural biology)**. Same workflow,
lead-verified, one push-note per batch.
