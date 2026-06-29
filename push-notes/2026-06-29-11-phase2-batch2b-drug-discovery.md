# Push 2026-06-29 #11 -- phase2 batch2b drug-discovery

> Push-note (CLAUDE.md section 7.1). Third Phase-2 batch: the last 5 domain-1 Beginner
> projects, each built by a worker agent and independently re-verified by the lead.

## 1. Summary

Five more **domain-1 (drug discovery)** projects are complete, taking the collection to
**26 -> 31 / 301 (10.3%)** — past the 10% mark. This batch finishes the **Beginner tier
of domain 1**: free-energy / enhanced-sampling methods (metadynamics, umbrella
sampling/WHAM, MM-GBSA rescoring) plus two electronic-structure / ML methods
(semi-empirical tight-binding, neural-network potentials). The theme of the batch is the
**ensemble pattern** — "one independent simulation per GPU thread" — appearing in four of
the five, and a second **batched cuSOLVER** library call. Every project was built in its
own folder by one worker, then the lead independently re-verified it (boundary check,
clean Release+Debug rebuild with zero warnings, demo PASS, `verify_project.py` DONE, and
a code spot-read of the batched-eigensolver project).

## 2. What changed

Five new fully-implemented projects under `projects/01-drug-discovery/`:

- [`1.06` Enhanced Sampling — Metadynamics & Replica Exchange](../projects/01-drug-discovery/1.06-enhanced-sampling-metadynamics-replica-exchange)
- [`1.08` Semi-Empirical & Tight-Binding Quantum Methods](../projects/01-drug-discovery/1.08-semi-empirical-tight-binding-quantum-methods)
- [`1.09` ML Interatomic Potentials (Neural-Network Potentials)](../projects/01-drug-discovery/1.09-ml-interatomic-potentials-neural-network-potentials)
- [`1.24` Umbrella Sampling / WHAM Free-Energy Profiles](../projects/01-drug-discovery/1.24-umbrella-sampling-wham-free-energy-profiles)
- [`1.27` MM-GBSA / MM-PBSA Rescoring](../projects/01-drug-discovery/1.27-mm-gbsa-mm-pbsa-rescoring)

`docs/STATUS.md` -> these 5 marked **done** (31/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **1.06 Metadynamics & Replica Exchange** — well-tempered, multi-walker metadynamics on
  a synthetic 1-D double well: one Langevin walker per GPU thread (the ensemble pattern),
  depositing tempered Gaussian hills onto a shared bias grid until the free-energy surface
  fills in. The teaching gem is in `THEORY.md`: because Langevin trajectories are
  **chaotic**, CPU and GPU cannot agree bit-for-bit per walker — so verification is on the
  *robust ensemble FES* (~0.17 kT) and the recovered 5 kT barrier, and `--fmad=false` is
  set (in both `.vcxproj` and CMake) to tame FMA-contraction divergence.
- **1.08 Semi-Empirical / Tight-Binding** — a *batched* Hückel/tight-binding solver: a
  custom kernel builds every molecule's padded Hamiltonian in parallel (one thread per
  matrix element, race-free), then **`cusolverDnDsyevjBatched`** diagonalizes the whole
  batch in a single library call. Verified against a CPU Jacobi reference (worst eigen
  diff ~3e-15) and closed-form Hückel energies (benzene 8|beta|, cyclobutadiene gap 0).
  The "no black box" comment on the batched eigensolver is the thing to read.
- **1.09 Neural-Network Potentials** — a reduced Behler-Parrinello NNP: per-atom radial
  ACSF descriptors feed a small tanh MLP (8->16->16->1) whose outputs sum to the total
  energy. One thread per atom, model weights in constant memory, shared
  `__host__ __device__` core (`nnp.h`) -> CPU==GPU to ~1e-15. A clean view of "ML force
  field inference" as an embarrassingly-parallel per-atom kernel.
- **1.24 Umbrella Sampling / WHAM** — one biased overdamped-Langevin trajectory per window
  on the GPU (ensemble pattern, shared core in `umbrella.h`); the CPU then runs WHAM to
  stitch the biased histograms into one PMF. GPU and CPU histograms match *exactly*
  (integer counts), and WHAM recovers the 4 kT barrier within 0.30 kT.
- **1.27 MM-GBSA / MM-PBSA Rescoring** — one MD snapshot per thread, each evaluating a
  binding free energy (LJ van der Waals + Coulomb + Generalized-Born solvation + a constant
  -T*dS entropy term) via a shared `snapshot_dg()`. Exact CPU/GPU agreement
  (max abs err 0.0). A compact intro to implicit-solvent end-point free-energy methods.

All five are clearly-labeled **reduced-scope teaching versions** (synthetic potentials and
toy molecules; real metadynamics / NNPs / MM-PBSA are research-grade) with the full method
described in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/01-drug-discovery/1.08-semi-empirical-tight-binding-quantum-methods   # (or any)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

1.08 links cuSOLVER (`.lib` in both `<Link>` sections + `CMakeLists.txt`). 1.06 sets the
nvcc flag `--fmad=false` in the `.vcxproj`/CMake for FP reproducibility (see its THEORY.md).

## 5. What to study here

Reading path: **1.27** (simplest ensemble: one snapshot per thread, exact agreement) ->
**1.09** (per-atom ML inference) -> **1.24** (biased sampling + a CPU post-processing step,
WHAM) -> **1.06** (the same ensemble idea but *chaotic*, and what that means for
verification) -> **1.08** (two GPU jobs: a hand-written Hamiltonian-builder kernel and a
batched cuSOLVER eigensolve). Exercise: in **1.06**, flip `--fmad=false` back to the default
and watch how much the per-walker paths diverge while the ensemble FES stays stable — a
direct, runnable demonstration of the floating-point determinism lesson in `PATTERNS.md §4`.

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 5 project folders changed; no shared/root file; no artifacts.
- ✅ **Clean rebuild** (`/t:Rebuild`, committed fat arch list) of all 5 in both
  `Release|x64` and `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (10/10 builds).
- ✅ All 5 **demos PASS**: rebuilt exe reproduces `expected_output.txt`, GPU==CPU
  (tight-binding 3e-15; NNP/umbrella/MM-GBSA exact-or-1e-9; metadynamics ensemble FES
  within 0.25 kT as documented).
- ✅ `verify_project.py` -> **DONE** for all 5 (comment ratios **0.90–1.13**).
- ✅ **Spot-read** of 1.08: `build_hamiltonians_kernel` launch config + the
  `cusolverDnDsyevjBatched` call documented to the "no black box" standard.
- **Workflow:** 5 agents, ~0.92M agent tokens, 436 tool uses (one window, no limit hit).
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All five are **reduced-scope teaching versions** with synthetic potentials/landscapes;
  labeled synthetic everywhere. Real enhanced-sampling and end-point free-energy methods
  need real force fields and far longer trajectories.
- 1.06's per-walker GPU/CPU paths are intentionally *not* bit-reproducible (chaos); only the
  ensemble FES is verified — documented honestly in its THEORY.md and demo output.

## 8. Next push preview

Domain 1 (drug discovery) **Beginner tier is now complete** (18/35 done). Next: the
**domain-1 Intermediate** projects (e.g. `1.10`, `1.15`, `1.17`–`1.23`, `1.25`, `1.26`,
`1.28`, `1.29`) in ~6-project batches, then the few Advanced ones, then move to domain 2
(structural biology). Same workflow, lead-verified, one push-note per batch.
