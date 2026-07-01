# 6.12 — Metabolic Flux / Constraint-Based Modeling

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟢 Beginner · Established** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.12`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

This project predicts how a cell distributes its metabolism using **Flux Balance
Analysis (FBA)** — the workhorse of systems biology. A metabolic network is a set
of biochemical reactions; FBA finds the reaction rates ("fluxes") that maximise
growth subject to mass balance and capacity limits. That is a **linear program
(LP)**. We then run a **gene-essentiality screen**: delete each reaction in turn,
re-solve, and see whether the cell can still grow — the classic way to nominate
drug targets. The screen is many *independent* LPs, so we give **each knockout its
own GPU thread**. Everything runs on a tiny, clearly-labeled **synthetic** toy
network whose answers you can check by hand, and the GPU result is verified
bit-for-bit against a plain CPU solver.

## What this computes & why the GPU helps

Flux balance analysis (FBA) finds optimal metabolic fluxes by solving a linear
program (LP) constrained by stoichiometry (`S v = 0`), thermodynamics, and enzyme
capacity (`lb ≤ v ≤ ub`) on genome-scale metabolic models with thousands of
reactions. A single LP is sequential, but the interesting studies solve **many**
LPs: one per single-gene knockout in an **essentiality screen**, or one per
condition in a drug/nutrient screen.

**The parallel bottleneck:** it is the *number of LP solves*, not one solve, that
dominates. A whole-genome essentiality screen is thousands of independent LPs.
Because deleting reaction *i* is unrelated to deleting reaction *j*, the solves
share nothing — perfect **embarrassingly-parallel** work. This project maps **one
LP solve to one GPU thread** (`screen_kernel` in `src/kernels.cu`); the solver
itself (a bounded-variable simplex) is shared `__host__ __device__` code so the
CPU and GPU compute the same thing.

## The algorithm in brief

- **Flux Balance Analysis (FBA):** `max cᵀv s.t. Sv=0, lb≤v≤ub` — a linear program.
- **Bounded-variable simplex** with **Bland's rule** (deterministic, anti-cycling)
  as the LP solver, written once for CPU and GPU.
- **Single-reaction knockout screen:** clamp each reaction to zero and re-solve;
  classify each as essential / growth-reducing / neutral.
- Related methods (FVA, pFBA, thermodynamic FBA, MILP gap-filling, shadow prices)
  are variations on the same LP core — see THEORY.md §7.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/metabolic-flux-constraint-based-modeling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/metabolic-flux-constraint-based-modeling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\metabolic-flux-constraint-based-modeling.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/toy_core_model.txt`, prints the
knockout screen, shows the GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/toy_core_model.txt` — a tiny **synthetic**
  4-metabolite × 8-reaction FBA model, so the demo runs with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print pointers to real
  genome-scale models (BiGG, Recon3D) and the COBRApy workflow that reads them.
- **Provenance & license:** see [data/README.md](data/README.md).

Real models (from the catalog): Recon3D — human genome-scale metabolic model
(<https://github.com/SBRG/Recon3D>); BiGG Models — curated GEMs
(<http://bigg.ucsd.edu>); HMDB (<https://hmdb.ca>); Reactome (<https://reactome.org>).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): the
wild-type growth (`10`), a per-reaction knockout table classified as
`ESSENTIAL` / `reduced` / `neutral`, a summary (`3 essential, 1 growth-reducing,
4 neutral`), and `RESULT: PASS`. The program computes the screen on both the
**GPU** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and
asserts the objectives agree within `1e-9`. Because both call the identical
simplex in `src/fba.h`, the observed difference is exactly `0` — that agreement
is the correctness guarantee.

## Code tour

Read in this order:

1. [`src/fba.h`](src/fba.h) — the FBA linear program and the shared
   `__host__ __device__` bounded-variable **simplex solver** (the heart of it).
2. [`src/main.cu`](src/main.cu) — loads the model, runs CPU + GPU screens, verifies, reports.
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the model loader and the trusted serial screen.
4. [`src/kernels.cuh`](src/kernels.cuh) — the "one LP per thread" idea.
5. [`src/kernels.cu`](src/kernels.cu) — the kernel and host wrapper.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **COBRApy** (<https://github.com/opencobra/cobrapy>) — the reference Python
  FBA/FVA toolkit with multiple LP/MILP backends. Study `model.optimize()` and
  `single_reaction_deletion` to see the production version of this project.
- **Recon3D** (<https://github.com/SBRG/Recon3D>) — a real human genome-scale
  model to scale up to; shows how huge and sparse real `S` matrices are.
- **Virtual Metabolic Human** (<https://vmh.life>) — interactive Recon3D portal.
- **SUNDIALS** (<https://github.com/LLNL/sundials>) — ODE integrators used for
  *dynamic* FBA, where the LP is re-solved along a time course.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**An ensemble of independent LPs — one LP solve per thread** (the same
embarrassingly-parallel shape as flagship 9.02's one-ODE-per-thread, but the
per-item work is a whole simplex solve). Each thread clamps one reaction, runs the
shared simplex in its own local-memory tableau, and writes one objective — no
shared memory, no atomics, no synchronisation. Block size is a deliberately modest
**64** because each thread holds a large private tableau (an occupancy-vs-footprint
lesson; see THEORY.md §4). The catalog's alternative "one LP per **block** with a
shared-memory constraint matrix and warp-level reductions" is the right layout for
*fewer, larger* LPs — discussed in THEORY §7 and Exercise 5.

## Exercises

1. **Grow the network.** Edit `scripts/make_synthetic.py` to add a reaction or
   metabolite (stay within `FBA_MAX_MET/RXN` in `fba.h`), regenerate the sample,
   and predict the new essentiality classes before you run it.
2. **Synthetic lethality.** The isozyme pair `A->B_1` / `A->B_2iso` is individually
   neutral. Extend `main.cu` to also screen all **double** knockouts and find the
   pair whose *joint* deletion is lethal — the basis of combination therapy.
3. **Flux Variability Analysis (FVA).** For a fixed optimal growth, an individual
   flux may still range over an interval. Add two extra LP solves per reaction
   (min and max of `v_j` subject to `cᵀv = optimum`) to report that interval.
4. **Condition screen.** Instead of knockouts, sweep the uptake bound `ub[0]` over
   many values (one LP per condition) and plot the growth curve — the same kernel,
   a different per-thread perturbation.
5. **Block-per-LP layout.** Re-implement the solver so a whole *block* cooperates
   on one LP with the tableau in **shared memory** and warp reductions for pricing.
   Compare against the one-thread-per-LP version as the LP grows.

## Limitations & honesty

- **Synthetic, tiny model.** The committed network is a hand-crafted **synthetic**
  toy (4 metabolites, 8 reactions), labeled synthetic everywhere. It illustrates
  FBA mechanics; it is **not** a real organism, and its "essential" reactions are
  **not** drug-target predictions. No clinical or biological claims are made.
- **Reduced-scope solver.** We use a **dense** bounded-variable simplex sized for
  didactic models. Genome-scale models (~thousands of reactions, ~99% sparse) need
  sparse **revised-simplex / interior-point** solvers (COBRApy + HiGHS/Gurobi).
  The math and the parallelism are identical; only the inner solver differs
  (THEORY.md §7).
- **Objective, not flux, is verified.** LPs can have alternate optima (same growth,
  different fluxes). We compare the unique **objective** value; a full flux
  comparison would need FVA (Exercise 3).
- **Timing is a teaching artifact.** With a handful of tiny LPs the GPU is
  launch-bound and *slower* than the CPU here; the win appears at 10³–10⁵
  knockouts/conditions. The printed milliseconds are illustrative, never a
  benchmark claim (CLAUDE.md §12).
