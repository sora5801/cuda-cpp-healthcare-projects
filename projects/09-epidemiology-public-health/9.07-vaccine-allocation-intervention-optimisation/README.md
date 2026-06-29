# 9.7 — Vaccine Allocation & Intervention Optimisation

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Epidemiology%20%26%20Public%20Health-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 9: Epidemiology & Public Health · Catalog ID `9.7`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

<!-- =======================================================================
     SCAFFOLD STATUS: this README was stamped from the catalog. The prose
     fields below (Deep dive / Algorithms / Datasets / Prior art) are filled
     in from the catalog. Sections marked TODO(impl)/TODO(theory) must be
     completed by the project author before this project is "done"
     (see CLAUDE.md §4.1 and tools/verify_project.py).
     ======================================================================= -->

## Summary

TODO(impl): One paragraph, plain language — what this project does and why a
learner should care. (Seed from the deep dive below.)

## What this computes & why the GPU helps

Determines optimal allocation of limited vaccines, treatments, or non-pharmaceutical interventions across age groups, geographic regions, or risk strata to minimise deaths or infections under resource constraints. GPU-accelerated simulation (agent-based or compartmental) enables rapid evaluation of thousands of candidate allocation policies within an optimisation loop. Reinforcement learning approaches (PPO, SAC) train on GPU-simulated environments where the epidemic simulator is the transition function. Multi-objective Pareto optimisation across equity and efficiency criteria requires GPU-parallelised NSGA-II or similar evolutionary algorithms.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Multi-objective optimisation (NSGA-II, NSGA-III), Proximal Policy Optimisation (PPO), Deep Q-Networks on simulation environments, Thompson sampling for adaptive allocation, network-based vaccinating-hub strategies (targeted vs. random), stochastic programming under epidemiological uncertainty, integer linear programming for logistics.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/vaccine-allocation-intervention-optimisation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/vaccine-allocation-intervention-optimisation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\vaccine-allocation-intervention-optimisation.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the result, shows the
GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/` — a tiny, offline input so the demo runs
  with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (documented, idempotent).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: GLEAM global mobility network for spatial allocation (https://www.gleamviz.org/) WHO Immunisation Data — vaccination coverage by country and vaccine (https://immunizationdata.who.int/) US Census commuting flows — for workplace transmission modelling (https://www.census.gov/) COVID-19 vaccination time series (Our World in Data) — historical rollout data for calibration (https://ourworldindata.org/covid-vaccinations)

## Expected output

Success looks like `demo/expected_output.txt`. The program computes the result on
both the **GPU** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`)
and asserts they agree within the documented tolerance — that agreement is the
correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
2. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
3. [`src/kernels.cu`](src/kernels.cu) — the kernel(s) and host wrapper.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline.
5. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

Covasim (https://github.com/InstituteforDiseaseModeling/covasim) — GPU-friendly Python COVID-19 agent-based model EMOD (https://github.com/InstituteforDiseaseModeling/EMOD) — high-performance individual-based disease model Stable Baselines 3 (https://github.com/DLR-RM/stable-baselines3) — GPU RL library for policy training on epidemic environments Pymoo (https://github.com/anyoptimization/pymoo) — multi-objective optimisation with GPU evaluation support

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuRAND for stochastic epidemic simulation, custom CUDA ODE kernels for compartmental model evaluation, CUDA graph for repeated fixed-topology GPU execution; pattern: population of candidate policies evaluated simultaneously across GPU thread blocks. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
