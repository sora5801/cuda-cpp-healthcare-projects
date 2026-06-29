# 13.9 — Target-Mediated Drug Disposition (TMDD)

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Pharmacology%20%26%20Clinical%20Quantitative%20Modeling-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 13: Pharmacology & Clinical Quantitative Modeling · Catalog ID `13.9`
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

TMDD models describe biologics (monoclonal antibodies, bispecifics) whose elimination is dominated by saturable binding to their pharmacological target, producing nonlinear, dose-dependent PK. The full TMDD ODE system (Mager-Jusko, 2001) is stiff due to fast receptor association/dissociation kinetics, requiring implicit stiff solvers. GPU parallelism is critical for virtual patient population simulations: fitting 1000 virtual patients × 100 dose schedules × stiff ODE = 10⁵ independent stiff integrations run simultaneously on GPU. Approximations (quasi-steady-state, Michaelis-Menten) reduce stiffness but must be validated against full TMDD for each compound — GPU enables this validation across large parameter grids cheaply.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Full TMDD ODE system (4 equations: free drug, free receptor, drug-receptor complex, total drug), Quasi-Equilibrium (QE) approximation, Quasi-Steady-State (QSS) / Michaelis-Menten approximation, stiff LSODA/CVODE integration, bivalent TMDD extensions (2025 Straube model), population NLME fitting of TMDD, slow-binding approximation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/target-mediated-drug-disposition-tmdd.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/target-mediated-drug-disposition-tmdd.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\target-mediated-drug-disposition-tmdd.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Published mAb PK datasets from Phase I trials (verify via PharmPK or ClinicalPharmacology.nih.gov) Open Systems Pharmacology TMDD model examples (https://github.com/Open-Systems-Pharmacology/) NONMEM TMDD example scripts (verify URL) BioModels Database TMDD models (https://www.ebi.ac.uk/biomodels/)

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

Pumas (https://pumas.ai/) — GPU population TMDD fitting in Julia NONMEM (https://www.iconplc.com/solutions/technologies/nonmem/) — industry standard NLME for TMDD (verify GPU support status) Monolix TMDD library (https://lixoft.com/model-libraries/pkpd-library/) — pre-built TMDD models (verify URL) nvQSP (https://github.com/NVIDIA-Digital-Bio/nvQSP) — GPU stiff ODE solver applicable to TMDD virtual patient simulations

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA CVODE/RODAS4 stiff solver, cuBLAS for Jacobian LU factorisation in implicit integration; pattern: batch-parallel stiff ODE integration — one virtual patient per CUDA thread block, receptor binding equations in shared memory. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
