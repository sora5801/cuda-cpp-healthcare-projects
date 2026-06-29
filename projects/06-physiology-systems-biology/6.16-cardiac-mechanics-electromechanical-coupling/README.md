# 6.16 — Cardiac Mechanics & Electromechanical Coupling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.16`
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

Extends electrophysiology simulation by coupling electrical activation to active mechanical contraction through calcium-troponin cross-bridge kinetics (e.g., Rice-Wang-Bers model). The resulting system couples a stiff ODE (ionic + cross-bridge) at each integration point to a nonlinear FEM problem (hyperelastic myocardium with active stress/strain). GPU accelerates both the per-Gauss-point ODE batch and the global Newton-Raphson iterations for the mechanical equilibrium solve. Ventricular pressure-volume loops, ejection fraction, and wall stress distributions are clinical outputs.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Active-stress / active-strain formulations, Holzapfel-Ogden hyperelastic constitutive law, Rice-Wang-Bers cross-bridge kinetics, monodomain EP coupling, Newton-Raphson nonlinear FEM, Guccione passive strain energy, incompressibility via penalty/mixed formulation, Windkessel boundary conditions.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/cardiac-mechanics-electromechanical-coupling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/cardiac-mechanics-electromechanical-coupling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\cardiac-mechanics-electromechanical-coupling.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: UK Biobank CMR + strain imaging (https://www.ukbiobank.ac.uk); Zenodo cardiac mechanics emulation dataset (https://zenodo.org/records/7075055); ACDC segmentation challenge (https://www.creatis.insa-lyon.fr/Challenge/acdc/); MICCAI STACOM cardiac mechanics challenge data (verify URL on grand-challenge.org).

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

FEBio (https://github.com/febiosoftware/FEBio) — nonlinear FEM cardiac/soft-tissue mechanics solver; simcardems (https://github.com/ComputationalPhysiology/simcardems) — FEniCS-based EP+mechanics coupling; OpenCMISS/cm (https://github.com/OpenCMISS/cm) — multi-physics FEM framework; Chaste (https://github.com/Chaste/Chaste) — cardiac electromechanics tutorial.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Batch CVODE GPU for per-Gauss-point ODE; cuSOLVER for Newton linear solve; cuSPARSE SpMV for stiffness matrix assembly; pattern: two-level CUDA grid—elements outer, Gauss points inner—with shared memory for per-element stiffness matrix accumulation. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
