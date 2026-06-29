# 11.2 — Enzyme Design & Catalysis Modeling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biotechnology%2C%20Bioprocess%20%26%20Synthetic%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 11: Biotechnology, Bioprocess & Synthetic Biology · Catalog ID `11.2`
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

Computational enzyme design requires evaluating active-site geometry, transition-state stabilization, and substrate binding simultaneously. GPU-accelerated QM/MM (quantum mechanics / molecular mechanics) couples a DFT or semi-empirical QM region around the catalytic residues with a classical MM region of the full enzyme, enabling thousands of candidate enzyme structures to be ranked. Rosetta enzyme design generates theozyme scaffolds and then repacks surrounding residues on GPU. AlphaFold-2 structure prediction + ProteinMPNN sequence design creates novel enzyme candidates at scale. De novo enzyme design for non-natural reactions (Diels-Alder, retro-aldol) has been demonstrated computationally; GPU acceleration is the bottleneck to scaling to large combinatorial searches.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Rosetta enzyme design (RIF docking, match/scaffold search), QM/MM (ONIOM, pDynamo), transition-state theory rate prediction, directed evolution fitness landscape modeling, SE(3)-equivariant active-site design (BindCraft/RFdiffusion), Monte Carlo backrub for enzyme refinement.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/enzyme-design-catalysis-modeling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/enzyme-design-catalysis-modeling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\enzyme-design-catalysis-modeling.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: BRENDA Enzyme Database — kinetics, substrates, organisms (https://www.brenda-enzymes.org/); SABIO-RK — enzyme kinetic parameters (https://sabiork.h-its.org/); UniProt/SwissProt enzyme entries (https://www.uniprot.org/); M-CSA Mechanism and Catalytic Site Atlas (https://www.ebi.ac.uk/thornton-srv/m-csa/).

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

RFdiffusion (https://github.com/RosettaCommons/RFdiffusion) — diffusion-based active-site design; PyRosetta (https://github.com/RosettaCommons/pyrosetta) — GPU-compatible Rosetta Python bindings; GROMACS (https://github.com/gromacs/gromacs) — GPU QM/MM enzyme MD via ORCA/CP2K coupling; DeepMind AlphaFold2 (https://github.com/google-deepmind/alphafold) — structure prediction for enzyme scaffold validation.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN for Rosetta energy term neural-network surrogate, CUDA ONIOM QM/MM kernels via GROMACS GPU engine, cuFFT for periodic electrostatics (PME); pattern: RFdiffusion generates active-site scaffold on GPU → ProteinMPNN designs sequence → GPU MD relaxation → GPU QM/MM ΔG‡ evaluation → rank and select. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
