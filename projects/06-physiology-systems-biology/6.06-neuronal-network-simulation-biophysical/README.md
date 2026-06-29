# 6.6 — Neuronal Network Simulation (Biophysical)

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.6`
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

Simulates networks of morphologically detailed (multi-compartment) neurons using Hodgkin-Huxley-style conductance-based kinetics in each dendritic/axonal segment. A single layer-5 pyramidal cell may have 1 000+ compartments each with 10–30 gating variables, and a cortical column model contains thousands of such cells—resulting in millions of coupled ODEs. The Hines solver (tridiagonal Thomas algorithm along each dendritic tree branch) enables efficient per-cell compartmental integration, but parallelizing across cells and synapses is where GPUs excel. Spike delivery (synaptic event processing) introduces irregular memory access that benefits from GPU-side event queues.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Hodgkin-Huxley conductance-based kinetics, Hines tridiagonal solver (branching cable equation), Rush-Larsen exponential integration for gates, event-driven spike delivery, exponential synapse models (AMPA/NMDA/GABA), adaptive time-stepping (CVODE).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/neuronal-network-simulation-biophysical.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/neuronal-network-simulation-biophysical.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\neuronal-network-simulation-biophysical.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: NeuroMorpho.Org — 200 000+ 3D neuronal reconstructions across 900+ species (https://neuromorpho.org); ModelDB / modeldb.science — curated computational neuron models with NEURON/GENESIS files (https://modeldb.science); Allen Brain Cell Atlas — single-cell transcriptomics + patch-seq morpho-electric data (https://portal.brain-map.org); DANDI Archive — neurophysiology datasets in NWB format (https://dandiarchive.org).

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

NEURON + CoreNEURON GPU (https://github.com/neuronsimulator/nrn) — canonical compartmental simulator with CUDA backend via CoreNEURON; NetPyNE (https://github.com/suny-downstate-medical-center/netpyne) — multiscale network builder on top of NEURON with HPC support; MOOSE (https://github.com/BhallaLab/moose-core) — multiscale OO simulator for neuronal + biochemical networks; Blue Brain / Open Brain Institute (https://github.com/BlueBrain) — production-grade cortical column models.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CoreNEURON uses cuSPARSE for Hines matrix batches; custom CUDA kernels for gate ODEs; cuRAND for stochastic synaptic release; pattern: one CUDA thread-block per cell, warp-level branching for dendritic trees; SOA memory layout for coalesced gating variable access. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
