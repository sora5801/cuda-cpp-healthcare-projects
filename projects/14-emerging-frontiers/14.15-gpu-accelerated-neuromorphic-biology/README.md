# 14.15 — GPU-Accelerated Neuromorphic Biology

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Emerging%2C%20Theoretical%20%26%20Grand--Challenge%20Frontiers-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 14: Emerging, Theoretical & Grand-Challenge Frontiers · Catalog ID `14.15`
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

Biological neural networks (retina, hippocampus, cortex) integrate spiking dynamics across billions of neurons with trillions of synaptic connections, exhibiting emergent phenomena relevant to neurological disease models and brain-computer interfaces. GPU implementations of spiking neural network (SNN) simulators (GeNN, Brian2CUDA) parallelize over neurons and synaptic update rules, achieving ~1000× speedup over CPU NEST for large-scale cortical column models. GPU neuromorphic simulation of Parkinson's basal ganglia circuits tests deep-brain stimulation parameter spaces in silico. Connection with biology: NVIDIA's H100 NVLink GPU cluster serves as a short-term neuromorphic analog for connectome-scale (C. elegans: 302 neurons, Drosophila: 130K neurons) simulation.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Leaky integrate-and-fire (LIF), Hodgkin-Huxley conductance-based model, spike-timing-dependent plasticity (STDP), GPU event-driven simulation, surrogate gradient training for SNN backpropagation, structural plasticity, large-scale connectome simulation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/gpu-accelerated-neuromorphic-biology.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/gpu-accelerated-neuromorphic-biology.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\gpu-accelerated-neuromorphic-biology.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: FlyWire Drosophila Connectome — 130K neuron wiring diagram (https://flywire.ai/); Allen Brain Connectivity Atlas (https://connectivity.brain-map.org/); Blue Brain Project neocortical data (https://bluebrain.epfl.ch/); OpenNeuromorphic benchmark datasets (verify URL via openneuromorphic.org).

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

GeNN (GPU-enhanced Neuronal Networks) (https://github.com/genn-team/genn) — GPU SNN simulator; Brian2CUDA (https://github.com/brian-team/brian2cuda) — GPU-compiled Brian2 spiking network simulator; PyNN (https://github.com/NeuralEnsemble/PyNN) — SNN abstraction layer; NEURON (GPU branch) (https://github.com/neuronsimulator/nrn) — biophysically detailed neuron simulation with GPU backend.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA warp-level primitives for parallel synaptic weight updates, cuSPARSE for sparse connectivity matrix (connectome), cuRAND for Poisson spike generation; pattern: connectome adjacency matrix (sparse) → GPU spike-event driven propagation → per-neuron LIF/HH ODE integration → STDP weight update → population firing-rate statistics for disease-state comparison. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
