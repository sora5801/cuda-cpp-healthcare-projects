# 6.7 — Spiking Neural Network (Point-Neuron) Simulation

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.7`
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

Point-neuron SNN models (leaky integrate-and-fire, Izhikevich, adaptive exponential IF) sacrifice morphological detail in exchange for simulating networks of millions to billions of neurons in real time. Each neuron updates a handful of state variables per time step; spikes generate synaptic current injections to thousands of target neurons via a connectivity matrix that is typically sparse (~10 000 synapses/neuron). GeNN generates custom CUDA kernels from user model descriptions, achieving real-time simulation of 10⁶-neuron Izhikevich networks on a single GPU. NEST GPU and Brian2CUDA follow similar kernel-generation approaches.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Leaky integrate-and-fire (LIF), Izhikevich neuron model, adaptive exponential integrate-and-fire (AdEx), spike-timing-dependent plasticity (STDP), exponential/alpha synapse kernels, delay-line spike queues, random balanced-network (Brunel) connectivity.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/spiking-neural-network-point-neuron-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/spiking-neural-network-point-neuron-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\spiking-neural-network-point-neuron-simulation.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Allen Brain Observatory — visual cortex spiking data from Neuropixels (https://portal.brain-map.org); DANDI Archive — electrophysiology datasets NWB format (https://dandiarchive.org); OpenNeuro — EEG/MEG recordings for network model validation (https://openneuro.org); Human Connectome Project structural connectivity matrices (https://db.humanconnectome.org).

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

GeNN (https://github.com/genn-team/genn) — GPU-enhanced SNN code generator (CUDA + HIP), includes Brian2GeNN and ml_genn deep SNN; SpikingJelly (https://github.com/fangwei123456/spikingjelly) — PyTorch-based SNN framework with CUDA extensions; Brian2CUDA (https://github.com/brian-team/brian2cuda) — CUDA code generation backend for Brian2; NEST GPU (https://github.com/nest/nest-simulator) — multi-GPU NEST backend scaling to 10⁹ neurons.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom generated CUDA kernels (GeNN/Brian2CUDA), cuSPARSE for synaptic current summation via sparse matrix-vector product, cuRAND for Poisson spike generation; pattern: one thread per neuron for state update, warp-shuffle for local spike detection, atomic-add for synaptic current accumulation. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
