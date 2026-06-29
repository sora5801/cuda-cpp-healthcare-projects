# 8.14 — Whole-Brain Simulation at Cellular Resolution

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Neuroscience%20%26%20Brain--Computer%20Interfaces-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 8: Neuroscience & Brain-Computer Interfaces · Catalog ID `8.14`
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

Simulating the entire mouse brain (~70 million neurons, ~1 trillion synapses) or human brain (~86 billion neurons) at point-neuron resolution requires exascale computing. Current GPU-capable implementations target mouse brain at simplified LIF models and are a grand-challenge benchmark for neuromorphic hardware. Even 1% of the human brain (~860 million neurons) needs ~10 GB of synaptic state alone. GPU cluster approaches (NEST GPU across many nodes, or NVIDIA H100 NVLink cluster) target this regime; the key bottleneck is sparse synaptic event communication.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Leaky integrate-and-fire / Izhikevich / AdEx at scale, distributed spike event routing (MPI + NCCL), synaptic delay management (distributed ring buffers), STDP online learning at scale, heterogeneous connectivity (random, small-world, structural), balanced E/I network dynamics (Brunel network).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/whole-brain-simulation-at-cellular-resolution.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/whole-brain-simulation-at-cellular-resolution.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\whole-brain-simulation-at-cellular-resolution.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Allen Mouse Brain Connectivity Atlas (https://portal.brain-map.org); HCP structural connectivity (https://db.humanconnectome.org); FlyEM Janelia Drosophila connectome for validation (https://neuprint.janelia.org); Blue Brain Cell Atlas (https://portal.brain-map.org).

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

NEST GPU (https://github.com/nest/nest-simulator) — multi-GPU NEST with CUDA kernel for large network simulation; GeNN (https://github.com/genn-team/genn) — GPU SNN code generation targeting large networks; The Virtual Brain (https://github.com/the-virtual-brain/tvb-root) — whole-brain mean-field at lower resolution; SpikingJelly (https://github.com/fangwei123456/spikingjelly) — PyTorch SNN framework scalable to large populations.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

NCCL for multi-GPU spike event all-to-all communication; custom CUDA kernels for per-neuron state update with register-resident state; cuSPARSE for connectivity matrix-vector product; pattern: GPU-direct MPI for spike routing, neuron state in global memory with warp-coalesced access, NVLink for intra-node GPU communication. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
