# 1.35 — QMMM/ML Potential Hybrid MD

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.35`
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

The next frontier beyond QM/MM is using ML potentials trained on QM data to replace the expensive QM region — enabling microsecond reactive MD at QM accuracy. GPU-accelerated equivariant NNPs (MACE, NequIP) can serve as drop-in QM replacements in an MM environment. This hybrid NNP/MM approach runs fully on GPU: the NNP forward pass and MM evaluation occur in overlapping CUDA streams. Challenges include training data coverage for reactive intermediates and accurate long-range electrostatics across the QM-ML/MM boundary.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

NNP/MM coupling, link-atom boundary treatment, active learning for reactive system NNP training, δ-ML correction to DFT, equivariant NNP with long-range electrostatic correction.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/qmmm-ml-potential-hybrid-md.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/qmmm-ml-potential-hybrid-md.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\qmmm-ml-potential-hybrid-md.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: ANI-1ccx reactive extensions (verify URL); DFT reaction pathway datasets from QM/MM studies; Transition1x — 10M DFT calculations along reaction paths (https://zenodo.org/record/5781475); SPICE dataset (https://github.com/openmm/spice-dataset).

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

TorchMD-Net (https://github.com/torchmd/torchmd-net) — equivariant NNP with MM coupling; MACE (https://github.com/ACEsuit/mace) — fast NNP for hybrid ML/MM; OpenMM-ML (https://github.com/openmm/openmm-ml) — NNP/MM interface for OpenMM; NNPOps (https://github.com/openmm/NNPOps) — CUDA-optimized NNP primitives.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA MACE kernels for equivariant message passing; OpenMM CUDA platform for MM region; CUDA streams for async NNP+MM; PyTorch autograd for NNP force gradients; cuBLAS for spherical harmonic transforms. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
