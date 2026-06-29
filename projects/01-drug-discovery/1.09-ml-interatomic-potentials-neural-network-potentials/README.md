# 1.9 — ML Interatomic Potentials (Neural Network Potentials)

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.9`
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

Neural network potentials (NNPs) learn the potential energy surface from ab initio data, reproducing DFT accuracy at near-classical MD speed. Architectures range from atom-centered symmetry functions (ANI) to equivariant message-passing networks (NequIP, MACE, SchNet). GPU acceleration is essential: each forward pass involves neighborhood construction, message passing over all atomic pairs within a cutoff, and backpropagation for forces. On an A100, a 500-atom protein+ligand system runs at ~10 ns/day — 1000× slower than classical FF but 100× faster than DFT, enabling reactive drug-target simulations previously impossible.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Atom-centered symmetry functions (ACSF/BEHLER), equivariant neural networks (E(3)-equivariant / SE(3)), message-passing neural networks (MPNN/SchNet/DimeNet), MACE (multi-ACE), NequIP, neural achitecture via PyTorch Geometric.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/ml-interatomic-potentials-neural-network-potentials.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/ml-interatomic-potentials-neural-network-potentials.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\ml-interatomic-potentials-neural-network-potentials.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: ANI-1ccx — CCSD(T) energies on 500k conformers of drug-like molecules (https://github.com/isayev/ANI1ccx_dataset); SPICE — quantum chemistry dataset for ML potentials covering drug-like molecules and proteins (https://github.com/openmm/spice-dataset); rMD17 — revised MD17 benchmark (https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038); OE62 — 62k organic molecules with DFT energetics (verify URL).

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

TorchANI (https://github.com/aiqm/torchani) — PyTorch ANI NNP with CUDA acceleration and OpenMM integration; TorchMD-Net (https://github.com/torchmd/torchmd-net) — equivariant NNPs with GPU-optimized neighbor list; MACE (https://github.com/ACEsuit/mace) — fast equivariant NNP with GPU kernels; NequIP (https://github.com/mir-group/nequip) — E(3)-equivariant network for accurate NNPs.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

PyTorch CUDA autograd for force computation via backpropagation; custom CUDA kernels for neighbor list construction with periodic boundaries; torch.compile/TorchScript for inference optimization; multi-GPU via DDP for training. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
