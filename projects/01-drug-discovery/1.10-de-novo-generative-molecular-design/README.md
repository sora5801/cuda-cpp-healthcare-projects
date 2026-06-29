# 1.10 — De Novo Generative Molecular Design

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.10`
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

Generative models learn the distribution of drug-like molecules and sample novel structures optimized for multiple properties (potency, selectivity, ADMET, synthesizability). GPU training is mandatory: large transformer/RNN/diffusion models over SMILES strings or 3D molecular graphs require days on multi-GPU nodes. At inference, reinforcement learning (RL) fine-tuning generates thousands of candidate molecules per GPU-second, enabling goal-directed optimization. REINVENT4 combines RL with curriculum learning on SMILES; diffusion-based methods (DiffSBDD, TargetDiff) generate molecules directly in 3D protein binding pockets.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Variational autoencoders (VAE), transformer language models on SMILES/SELFIES, graph generative models, denoising diffusion probabilistic models (DDPM), reinforcement learning with REINFORCE/PPO, scoring functions (docking, QED, SA score).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/de-novo-generative-molecular-design.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/de-novo-generative-molecular-design.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\de-novo-generative-molecular-design.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: ChEMBL — 2M+ bioactive molecules (https://www.ebi.ac.uk/chembl/); ZINC20 — 1.4B purchasable compounds (https://zinc20.docking.org); GuacaMol benchmark — distribution learning and goal-directed generation benchmarks (https://github.com/BenevolentAI/guacamol); MOSES — molecular generation benchmarks (https://github.com/molecularsets/moses).

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

REINVENT4 (https://github.com/MolecularAI/REINVENT4) — production SMILES generative model with RL, Apache 2.0 license; DiffSBDD (https://github.com/arneschneuing/DiffSBDD) — 3D structure-based diffusion design; DiffDock (https://github.com/gcorso/DiffDock) — diffusion model for pose generation used in SBDD pipelines; DeepChem (https://github.com/deepchem/deepchem) — broad ML drug discovery toolkit including generative models.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN for transformer/RNN layers; custom CUDA scatter/gather for molecular graph message passing; multi-GPU DDP training; FP16 mixed precision via torch.amp; GPU-batched scoring function evaluation during RL rollouts. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
