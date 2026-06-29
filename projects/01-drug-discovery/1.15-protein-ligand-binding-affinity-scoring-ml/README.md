# 1.15 — Protein-Ligand Binding Affinity Scoring (ML)

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.15`
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

End-to-end ML scoring functions learn protein-ligand interaction energy surrogates directly from structural data, bypassing physics-based force fields. Models range from 3D-CNNs over voxelized complexes to equivariant GNNs over atom graphs to transformer co-folding models (NeuralPLexer3). GPU inference enables rapid rescoring of millions of docking poses in virtual screening — a 3D-CNN scores a pose in ~1 ms on a GPU vs. >1 s for FEP. The fundamental challenge is generalization across chemical space and protein families.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

3D-CNN on atomic density grids, equivariant graph neural networks (SchNet/DimeNet++), attention-based protein-ligand co-attention, diffusion-based co-folding (NeuralPLexer), Random Forest on PLEC/ECIF interaction fingerprints.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/protein-ligand-binding-affinity-scoring-ml.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/protein-ligand-binding-affinity-scoring-ml.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\protein-ligand-binding-affinity-scoring-ml.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: PDB-bind v2020 — 19,443 protein-ligand complexes with Kd/Ki (http://www.pdbbind.org.cn); CASF-2016 benchmark (http://www.pdbbind.org.cn/casf.php); ChEMBL activity data (https://www.ebi.ac.uk/chembl/); BindingDB — 2.8M measured binding affinities (https://www.bindingdb.org).

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

NeuralPLexer (https://github.com/zrqiao/NeuralPLexer) — state-specific co-folding with binding affinity, requires CUDA; GNINA (https://github.com/gnina/gnina) — CNN rescoring in docking pipeline; DiffDock (https://github.com/gcorso/DiffDock) — generative docking with affinity proxy; DeepChem (https://github.com/deepchem/deepchem) — includes AtomicConvolutions and MPNN-based scoring.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN for 3D-CNN layers; PyTorch Geometric CUDA kernels for equivariant message passing; FP16 mixed precision for throughput; GPU-parallel batch scoring for post-docking rescoring of millions of poses. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
