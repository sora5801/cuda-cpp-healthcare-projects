# 2.1 — Protein Structure Prediction Inference (AlphaFold-class)

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟢 Beginner · Established** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.1`
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

AlphaFold2 and its successors (RoseTTAFold, ESMFold, OpenFold, Boltz-1, AlphaFold3) predict atomic-resolution 3D protein structures from amino acid sequences using deep learning. The Evoformer stack processes multiple sequence alignments (MSAs) and pair representations through stacked self-attention and triangle-multiplicative update layers — each requiring enormous GPU memory (an A100 40GB handles ~5000 residues for AF2). GPU inference is mandatory: predicting a 500-residue protein takes ~5 minutes on GPU vs. ~12 hours on CPU. ESMFold bypasses MSA entirely, using a 15B-parameter language model for 10–60× faster prediction.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Evoformer (MSA row/column attention + triangle updates), Structure Module (invariant point attention, IPA), recycling iterations, template attention, diffusion-based structure generation (AF3), confidence scoring (pLDDT, PAE).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/protein-structure-prediction-inference-alphafold-class.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/protein-structure-prediction-inference-alphafold-class.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\protein-structure-prediction-inference-alphafold-class.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: AlphaFold Database — 200M+ predicted structures (https://alphafold.ebi.ac.uk/); RCSB PDB — 227k+ experimental structures (https://www.rcsb.org); UniProt/UniRef90 MSA databases (https://www.uniprot.org); CAMEO/CASP15 structure prediction benchmarks (https://www.cameo3d.org).

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

AlphaFold2 (https://github.com/google-deepmind/alphafold) — official DeepMind implementation; OpenFold (https://github.com/aqlaboratory/openfold) — trainable GPU-friendly PyTorch AF2; ESMFold (https://github.com/facebookresearch/esm) — MSA-free language model structure prediction; Boltz-1 (https://github.com/jwohlwend/boltz) — fully open AF3-level biomolecular complex prediction.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN multi-head attention for Evoformer; custom CUDA triangle update kernels; FP16/BF16 mixed precision; flash attention (FlashAttention2) for memory-efficient MSA attention; multi-GPU model parallelism for large complexes. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
