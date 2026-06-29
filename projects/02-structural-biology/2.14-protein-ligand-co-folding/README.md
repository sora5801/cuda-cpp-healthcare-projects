# 2.14 — Protein-Ligand Co-Folding

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.14`
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

Co-folding models simultaneously predict protein structure and ligand binding pose in a single forward pass, bypassing separate docking steps. Boltz-1 and AlphaFold3 accept ligand SMILES and protein sequence as joint inputs to a diffusion model conditioned on molecular features. GPU inference generates protein-ligand complex structures at near-FEP accuracy for pose prediction in minutes per complex. The GPU bottleneck is the diffusion sampling loop (50–200 denoising steps), each requiring a full attention forward pass over the joint protein-ligand token sequence.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Joint protein-ligand diffusion (DDPM on 3D positions), conditional atom-type and geometry generation, atom-level self-attention with periodic boundary handling, confidence (pLDDT/iPAE) scoring, cross-attention between protein and ligand tokens.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/protein-ligand-co-folding.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/protein-ligand-co-folding.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\protein-ligand-co-folding.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: PoseBusters benchmark — 428 recently released PDB complexes (https://github.com/maabuu/posebusters); PDB-bind v2020 (http://www.pdbbind.org.cn); Astex Diverse Set — 85 drug-like ligand complex structures (verify URL); CASF cross-docking benchmarks (http://www.pdbbind.org.cn/casf.php).

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

Boltz-1 (https://github.com/jwohlwend/boltz) — GPU co-folding of protein-ligand-nucleic acid complexes; NeuralPLexer3 (https://github.com/zrqiao/NeuralPLexer) — state-specific co-folding with CUDA; AlphaFold3 (https://github.com/google-deepmind/alphafold3) — official AF3 with ligand support; DiffDock (https://github.com/gcorso/DiffDock) — diffusion docking without co-folding (complementary).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Flash attention (FlashAttention2) for long joint sequences; cuDNN transformer blocks; GPU diffusion denoising loop with CUDA noise schedules; FP16/BF16 precision; multi-GPU model parallelism for large complexes. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
