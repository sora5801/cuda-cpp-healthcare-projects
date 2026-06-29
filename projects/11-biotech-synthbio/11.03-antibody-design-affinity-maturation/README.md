# 11.3 — Antibody Design & Affinity Maturation

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biotechnology%2C%20Bioprocess%20%26%20Synthetic%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 11: Biotechnology, Bioprocess & Synthetic Biology · Catalog ID `11.3`
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

Antibody engineering spans CDR-loop design, affinity maturation, and developability optimization — each requiring GPU inference over large sequence/structure spaces. RFdiffusion-Antibody (Baker Lab, 2025) generates novel CDR-H3 loops conditioned on antigen epitopes via SE(3)-equivariant diffusion on GPU. Affinity maturation via flow matching (AffinityFlow, 2025) guides sequence trajectories toward high-affinity regions on GPU. Structure-aware inverse folding (AbMPNN) redesigns CDR sequences while preserving Fv geometry. The AbBiBench benchmark (2025) standardizes evaluation across 10+ affinity maturation methods. The FDA approved 13 new monoclonal antibodies in 2024, underlining the industrial importance of accelerated in silico design.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

SE(3)-equivariant diffusion (RFdiffusion), flow matching for affinity maturation (AffinityFlow), inverse folding (AbMPNN/ProteinMPNN), language-model-guided combinatorial optimization (LLM + genetic algorithm + simulated annealing), ΔΔG binding affinity prediction (Rosetta flex_ddg, FoldX), multi-objective developability scoring.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/antibody-design-affinity-maturation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/antibody-design-affinity-maturation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\antibody-design-affinity-maturation.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: SAbDab — Structural Antibody Database, 10000+ Fv structures (https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/); AbBiBench Benchmark — standardized affinity maturation evaluation (https://arxiv.org/abs/2506.04235); OAS — Observed Antibody Space, 2B+ sequences (https://opig.stats.ox.ac.uk/webapps/oas/oas); CoV-AbDab — SARS-CoV-2 antibody database (https://opig.stats.ox.ac.uk/webapps/covabdab/).

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

RFdiffusion (https://github.com/RosettaCommons/RFdiffusion) — CDR design via SE(3) diffusion (RFdiffusion2 available 2025); ABodyBuilder3 (https://github.com/oxpig/ABDB) — GPU antibody structure prediction; ImmuneBuilder (https://github.com/oxpig/ImmuneBuilder) — GPU-fast Fv structure modeling; AbMPNN/ProteinMPNN (https://github.com/dauparas/ProteinMPNN) — GPU CDR sequence design.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Flash Attention for long CDR+antigen context, cuDNN Transformer inference for LLM-based sequence scoring, CUDA kernels for parallel ΔΔG evaluation; pattern: antigen epitope input → RFdiffusion GPU generates CDR scaffold ensemble → AbMPNN scores/redesigns sequences in batch → GPU ΔΔG filter → developability scoring → top candidates to wet lab. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
