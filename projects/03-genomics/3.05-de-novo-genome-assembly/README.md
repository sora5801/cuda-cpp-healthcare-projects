# 3.5 — De Novo Genome Assembly

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.5`
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

De novo assembly reconstructs a genome from raw reads without a reference. The three GPU-amenable bottlenecks are: (1) all-vs-all read overlap detection (O(n²) pairwise alignment), (2) string-graph / De Bruijn graph construction from k-mers, and (3) consensus polishing of draft contigs. NVIDIA's GenomeWorks / racon-GPU accelerates the polishing stage (partial-order alignment MSA) by 70× vs. CPU. The Darwin accelerator paper showed 109× GPU speedup for read overlap on PacBio data. Modern HiFi assembly (hifiasm) is CPU-centric for the string-graph phase, but GPU kernels for pairwise overlap computation are an active insertion point; NVIDIA's Clara de novo pipeline on NGC wraps these components.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

All-vs-all minimiser-based overlap (minimap2 kernel); De Bruijn graph construction and traversal; string-graph simplification (unitig / contig threading); partial-order alignment (POA) for polishing consensus; repeat resolution by Hi-C scaffolding.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/de-novo-genome-assembly.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/de-novo-genome-assembly.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\de-novo-genome-assembly.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: CHM13 telomere-to-telomere human genome — the T2T gold standard for assembly benchmarking (https://github.com/marbl/CHM13); GenomeArk — vertebrate genome assembly data (https://genomeark.github.io/); Human Pangenome Reference Consortium data (https://humanpangenome.org/); SRA PacBio HiFi and ONT datasets — species-specific de novo projects (https://www.ncbi.nlm.nih.gov/sra).

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

GenomeWorks / racon-GPU (https://github.com/NVIDIA-Genomics-Research/GenomeWorks) — GPU-accelerated overlap and polishing; Clara De Novo Assembly (https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/resources/clara_denovo_assembly_pipeline) — NVIDIA NGC end-to-end pipeline; hifiasm (https://github.com/chhylp123/hifiasm) — state-of-the-art HiFi assembler (CPU, GPU overlap insertion point); Racon CPU reference (https://github.com/lbcb-sci/racon) — CPU polishing baseline.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom POA kernels in GenomeWorks (shared-memory DP); CUDA thrust for k-mer sorting; minimiser hash tables in GPU global memory; multi-GPU for embarrassingly parallel read-pair overlaps. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
