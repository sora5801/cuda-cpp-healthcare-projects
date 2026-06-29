# 3.13 — Pangenome Graph Alignment

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.13`
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

Pangenome graphs encode the genomic variation of an entire population as a sequence graph (GFA format) rather than a single linear reference; aligning reads to this graph involves generalised DP over a DAG of paths rather than a 1D reference. The vg toolkit's graph alignment applies a generalised Smith-Waterman on the graph DAG, which is harder to parallelise than linear alignment due to irregular memory access. A 2024 SC paper demonstrated GPU-accelerated pangenome layout achieving 57.3× speedup over multi-core CPU for the ODGI layout algorithm by mapping node-force computations to GPU threads. Graph seeding via GBWT/r-index also benefits from parallelised BWT operations.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Generalised DAG DP alignment; GBWTgraph / r-index graph BWT; pangenome graph layout (force-directed, GPU particles); ODGI path sorting and sorting optimisation; seqwish overlap-to-graph induction; wfmash wavefront alignment for all-to-all seeding.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/pangenome-graph-alignment.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/pangenome-graph-alignment.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\pangenome-graph-alignment.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Human Pangenome Reference Consortium (HPRC) — 94 haplotype-resolved assemblies (https://humanpangenome.org/); 1000 Genomes Project GVCFs — variant calls for graph construction (https://www.internationalgenome.org/data); Ensembl Pangenome — multi-species graphs (https://www.ensembl.org/); PGGB tutorial data (https://github.com/pangenome/pggb).

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

vg (https://github.com/vgteam/vg) — comprehensive variation graph toolkit; PGGB (https://github.com/pangenome/pggb) — Pangenome Graph Builder pipeline; ODGI (https://github.com/pangenome/odgi) — GPU layout algorithms; Rapid GPU-based pangenome layout paper (https://www.csl.cornell.edu/~zhiruz/pdfs/pangenome-layout-sc2024.pdf) — 57× speedup reference.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA force-directed layout kernels (Barnes-Hut approximation on GPU); parallel graph BFS for BWT construction; thrust for node-position sort; cuSPARSE for sparse adjacency matrix traversal; one CUDA thread per node-force computation. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
