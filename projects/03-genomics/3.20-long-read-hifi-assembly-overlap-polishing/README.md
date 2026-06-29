# 3.20 — Long-Read HiFi Assembly Overlap & Polishing

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.20`
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

PacBio HiFi reads (10–25 kb, >99.5% accuracy) enable near-perfect de novo assemblies but the all-vs-all read overlap step—finding which reads share sequence—is computationally prohibitive: N=20 M reads requires O(N²) comparisons naively. GPU parallelism accelerates the minimiser-based seed look-up and seed chain extension across read pairs. The Darwin read overlapper GPU implementation achieved 109× speedup by storing the minimiser hash table in GPU global memory and resolving seed chains in parallel CUDA blocks. Post-overlap polishing (racon, medaka) is similarly accelerated by GPU POA and RNN inference kernels.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Minimiser hashing for all-vs-all overlap seeding; sparse chain DP for seed-chain scoring; partial-order alignment (POA) for consensus polishing; string graph simplification and unitig generation; haplotype phasing via heterozygous marker threading.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/long-read-hifi-assembly-overlap-polishing.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/long-read-hifi-assembly-overlap-polishing.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\long-read-hifi-assembly-overlap-polishing.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: PacBio SMRT Human WGS (HG002/HG003/HG004 trio) (https://www.ncbi.nlm.nih.gov/sra); Vertebrate Genomes Project PacBio HiFi assemblies (https://vertebrategenomesproject.org/); GenomeArk HiFi datasets (https://genomeark.github.io/); CHM13 T2T HiFi reads (https://github.com/marbl/CHM13).

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

Darwin GPU overlapper (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7495891/) — 109× GPU speedup for PacBio overlap; hifiasm (https://github.com/chhylp123/hifiasm) — state-of-the-art HiFi assembler; racon-GPU (https://github.com/NVIDIA-Genomics-Research/racon-gpu) — GPU consensus polishing; Medaka (https://github.com/nanoporetech/medaka) — RNN-based polishing with GPU inference.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

GPU-resident minimiser hash map; custom seed-chain CUDA kernels; POA DP in shared memory per thread block; cuDNN RNN for Medaka polishing; CUDA streams pipelining I/O and compute. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
