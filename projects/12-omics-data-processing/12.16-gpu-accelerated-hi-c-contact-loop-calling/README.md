# 12.16 — GPU-Accelerated Hi-C Contact Loop Calling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Analytical%20%26%20Omics%20Data%20Processing-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 12: Analytical & Omics Data Processing · Catalog ID `12.16`
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

Hi-C loop calling (HiCCUPS) identifies chromatin loops as enriched point interactions above a 2D background, estimated by a sliding Donut kernel convolution over the contact map. At 5 kb resolution, a human contact map is ~600 k × 600 k (sparse, stored as pairs); the Donut convolution at each potential loop pixel is a GPU embarrassingly parallel 2D operation. NVIDIA's original HiCCUPS paper used a GPU implementation as the default, making this one of the earliest established GPU genomics tools. Recent deep-learning loop callers (Peakachu) apply CNNs to local contact map patches, each patch independently inferrable on GPU.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Donut kernel background estimation (2D convolution on sparse contact map); Poisson enrichment scoring per pixel; multi-resolution peak merging; FDR control for loop calls; Peakachu CNN local patch classification; Fit-Hi-C probability model; anchor pair exhaustive scoring.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/gpu-accelerated-hi-c-contact-loop-calling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/gpu-accelerated-hi-c-contact-loop-calling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\gpu-accelerated-hi-c-contact-loop-calling.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: 4DN Hi-C datasets (https://data.4dnucleome.org/); GEO GSE63525 (Rao 2014) — original HiCCUPS benchmark (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525); ENCODE Hi-C (https://www.encodeproject.org/); 3D Genome Browser datasets (http://3dgenome.fsm.northwestern.edu/).

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

Juicer / HiCCUPS (https://github.com/aidenlab/juicer) — GPU loop caller, original CUDA implementation; Peakachu (https://github.com/tariks/peakachu) — CNN-based loop caller (GPU inference); Higashi (https://github.com/ma-compbio/Higashi) — single-cell Hi-C GPU model; MUSTACHE (https://github.com/ay-lab/mustache) — multi-scale Hi-C loop caller.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom 2D convolution kernels for Donut background; cuSPARSE for sparse contact matrix operations; cuDNN for CNN local-patch loop classification; thrust for sparse pixel sorting; GPU-resident contact map tiles in texture memory. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
