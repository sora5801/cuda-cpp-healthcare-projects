# 12.5 — Real-Time Sequencing Analysis / Adaptive Sampling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Analytical%20%26%20Omics%20Data%20Processing-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 12: Analytical & Omics Data Processing · Catalog ID `12.5`
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

Oxford Nanopore adaptive sampling (ReadUntil API) allows the sequencer to reject reads in real time (within 200 ms per read) based on a computational decision—requiring GPU basecalling and alignment to complete in under ~100 ms per read chunk. The pipeline: raw signal → GPU basecalling (Dorado, HAC model) → GPU seed-extension to reference → accept/reject decision → signal to sequencer. GPU processing is not optional; CPU pipelines are too slow for the 200 ms window. This enables on-target enrichment without library preparation: unwanted chromosomal regions are skipped by reversing the voltage to eject the DNA strand.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

GPU CTC basecalling (Dorado transformer); approximate hash seed alignment (minimap2 GPU); streaming input buffer management; read-until decision tree; pore blocking prediction; real-time sequence classification (pathogen typing).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/real-time-sequencing-analysis-adaptive-sampling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/real-time-sequencing-analysis-adaptive-sampling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\real-time-sequencing-analysis-adaptive-sampling.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: ONT open datasets with ReadUntil metadata (https://github.com/GoekeLab/awesome-nanopore); NCBI SRA real-time sequencing runs (https://www.ncbi.nlm.nih.gov/sra); ENA clinical nanopore studies (https://www.ebi.ac.uk/ena); Oxford Nanopore public data portal (https://labs.epi2me.io/dataindex/).

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

Dorado (https://github.com/nanoporetech/dorado) — GPU basecaller with low-latency streaming mode; ReadFish (https://github.com/looselab/readfish) — ReadUntil adaptive sampling controller; Icarust (https://github.com/LooseLab/Icarust) — real-time nanopore simulator for pipeline testing; MinKNOW (ONT proprietary) — sequencer control with GPU basecalling integration.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

TensorRT for ultra-low-latency RNN inference; CUDA streams for overlapping signal decode and alignment; persistent GPU kernel for continuous signal ingestion; GPU ring buffer for streaming POD5 signal; multi-GPU for PromethION multi-flow-cell setups. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
