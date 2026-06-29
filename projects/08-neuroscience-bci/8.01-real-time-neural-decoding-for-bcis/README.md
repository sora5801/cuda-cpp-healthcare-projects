# 8.1 — Real-Time Neural Decoding for BCIs

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Neuroscience%20%26%20Brain--Computer%20Interfaces-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 8: Neuroscience & Brain-Computer Interfaces · Catalog ID `8.1`
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

Brain-computer interfaces decode motor intent, speech, or cognitive state from simultaneous recordings of 100–1 000+ neural channels at 30 kHz sampling. The decoding pipeline—bandpass filtering, spike detection, feature extraction, Kalman/population vector decode, output command generation—must complete within 5–50 ms to feel natural to the user. GPU acceleration allows running deep neural decoder networks (1D-CNN, transformer, WaveNet) directly in the decode loop without sacrificing latency through CUDA stream pipelining.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Kalman filter (linear decoder), population vector algorithm (PVA), Wiener filter, linear discriminant analysis (LDA), point-process filter, recurrent neural networks (GRU/LSTM), convolutional temporal decoder, optimal linear estimator (OLE), variational autoencoder latent space decoding.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/real-time-neural-decoding-for-bcis.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/real-time-neural-decoding-for-bcis.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\real-time-neural-decoding-for-bcis.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: BrainGate clinical trial data (https://www.braingate.org — access via collaboration); DANDI Archive intracortical array datasets (https://dandiarchive.org); Allen Brain Observatory Neuropixels data (https://portal.brain-map.org); NLB (Neural Latents Benchmark) — standardized BCI decode benchmarks (https://neurallatents.github.io).

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

BrainFlow (https://github.com/brainflow-dev/brainflow) — unified BCI SDK with real-time GPU-compatible data streaming; OpenBCI GUI (https://github.com/OpenBCI/OpenBCI_GUI) — open-source BCI hardware + software; NLB challenge tools (https://github.com/neurallatents/nlb_tools) — neural latents benchmark evaluation; NDT2/FALCON BCI decode benchmark (verify URL on neurallatents.github.io).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuBLAS for real-time matrix multiply in Kalman predict/update; TensorRT for inference-optimized deep decoder; CUDA streams for pipelined acquire→decode→output with <5 ms latency; pattern: producer-consumer stream pipeline with pinned host memory for zero-copy data ingestion from acquisition hardware. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
