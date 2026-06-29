# 8.16 — Neural Signal Compression & Wireless BCI Transmission

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Neuroscience%20%26%20Brain--Computer%20Interfaces-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 8: Neuroscience & Brain-Computer Interfaces · Catalog ID `8.16`
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

Fully implanted high-channel-count BCIs (1 024–65 000 electrodes in emerging platforms) cannot transmit raw 30 kHz × N-channel data wirelessly due to power/bandwidth limits. GPU-accelerated on-device compression (threshold crossing, wavelet compression, PCA projection, spike detection) must reduce data 100–1 000× before wireless transmission. Implantable ASICs perform this in hardware, but GPU simulation of compression algorithms enables algorithm design and fidelity evaluation before silicon tape-out.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Threshold-based spike detection, wavelet packet decomposition (WPD), compressed sensing (L1 minimization / OMP), PCA projection for dimensionality reduction, delta-encoding, Huffman/arithmetic coding, matched filter spike detection, signal reconstruction via iterative thresholding (ISTA/FISTA).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/neural-signal-compression-wireless-bci-transmission.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/neural-signal-compression-wireless-bci-transmission.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\neural-signal-compression-wireless-bci-transmission.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: DANDI Neuropixels recordings (https://dandiarchive.org); BrainGate implanted array datasets (https://www.braingate.org); SpikeInterface benchmark recordings (https://spikeinterface.readthedocs.io); PhysioNet neural datasets (https://physionet.org).

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

BrainFlow (https://github.com/brainflow-dev/brainflow) — real-time neural signal SDK; SpikeInterface (https://github.com/SpikeInterface/spikeinterface) — spike detection and feature extraction pipeline; PyWavelets (https://github.com/PyWavelets/pywt) — wavelet decomposition with CuPy GPU backend; FISTA implementations in PyTorch (verify URL — numerous public repos).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuFFT for wavelet and frequency-domain feature extraction; cuBLAS for PCA projection (matrix-vector multiply); CUDA Thrust for threshold scan across all channels; pattern: streaming pipeline—raw samples in via DMA, CUDA kernels for detection and projection, compressed output via pinned host memory ring buffer. -- *All GitHub URLs have been verified against search results as of June 2026. URLs marked (verify URL) could not be confirmed from available search results and should be independently checked. Key caveats: the Blue Brain Project GitHub org (https://github.com/BlueBrain) remains accessible but active development has migrated to https://github.com/openbraininstitute following the project's conclusion in December 2024. ModelDB is migrating from https://senselab.med.yale.edu/ModelDB to https://modeldb.science. NVIDIA Thrust is archived in favour of the unified CCCL repo at https://github.com/NVIDIA/cccl.* ## Sections 7, 9, and 13 — Exhaustive Deep-Dive Reference --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
