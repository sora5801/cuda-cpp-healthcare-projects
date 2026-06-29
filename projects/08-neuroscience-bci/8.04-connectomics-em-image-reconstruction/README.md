# 8.4 — Connectomics / EM Image Reconstruction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Neuroscience%20%26%20Brain--Computer%20Interfaces-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 8: Neuroscience & Brain-Computer Interfaces · Catalog ID `8.4`
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

Volume electron microscopy (serial-section TEM, FIB-SEM) generates terabyte-to-petabyte image volumes (4 nm/voxel for nanometer-resolution synaptic ultrastructure). GPU-accelerated convolutional neural networks (3D U-Net, flood-filling networks) perform dense semantic segmentation of neurons, mitochondria, and synapses. Watershed-based instance segmentation and agglomeration follow, then automated synapse detection and connectivity graph extraction. The H01 human cortical connectome dataset is 1.4 PB; the FlyEM hemibrain is 26 TB.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

3D U-Net for voxel affinity prediction, flood-filling networks (recurrent CNN), watershed agglomeration (Kruskal/Prim on affinity graph), multicut graph partitioning, synapse detection (3D detection network), stitching and alignment (SIFT + RANSAC), mean shift/DBSCAN for spine detection.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/connectomics-em-image-reconstruction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/connectomics-em-image-reconstruction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\connectomics-em-image-reconstruction.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Google H01 Human Cortex Connectome — 1.4 PB, 1 mm³ human cortex (https://h01-release.storage.googleapis.com/landing.html); FlyEM Janelia Hemibrain — Drosophila full connectome (https://neuprint.janelia.org); CREMI challenge — Drosophila larval neuromuscular junction EM (https://cremi.org); SNEMI3D — mouse cortex EM (https://snemi3d.grand-challenge.org/).

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

PyTorch Connectomics (https://github.com/zudi-lin/pytorch_connectomics) — modular GPU connectomics segmentation framework; DVID (https://github.com/janelia-flyem/dvid) — Janelia distributed EM data management; NeuTu (https://github.com/janelia-flyem/NeuTu) — proofreading and reconstruction visualization; VAST (verify URL — Harvard Lichtman lab large volume annotation tool).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN for 3D convolution in U-Net (dominant cost); NCCL for multi-GPU tensor-parallel training on large 3D crops; cuSPARSE for agglomeration graph operations; pattern: 3D sub-volume data parallelism across GPUs; sliding-window inference with overlap-tile strategy; mixed FP16/FP32 training. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
