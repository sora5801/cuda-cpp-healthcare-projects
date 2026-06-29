# 2.20 — Heterogeneous Cryo-EM Reconstruction (3D Variability)

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.20`
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

Real protein complexes adopt multiple conformational states simultaneously. Heterogeneous reconstruction methods disentangle these states from particle images. CryoDRGN uses a variational autoencoder (VAE) with an amortized encoder that maps each particle image to a latent code representing its conformation, and a decoder that generates the 3D density from the latent code via a coordinate MLP. GPU training is essential: a cryoDRGN run on 100k particles requires hours on A100. 3DVA (cryoSPARC) uses PCA-like linear subspace methods. Applications reveal continuous flexibility in ribosomes, GPCR complexes, and viral assembly intermediates.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Variational autoencoder (VAE) with image encoder and volume decoder, coordinate-based implicit neural representation (NeRF/MLP decoder), 3D variability analysis (PCA on volume subspace), pose estimation EM, Fourier-slice theorem in the network, manifold learning.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/heterogeneous-cryo-em-reconstruction-3d-variability.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/heterogeneous-cryo-em-reconstruction-3d-variability.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\heterogeneous-cryo-em-reconstruction-3d-variability.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: EMPIAR-10180 (spliceosome), EMPIAR-10076 (80S ribosome), EMPIAR-10028 (TRPV1) (all at https://www.ebi.ac.uk/empiar/); cryoDRGN benchmark datasets (https://github.com/ml-struct-bio/cryodrgn); simulated heterogeneous datasets from IgG/spike protein.

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

CryoDRGN (https://github.com/ml-struct-bio/cryodrgn) — GPU VAE for heterogeneous reconstruction; cryoSPARC 3DVA (https://cryosparc.com) — GPU linear 3D variability analysis; Recovar (verify URL) — GPU regularized covariance heterogeneous reconstruction; DrgnAI (verify URL) — neural 3D reconstruction with pose optimization.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

PyTorch CUDA for VAE encoder/decoder; FlashAttention for particle image attention layers; GPU Fourier-slice theorem evaluation via differentiable nufft; cuFFT for power spectrum during training; FP16 mixed precision. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
