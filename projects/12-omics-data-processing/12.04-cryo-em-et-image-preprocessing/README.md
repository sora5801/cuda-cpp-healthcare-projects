# 12.4 — Cryo-EM / ET Image Preprocessing

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Analytical%20%26%20Omics%20Data%20Processing-lightgrey)

> **🟢 Beginner · Established** — Domain 12: Analytical & Omics Data Processing · Catalog ID `12.4`
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

Cryo-electron microscopy produces thousands of noisy micrographs (4k×4k pixels) that must be motion-corrected, CTF-estimated, particle-picked, and 2D/3D classified before structure determination. CryoSPARC and RELION both natively use CUDA for all major processing steps: motion correction via cross-correlation in Fourier space (cuFFT), CTF estimation via Thon ring fitting on GPU, particle picking via neural network (Topaz, crYOLO), and 3D refinement via GPU-accelerated back-projection and real-space expectation-maximisation. A single H100 processes hundreds of micrographs per minute end-to-end, enabling real-time feedback during cryo-EM sessions.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Fourier-space cross-correlation for frame alignment (MotionCor2); CTF fitting via Thon ring power spectrum (CTFFIND); 2D class averaging (RELION E-M); 3D gold-standard FSC refinement; CNN particle picking (Topaz); back-projection 3D reconstruction; Wiener filter CTF correction.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/cryo-em-et-image-preprocessing.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/cryo-em-et-image-preprocessing.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\cryo-em-et-image-preprocessing.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: EMDB — Electron Microscopy Data Bank, raw micrographs and maps (https://www.ebi.ac.uk/emdb/); EMPIAR — raw cryo-EM micrograph repository (https://www.ebi.ac.uk/empiar/); wwPDB cryo-EM entries (https://www.rcsb.org/); CryoSPARC demo datasets (https://cryosparc.com/download).

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

CryoSPARC (https://cryosparc.com/) — fully GPU-native cryo-EM pipeline, particle picking through 3D refinement; RELION4 (https://github.com/3dem/relion) — GPU-accelerated 3D classification and refinement; Topaz (https://github.com/tbepler/topaz) — GPU CNN particle picker; MotionCor2 (verify URL — Zheng lab UCSF) — GPU frame alignment.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuFFT for Fourier-domain frame alignment and CTF power spectrum; cuDNN for CNN particle picking; custom back-projection CUDA kernels; atomic operations for back-projection accumulation; multi-GPU 3D refinement with gradient averaging. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
