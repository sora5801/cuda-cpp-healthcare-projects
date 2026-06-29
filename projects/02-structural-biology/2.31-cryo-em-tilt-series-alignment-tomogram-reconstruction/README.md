# 2.31 — Cryo-EM Tilt-Series Alignment & Tomogram Reconstruction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.31`
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

Cryo-ET tilt-series reconstruction requires (1) frame alignment (beam-induced motion), (2) tilt-series alignment (fiducial or fiducial-free), and (3) tomogram reconstruction (weighted back-projection or iterative SART/ASTRA). All three steps are GPU-parallelizable: GPU-accelerated SART iterates over projection angles simultaneously; WBP uses GPU FFT and filter application. IMOD, AreTomo, and etomo handle tilt-series alignment; the ASTRA Toolbox provides GPU iterative reconstruction via CUDA. Cryo-ET remains limited by the missing wedge artifact, which deep learning (IsoNet) corrects post hoc on GPU.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Weighted back-projection (WBP), SART (simultaneous algebraic reconstruction), AreTomo beam-induced motion correction, fiducial marker alignment, beam-induced motion correction (MotionCor2-TomoTilt), iterative reconstruction convergence.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/cryo-em-tilt-series-alignment-tomogram-reconstruction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/cryo-em-tilt-series-alignment-tomogram-reconstruction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\cryo-em-tilt-series-alignment-tomogram-reconstruction.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: EMPIAR tilt series archives (https://www.ebi.ac.uk/empiar/); EMDB subtomogram averages (https://www.ebi.ac.uk/emdb/); SHREC cryo-ET benchmark (verify URL); in situ ribosome tilt series (EMPIAR-10045).

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

IMOD (https://bio3d.colorado.edu/imod/) — standard tomographic reconstruction suite; ASTRA Toolbox (https://github.com/astra-toolbox/astra-toolbox) — GPU CUDA reconstruction algorithms; AreTomo2 (https://github.com/czimaginginstitute/AreTomo2) — GPU tilt-series alignment; IsoNet (https://github.com/IsoNet-cryoET/IsoNet) — GPU deep learning missing wedge correction.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA WBP kernel over tilt projection angles; cuFFT for filter application in filtered back-projection; GPU SART iteration with CUDA atomic updates; PyTorch CNN for IsoNet missing-wedge correction; multi-GPU for large tomogram reconstruction. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
