# 1.30 — Trajectory RMSD, Clustering & Contact Analysis

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.30`
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

Post-MD analysis of multi-microsecond trajectories generates terabytes of coordinate data requiring GPU-accelerated analytics. RMSD calculation requires aligning every frame to a reference (Kabsch algorithm: SVD of 3×3 matrices — trivially parallelized over frames). Pairwise RMSD for clustering requires O(N²) comparisons of millions of frames. H-bond network analysis and contact map generation are similarly parallelizable. MDTraj and cuML enable GPU-accelerated trajectory analysis with RAPIDS. The bottleneck is I/O bandwidth from trajectory files.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Kabsch RMSD algorithm (SVD), GROMOS/DBSCAN/k-medoids clustering, contact map calculation (distance cutoff), H-bond donor-acceptor angle+distance criteria, radial distribution function (RDF), NMR order parameter S².

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/trajectory-rmsd-clustering-contact-analysis.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/trajectory-rmsd-clustering-contact-analysis.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\trajectory-rmsd-clustering-contact-analysis.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: MDCATH trajectory dataset (https://huggingface.co/datasets/compsciencelab/mdcath); PDB trajectory depositions; GPCRmd (https://gpcrmd.org); MDDB (https://www.mddbr.eu) — molecular dynamics database.

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

MDTraj (https://github.com/mdtraj/mdtraj) — GPU-accelerated RMSD and trajectory analysis; RAPIDS cuML (https://github.com/rapidsai/cuml) — GPU clustering for MSM construction; MDAnalysis (https://github.com/MDAnalysis/mdanalysis) — trajectory analysis with GPU support; HTMD (https://github.com/Acellera/htmd) — GPU-accelerated adaptive MD analysis.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA kernels for batched 3×3 SVD (Kabsch rotation); GPU pairwise distance matrix via cuBLAS (outer product formulation); atomic contact map via GPU distance thresholding; RAPIDS cuDF for trajectory frame I/O. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
