# 3.15 — Hi-C / 3D Genome Contact Analysis

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.15`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A **Hi-C** experiment measures, for every pair of genomic bins, how often the two
loci were found physically touching inside the cell nucleus — yielding a large,
sparse, symmetric **contact matrix**. This project does the two foundational steps
of Hi-C analysis on the GPU: (1) **ICE balancing** — iteratively removing per-bin
sequencing/visibility bias so every bin is "equally visible", and (2) **TAD
calling** — computing the **insulation score** along the diagonal of the balanced
matrix and reporting **Topologically Associating Domain (TAD) boundaries** as its
local minima. The headline GPU lesson is a **deterministic sparse reduction**: one
thread per nonzero, scattering bias-corrected contributions into per-bin row sums
with **fixed-point integer atomics**, so the GPU result matches the CPU reference
**exactly**.

## What this computes & why the GPU helps

Hi-C maps chromatin contacts genome-wide, producing sparse contact matrices of size
(genome_bins × genome_bins) at 1–10 kb resolution. Downstream analysis — matrix
normalisation (ICE/KR balancing), TAD boundary calling, compartment A/B
classification, and loop detection — involves iterative matrix operations on
matrices with up to 3×10⁶ bins (3 Gb of data at 1 kb). GPU acceleration of the ICE
iterative-correction algorithm (repeated sparse matrix-vector products) and the
convolution-based loop callers is particularly impactful.

**The parallel bottleneck:** the ICE hot loop computes, every iteration, the **row
sum of the bias-corrected matrix at every bin** — a reduction over *all* nonzeros,
repeated ~20–50 times. With up to ~10⁹ nonzeros that is the dominant cost. Each
nonzero's contribution is independent, so we give **one GPU thread per nonzero** and
let it `atomicAdd` into its two endpoint row sums. This is a textbook
**parallel-scatter + atomic-reduce** (the same pattern as flagship `11.09` k-means).

## The algorithm in brief

- **ICE / matrix balancing (Imakaev 2012):** find a per-bin **bias** `b` so that the
  corrected matrix `M'_{ij} = M_{ij} / (b_i b_j)` has equal row sums. Fixed-point
  iteration: compute row sums → `b_k *= rowsum_k / mean` → repeat. (This is the
  Sinkhorn–Knopp idea applied to a symmetric matrix.)
- **Insulation score (Crane 2015):** for each bin, the mean balanced contact in a
  small **diamond window** straddling the diagonal — low where a domain border
  blocks cross-contacts.
- **TAD boundaries:** the **local minima** of the insulation score.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation (including the eigenvector A/B-compartment and HiCCUPS loop steps that we
*describe* but do not implement here).

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/hi-c-3d-genome-contact-analysis.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/hi-c-3d-genome-contact-analysis.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\hi-c-3d-genome-contact-analysis.sln /p:Configuration=Release /p:Platform=x64
```

This project links only the CUDA runtime (`cudart_static.lib`) — the sparse
matrix-vector reduction is **hand-rolled** so nothing is a black box. THEORY.md
shows the equivalent cuSPARSE SpMV a production tool would use.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/hic_sample.txt`, prints the balanced
biases, the insulation score, and the called TAD boundaries, shows the GPU-vs-CPU
agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/hic_sample.txt` — a tiny **synthetic** 12-bin
  Hi-C matrix with three known TADs (borders at bins 4 and 8) and a known per-bin
  coverage bias, so the demo's answer is verifiable. Runs offline, zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print the vetted steps to
  fetch a real `.mcool`/`.hic` map and convert it to this project's `i j count` text.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: 4DN Data Portal (https://data.4dnucleome.org/); ENCODE
(https://www.encodeproject.org/); GEO GSE63525, Rao et al. 2014
(https://www.ncbi.nlm.nih.gov/geo/); OpenChromatin Consortium ATAC/Hi-C.

## Expected output

Success looks like `demo/expected_output.txt`. The program runs ICE on both the
**GPU** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and
asserts the per-bin biases agree within `1e-9` (in fact they agree **exactly** —
the fixed-point reduction makes both sides sum identical integers). The downstream
insulation score then recovers the planted TAD boundaries at **bins 4 and 8** — an
end-to-end correctness check on the science, not just CPU == GPU agreement.

## Code tour

Read in this order:

1. [`src/hic.h`](src/hic.h) — the shared `__host__ __device__` core: the corrected-
   contact formula and the fixed-point quantization both CPU and GPU use.
2. [`src/main.cu`](src/main.cu) — loads the matrix, runs CPU + GPU ICE, verifies,
   computes insulation + boundaries, reports.
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`src/reference_cpu.cpp`](src/reference_cpu.cpp)
   — the trusted serial baseline and the shared host helpers (bias update,
   insulation, boundary calling).
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
5. [`src/kernels.cu`](src/kernels.cu) — the `ice_rowsum_kernel` and host driver.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **cooler** (https://github.com/open2c/cooler) — the standard `.cool`/`.mcool`
  sparse Hi-C container and I/O. Study its CSR-like storage; we mimic its COO layout.
- **cooltools** (https://github.com/open2c/cooltools) — reference CPU implementations
  of ICE balancing, insulation score, and saddle/compartment analysis. The clearest
  source for the exact insulation-diamond definition.
- **Juicer / HiCCUPS** (https://github.com/aidenlab/juicer) — GPU-accelerated loop
  caller; see how the 2D donut convolution maps to CUDA.
- **Higashi** (https://github.com/ma-compbio/Higashi) — single-cell Hi-C GPU
  hypergraph model.
- **ChromaFold** (https://www.nature.com/articles/s41467-024-53628-0) — GPU CNN that
  predicts contact maps from 1D accessibility.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Parallel scatter + deterministic atomic reduce** over a sparse COO matrix: one
thread per nonzero, `atomicAdd` of **fixed-point integers** into per-bin row-sum
accumulators (integer adds commute → order-independent → bit-identical to the CPU).
The cheap O(n) bias update stays on the host via a **shared helper** the CPU
reference also calls, so the two paths cannot diverge. A production tool would
express the same row sum as a **cuSPARSE** sparse matrix-vector product
(`rowsum = M' · 1`); we hand-roll it so the reduction is fully visible.

## Exercises

1. **KR balancing.** ICE is one of two classic balancers. Implement the
   Knight–Ruiz (KR) algorithm (a faster Newton-style balancer) and compare its bias
   vector and iteration count against ICE on the sample.
2. **Real resolution sweep.** Use `cooler dump` (see `scripts/download_data.*`) to
   export a real chromosome at 100 kb and 25 kb; watch how many TAD boundaries the
   insulation caller finds at each resolution.
3. **A/B compartments.** Add the eigenvector step: form the observed/expected
   correlation matrix and take its first eigenvector (sign = compartment A vs B).
   Use **cuSOLVER `Dsyevd`** (see flagship `2.06`) for the eigensolve.
4. **Shared-memory binning.** The atomic reduction collides heavily on hub bins.
   Try per-block partial sums in shared memory, then one atomic per block — measure
   the reduction in global-atomic traffic.
5. **Tolerance experiment.** Replace the fixed-point atomics with a *float* atomic
   and observe the run-to-run nondeterminism (and CPU/GPU mismatch) it reintroduces.

## Limitations & honesty

- **Synthetic, tiny data.** The sample is a 12-bin toy with planted structure,
  labelled synthetic everywhere. It exists to make the answer checkable, not to
  model a real genome. **No output here is clinically or biologically valid.**
- **Reduced scope.** We implement ICE balancing + insulation TAD calling — the parts
  that best teach the GPU sparse-reduction pattern. A/B compartments
  (eigendecomposition), HiCCUPS loop calling (2D convolution), and CNN contact
  prediction are *described* in THEORY.md but not implemented (each is its own
  project-sized effort, noted under "Where this sits in the real world").
- **Host bias update.** We keep the O(n) bias update on the CPU each iteration for
  exact CPU/GPU parity; a fully GPU-resident pipeline would fuse it on-device.
- **Balancing target.** Our ICE normalises row sums to their mean (a simple,
  readable variant); production ICE often targets a fixed total and applies a mask
  of low-coverage bins from a coverage histogram — see THEORY.md.
