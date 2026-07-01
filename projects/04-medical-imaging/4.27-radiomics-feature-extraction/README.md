# 4.27 — Radiomics Feature Extraction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.27`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

**Radiomics** turns a segmented region of a medical image — say a lung tumour
outlined on a CT scan — into a vector of quantitative numbers ("features") that a
model can correlate with survival, treatment response, or genotype. This project
computes the two workhorse feature families on the GPU: **first-order statistics**
(the ROI's gray-level histogram → mean, variance, energy, entropy) and **texture
features** from the **Gray-Level Co-occurrence Matrix (GLCM)** (contrast, energy,
homogeneity, correlation, entropy). The GLCM is a spatial histogram: for every
voxel and each of the 13 symmetric 3-D neighbour directions, we count how often
gray level *i* sits next to gray level *j*. That counting is embarrassingly
parallel — one GPU thread per voxel, `atomicAdd` into the matrix — which is why
GPU radiomics runs ~100× faster than CPU pipelines on real ROIs.

## What this computes & why the GPU helps

Radiomics extracts hundreds of quantitative features (shape, first-order statistics, texture: GLCM, GLRLM, GLSZM, NGTDM) from 3D segmented ROIs in CT/PET/MRI. For a cohort of 10,000 patients with large ROIs (~10⁶ voxels each), CPU-based PyRadiomics takes 10–30 min per patient; GPU-accelerated cuRadiomics and PyRadiomics-CUDA achieve 143× speedup by parallelizing all histogram and co-occurrence matrix computations across voxels on GPU. Texture features require computing co-occurrence matrices from 26 3D neighbor directions simultaneously — each direction's computation is independent, enabling massive GPU parallelism. Radiomics biomarker discovery pipelines must process thousands of scans for statistical power.

**The parallel bottleneck:** building the **GLCM** dominates. It is a scatter over
`nroi × 13` voxel–neighbour pairs, each incrementing a matrix cell. On the CPU
that is a serial triple loop; on the GPU every voxel is independent, so we assign
**one thread per voxel** and let them `atomicAdd` into a **block-private
shared-memory GLCM** (privatized histogram), flushed once to global memory. The
accumulators are **integers**, so the atomic adds commute → the GPU counts are
deterministic *and* bit-identical to the CPU (docs/PATTERNS.md §2–3).

## The algorithm in brief

Catalog key algorithms: GLCM (gray-level co-occurrence matrix), GLRLM (run-length matrix), GLSZM (size-zone matrix), NGTDM (neighborhood gray-tone difference matrix), first-order statistics, 3D shape descriptors, wavelet-decomposition features, multi-scale radiomics, IBSI (Image Biomarker Standardization Initiative) compliant features.

This teaching version implements the **first-order** and **GLCM texture** families
(the most-used pair; the others are described in THEORY.md "Where this sits in the
real world"):

- **Quantize** ROI intensities into `Ng` gray levels (fixed-bin-count discretization).
- **Histogram** the levels → first-order features (mean, variance, energy, entropy).
- **GLCM**: for each ROI voxel and the 13 symmetric 3-D directions, count the
  co-occurring level pair (symmetrized), summed over directions.
- **Normalize** the GLCM to a probability matrix and read off Haralick scalars:
  contrast, energy (ASM), homogeneity, correlation, entropy.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/radiomics-feature-extraction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/radiomics-feature-extraction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\radiomics-feature-extraction.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/radiomics_sample.txt`, prints the
feature vector, shows the GPU-vs-CPU agreement check (GLCM counts identical), and
prints a timing line to stderr.

## Data

- **Sample (committed):** `data/sample/radiomics_sample.txt` — a tiny **synthetic**
  `6×6×5` ROI (56 masked voxels) so the demo runs with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (documented, idempotent;
  prints TCIA/GDC pointers, never bypasses a Data Use Agreement).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: TCIA NSCLC-Radiomics (https://www.cancerimagingarchive.net/collection/nsclc-radiomics/) — 422 lung CTs with survival; RIDER Breast MRI (via TCIA); QIN-HEADNECK (via TCIA) — head and neck RT; TCGA collections (https://portal.gdc.cancer.gov/).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program computes the features on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and asserts they agree: the integer GLCM and
histogram counts match **exactly** (integer atomics commute), and every derived
feature within `1e-9`. That agreement is the correctness guarantee. The `820`
GLCM pairs and the negative `correlation` are the signature of the sample's
built-in checkerboard texture (see `demo/README.md`).

## Code tour

Read in this order:

1. [`src/radiomics.h`](src/radiomics.h) — the shared `__host__ __device__` core:
   gray-level quantization, the 13 direction offsets, flat indexing. This is the
   single source of per-voxel math used by *both* CPU and GPU.
2. [`src/main.cu`](src/main.cu) — loads the ROI, runs CPU + GPU, verifies, reports.
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`reference_cpu.cpp`](src/reference_cpu.cpp)
   — the `Volume`/`Features` types, the serial GLCM builder (the trusted
   baseline), and the count→feature reductions shared with the GPU.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the parallel-histogram idea.
5. [`src/kernels.cu`](src/kernels.cu) — the atomic GLCM/histogram kernels + host wrapper.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

PyRadiomics-CUDA (https://arxiv.org/abs/2510.02894 — code on https://github.com/mis-wut/pyradiomics-CUDA) — GPU radiomics, 143× speedup; cuRadiomics (published in AAPM proceedings) — CUDA texture/GLCM GPU extraction; PyRadiomics CPU baseline (https://github.com/AIM-Harvard/pyradiomics) — IBSI-compliant reference; MONAI (https://github.com/Project-MONAI/MONAI) — integrated GPU radiomics pipeline.

- **PyRadiomics (CPU)** — the IBSI-compliant reference: study its GLCM/first-order
  definitions and the *fixed-bin-width vs. fixed-bin-count* discretization choice.
- **PyRadiomics-CUDA / cuRadiomics** — how the histogram/GLCM scatter is mapped to
  the GPU at scale; where the 143× comes from.
- **MONAI** — how radiomics slots into an end-to-end imaging-AI pipeline.
- **IBSI** ([theibsi.github.io](https://theibsi.github.io/)) — the standard that
  pins down exact feature formulas so different tools agree.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Catalog GPU pattern: Custom CUDA for co-occurrence matrix (atomic add into per-direction GLCM per thread block); shared memory for voxel neighborhood; parallel histogram across all voxels; warp-level reductions for matrix statistics.

Concretely: **parallel histogram / atomic co-occurrence scatter** (docs/PATTERNS.md
§1, exemplar 11.09). One thread per voxel; a **block-private GLCM in shared
memory** absorbs the intra-block atomics cheaply; one flush merges each block's
copy into the global matrix. **Integer** accumulators keep the reduction
deterministic and exactly CPU-matching. The 13 direction offsets live in
**constant memory** (broadcast to every thread).

## Exercises

1. **Change the texture.** Edit `scripts/make_synthetic.py` to drop the
   checkerboard `ripple` (leave only the smooth gradient). Rebuild the sample and
   re-run: watch `contrast` fall and `homogeneity`/`correlation` rise. Regenerate
   `expected_output.txt` from the real run.
2. **Add a feature.** Implement GLCM *dissimilarity* (`sum P(i,j)|i−j|`) or
   *cluster shade* in the shared `haralick_from_glcm` so CPU and GPU stay in lock-step.
3. **Per-direction GLCMs.** Right now the 13 directions are summed. Keep them
   separate (13 matrices) and report a feature per direction plus its mean/range —
   the "directional" radiomics variant. (Grid a second dimension over directions.)
4. **Scale up.** Generate a `128³` ROI (`--nx 128 --ny 128 --nz 128`) and compare
   CPU vs. GPU timing — where does the GPU start to win? (Watch the launch-overhead
   crossover; see THEORY §numerics/timing.)
5. **Fixed-bin-width discretization.** Swap `rad_quantize` for IBSI fixed-bin-width
   (`level = floor((v − vmin)/binwidth)`); note how the feature values shift and
   why standardization (IBSI) matters for cross-study comparability.

## Limitations & honesty

- **Reduced scope.** We implement first-order + GLCM texture. GLRLM, GLSZM, NGTDM,
  3-D shape descriptors, and wavelet/multi-scale features (in the catalog) are
  described in THEORY.md but not coded — they follow the same scatter/histogram
  pattern.
- **Synthetic data.** The committed sample is a hand-designed synthetic ROI,
  labeled synthetic everywhere. It is not a real scan and has no clinical meaning.
- **Simplified I/O.** Real radiomics reads DICOM/NIfTI with world-coordinate
  spacing; we read a plain dense text grid + mask. Voxel anisotropy, resampling,
  and intensity-outlier clipping (all IBSI concerns) are omitted.
- **Not IBSI-certified.** The formulas follow the standard definitions but this is
  a teaching implementation, not a validated, IBSI-benchmarked tool. Do not use
  any output for diagnosis or treatment (CLAUDE.md §8).
