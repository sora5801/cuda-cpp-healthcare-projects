# 4.21 — MR Fingerprinting Reconstruction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.21`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Magnetic Resonance Fingerprinting (MRF) measures tissue **T1 and T2 relaxation
times simultaneously** by playing a pseudorandom pulse sequence so every tissue
emits a distinctive **signal fingerprint** over time. Reconstruction is pattern
matching: for each image voxel, find the entry in a precomputed **dictionary** of
simulated fingerprints whose shape best matches the measured signal, and read off
its `(T1, T2)`. This project builds the dictionary, matches all voxels against it,
and verifies the GPU result against a CPU reference — on a tiny **synthetic**
sample that runs offline in milliseconds.

## What this computes & why the GPU helps

The compute bottleneck is the **match**: comparing every voxel's time course to
every dictionary atom is an inner product, and there are `V × D` of them, each of
length `T`. At clinical scale (`V ~ 10⁵` voxels, `D ~ 10⁵` atoms, `T ~ 10³`
frames) that is ~**10¹¹ inner products**. Crucially, the whole set of comparisons
is a single **matrix–matrix product** `S = Ŷ · F̂ᵀ` (normalized voxels × normalized
atomsᵀ). Dense GEMM is the GPU's home turf, so we hand the match to **cuBLAS
SGEMM** — one call replaces the CPU's `O(V·D·T)` triple loop — then pick each
voxel's best atom with a per-voxel `argmax` kernel. Building the dictionary and
normalizing the signals are embarrassingly parallel (one thread per atom / voxel).

## The algorithm in brief

- **Bloch-simulate** each `(T1, T2)` atom's length-`T` fingerprint (a compact
  RF-tip → T2-decay → T1-recovery recursion) and **L2-normalize** it.
- **L2-normalize** each voxel signal (removes the unknown proton-density scale, so
  matching is a pure cosine).
- **Match**: `S = Ŷ · F̂ᵀ` via **cuBLAS SGEMM** (the `V × D` cosine matrix), then
  **argmax** per voxel → the matched atom's `(T1, T2)`.
- **Verify** the GPU dictionary and per-voxel argmax **index** against the CPU
  reference; report reconstruction accuracy vs. the synthetic ground truth.

See [`THEORY.md`](THEORY.md) for the physics, the math, and the GPU mapping in
depth.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3** (the repo's
ratified toolchain; see [`docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md)).

1. Open [`build/mr-fingerprinting-reconstruction.sln`](build/mr-fingerprinting-reconstruction.sln)
   in Visual Studio 2026.
2. Select **`Release | x64`**.
3. **Build** (Ctrl+Shift+B). The linker pulls in `cublas.lib` and
   `cudart_static.lib` (see the commented `.vcxproj`); no manual path edits.

The executable lands in `build/x64/Release/mr-fingerprinting-reconstruction.exe`.
A cross-platform `CMakeLists.txt` is provided as a bonus (`CUDA::cublas`); the VS
solution is the required deliverable.

## Run the demo

One command builds (if needed), runs on the committed sample, and diffs the
deterministic stdout against `demo/expected_output.txt`:

```powershell
./demo/run_demo.ps1        # Windows (PowerShell)
```
```bash
./demo/run_demo.sh         # Linux/macOS (uses the optional CMake build)
```

## Data

The committed sample [`data/sample/mrf_sample.txt`](data/sample/mrf_sample.txt) is
**synthetic** (labeled as such everywhere): a `T=120`-frame schedule, a `D=22`-atom
well-separated `(T1, T2)` dictionary, and `V=64` voxels, each **drawn from a known
atom** (plus a random proton density and 0.5 % noise) so accuracy is measurable.
It is generated deterministically by
[`scripts/make_synthetic.py`](scripts/make_synthetic.py); provenance and the file
format are documented in [`data/README.md`](data/README.md). Real MRF datasets
(fastMRI, IEEE DataPort, qMRI.org) require registration and are **not**
redistributed — [`scripts/download_data.ps1`](scripts/download_data.ps1) /
[`.sh`](scripts/download_data.sh) print instructions and links without bypassing
any credential.

## Expected output

Deterministic result → **stdout** (diffed by the demo); timing + verification
detail → **stderr** (shown, not diffed). Success:

```
4.21 -- MR Fingerprinting Reconstruction
problem: T=120 frames, D=22 dictionary atoms, V=64 voxels (synthetic)
match accuracy: 64/64 voxels matched their ground-truth atom
median |T1 error| = 0.000 ms ; median |T2 error| = 0.000 ms
recovered T1 map range: [250.0, 2800.0] ms ; T2 map range: [25.0, 160.0] ms
first 5 voxels (voxel: truth_atom -> matched_atom  T1  T2  cos  PD):
  v00: truth=  3 -> atom=  3  T1= 250.0  T2= 100.0  cos=0.9996  PD=1.7314
  ...
RESULT: PASS (GPU dictionary + argmax match CPU; indices exact)
```

**How the GPU result is checked against the CPU:** the GPU-built dictionary must
match the CPU's within `1e-4` (observed `0`), the per-voxel best-atom **index**
must match **exactly**, and the cosine scores must agree within `1e-4` (observed
~`2e-7`). The exact-index match is safe because the dictionary is separated far
beyond the SGEMM's floating-point wobble — see [`THEORY.md`](THEORY.md)
§"Numerical considerations".

## Code tour

Start in [`src/main.cu`](src/main.cu) (load → CPU → GPU → verify → report), then:

1. [`src/mrf_core.h`](src/mrf_core.h) — the shared `__host__ __device__` math:
   the Bloch step, atom simulation, normalization, inner product. **Read this
   first for the physics**; both CPU and GPU call it, so results match.
2. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface and the pattern each
   stage uses.
3. [`src/kernels.cu`](src/kernels.cu) — the four device stages: build dictionary,
   normalize signals, **cuBLAS SGEMM** (with the row-major↔column-major layout
   explained at the call site), per-voxel argmax.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the serial baseline (and the
   sample-file parser) the GPU is verified against.

## Prior art & further reading

- **BART** — <https://github.com/mrirecon/bart> — reference low-rank *subspace*
  MRF reconstruction; study how it compresses the temporal dimension before
  matching.
- **SigPy** — <https://github.com/mikgroup/sigpy> — NUFFT-based MRF recon for
  non-Cartesian (spiral) trajectories; the piece this teaching version omits.
- **MRzero** — differentiable MR sequence simulation; learn how the *sequence*
  itself can be optimized for parameter separability (Cramér–Rao bound).
- **PyTorch MRF dictionary matching** (search GitHub) — the same GEMM match in a
  deep-learning stack; good for comparing the linear-algebra pattern.
- Ma et al., "Magnetic resonance fingerprinting," *Nature* 495 (2013) — the
  founding paper.

## Exercises

1. **Bigger dictionary.** Loosen `--sep_thresh` toward `0.9999` (or add T1/T2 grid
   points) in `make_synthetic.py`. Watch reconstruction accuracy drop and the
   GPU-vs-CPU index mismatches appear as atoms become near-collinear — the
   ill-conditioning that motivates SVD compression. Why does it happen?
2. **Constant memory.** Move the schedule (`α, TR, TE`) into `__constant__` memory
   in `kernels.cu` and measure the change on the dictionary-build kernel.
3. **Fused argmax.** cuBLAS gives the whole score matrix; write a fused kernel that
   computes each voxel's row *and* its argmax without materializing `S`, and
   compare memory traffic.
4. **Batched matching.** Split the voxels into batches and call `cublasSgemm` per
   batch (the real-world tactic when `S` doesn't fit in memory). Does the answer
   change? (It shouldn't.)
5. **Noise sweep.** Raise `--noise` and plot accuracy vs. noise for this dictionary
   — an empirical look at MRF's noise robustness.

## Limitations & honesty

- **Synthetic data**, labeled as such; no scanner or patient data; **no clinical
  claims**.
- **Reduced-scope forward model.** A closed-form real-valued recursion, not a full
  complex **Bloch/EPG** simulation — no off-resonance, `B1+`, slice profile, or
  spoiling. It captures qualitative T1/T2 sensitivity only.
- **Cartesian, single-pass.** Real MRF uses spiral k-space and a **NUFFT**
  (`cuFFT`) step, often inside an iterative low-rank/ADMM reconstruction. We
  implement only the **dictionary-match GEMM**, which *is* the production compute
  pattern the catalog names.
- **Tiny dictionary** (22 atoms) chosen for separability and clean verification; a
  real dictionary is `10⁵–10⁶` atoms and genuinely ill-conditioned.
- **Timing is a teaching artifact, not a benchmark** — the sample is too small for
  the GPU to win; the SGEMM's edge grows as `O(V·D·T)`.
