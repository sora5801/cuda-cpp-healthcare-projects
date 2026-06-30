# 2.2 — Protein-Protein Docking

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟢 Beginner · Established** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.2`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

When two proteins meet, their shapes have to fit — a knob on one slots into a
pocket on the other. **Rigid-body docking** searches for the relative placement
(translation + rotation) of two proteins that maximizes this geometric
complementarity. This project implements the classical **FFT-correlation**
docking method (the engine inside ZDOCK and ClusPro): voxelize both proteins onto
a 3D grid, then score *every* relative translation at once with a single Fast
Fourier Transform. The slow, obvious way to score all translations is an
`O(Ng²)` brute-force correlation; the FFT does it in `O(Ng log Ng)`. We run the
brute-force version on the CPU as a trusted reference and the FFT version on the
GPU with **cuFFT**, and check they agree.

## What this computes & why the GPU helps

Predicting protein-protein complex structures is critical for understanding
signaling pathways, antibody-antigen recognition, and designing PPI inhibitors.
Classical docking (ClusPro, ZDOCK) uses an FFT-based rigid-body search over the
translational degrees of freedom: for a fixed ligand orientation, the
shape-complementarity score of *all* translations `t` is the cross-correlation
`S(t) = Σ_x R(x)·L(x − t)` of the two proteins' shape grids. By the
**correlation theorem** that whole grid of scores is `IFFT(FFT(R)·conj(FFT(L)))`.

**The parallel bottleneck:** evaluating `S(t)` directly costs `O(Ng²)` — for a
real `~100³ ≈ 10⁶`-voxel grid that is `~10¹²` multiply-adds *per orientation*,
times thousands of orientations. The FFT route collapses each translational
search to three transforms (`O(Ng log Ng)`) plus a pointwise multiply. cuFFT runs
those 3D transforms on the GPU; the pointwise spectrum-multiply and rescale are
two tiny custom kernels. On the committed sample the GPU FFT path is ~3000×
faster than the brute-force CPU correlation.

## The algorithm in brief

- **Voxelize** receptor and ligand onto a shared `N×N×N` grid as a two-value
  *shape function*: `+1` in the buried interior ("core"), a large negative
  *penalty* in a one-voxel surface "skin" (a steric-clash deterrent), `0` in
  empty space (Katchalski-Katzir / ZDOCK grid model).
- **Correlate** the two grids over all translations. Direct `O(Ng²)` sum on the
  CPU; FFT `O(Ng log Ng)` on the GPU (`FFT(R)·conj(FFT(L))`, inverse-FFT, rescale).
- **Argmax** the score grid → the best translation = the predicted dock.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation (including why conjugating the *ligand* spectrum gives the right shift
sign, and the FP round-off analysis).

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)). This project links
**cuFFT** (already wired into the `.vcxproj` and `CMakeLists.txt`).

1. Open `build/protein-protein-docking.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/protein-protein-docking.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\protein-protein-docking.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/dock_sample.txt`, prints the
recovered docking translation + score, shows the GPU-vs-CPU agreement check, and
prints a timing line on stderr.

## Data

- **Sample (committed):** `data/sample/dock_sample.txt` — a tiny **synthetic**
  docking pair with a *known* best translation, so the demo runs offline and its
  answer is verifiable.
- **Full datasets:** `scripts/download_data.ps1` / `.sh` print the real
  benchmarks (Docking Benchmark 5.5, SAbDab, PDB) and how to format a complex.
- **Provenance & license:** see [data/README.md](data/README.md). The sample is
  labeled synthetic everywhere.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
best translation (voxels): t = (-3, -2, 1)
known-answer translation:  t = (-3, -2, 1)  -> RECOVERED
RESULT: PASS (cuFFT score grid matches CPU within 5e-01; best pose identical)
```

The program computes the correlation on both the **GPU** (cuFFT, `src/kernels.cu`)
and a **CPU reference** (brute force, `src/reference_cpu.cpp`) and asserts (a) the
two score grids agree within a documented round-off tolerance and (b) the single
best-scoring translation is identical. The ligand is a copy of the receptor
displaced by `D=(3,2,−1)` voxels, so recovering `t = −D = (−3,−2,1)` confirms the
search found the true re-registration.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the two proteins, voxelizes, runs CPU +
   GPU correlation, verifies grid + pose, reports.
2. [`src/reference_cpu.h`](src/reference_cpu.h) — the data model, the shared
   `__host__ __device__` indexing helpers, and the full project overview.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the correlation
   theorem idea (why cuFFT).
4. [`src/kernels.cu`](src/kernels.cu) — the cuFFT 3D R2C/C2R calls (documented,
   not a black box) + the two tiny custom kernels.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial
   voxelization + brute-force correlation.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, I/O helpers.

## Prior art & further reading

- **ClusPro** (<https://cluspro.bu.edu>) — the canonical FFT docking server; its
  back end (PIPER) is exactly this correlation idea with richer scoring terms.
- **ZDOCK** — pairwise shape/electrostatics/desolvation FFT docking; study its
  grid scoring (the core/skin model here is a simplified ZDOCK grid).
- **DiffDock-PP** (<https://github.com/ketatam/DiffDock-PP>) — the modern deep
  learning take: an equivariant *diffusion* model over poses instead of an
  exhaustive grid search. Read it to see what replaced the FFT in DL pipelines.
- **HADDOCK** (<https://wenmr.science.uu.nl/haddock2.4/>) — data-driven docking
  with GPU MD refinement (how poses get relaxed after the rigid search).
- **RoseTTAFold / RoseTTAFold2NA** (<https://github.com/RosettaCommons/RoseTTAFold>)
  — co-folding of complexes from sequence/MSA, no explicit docking search.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Use a CUDA library without a black box** (PATTERNS.md §1, exemplified by 8.03):
the FFT is a solved problem, so we call **cuFFT** for the 3D real-to-complex and
complex-to-real transforms — but `kernels.cu` documents exactly what each call
computes, the Hermitian-half data layout it expects, and what hand-rolling would
take. The only custom device code is a one-line spectrum multiply (the
correlation theorem's pointwise step) and a one-line rescale.

## Exercises

1. **Bigger grid:** regenerate the sample at `--N 64` (and rebuild). Watch the
   CPU brute-force time grow `~8×` (it's `O(Ng²)`, `Ng = N³`) while the cuFFT
   time barely moves — the whole point of the FFT route.
2. **Add a rotation loop:** real docking searches *orientations* too. Wrap the
   correlation in a loop over a handful of ligand rotations (rotate the atoms
   before voxelizing) and keep the global best `(rotation, translation)`.
3. **Batch the FFTs:** cuFFT can transform many grids in one call
   (`cufftPlanMany`). Batch several rotated ligands' forward FFTs together and
   compare the throughput.
4. **Electrostatics term:** add a second correlation of a charge grid and combine
   it with the shape score (a weighted sum) — the start of a ZDOCK-style score.
5. **Top-K poses:** instead of a single argmax, extract the K highest-scoring,
   spatially-separated translations (real pipelines cluster many candidate poses).

## Limitations & honesty

- **Single orientation only.** This teaching version searches *translations* for
  one fixed ligand orientation. Production docking also scans thousands of
  **rotations** — the FFT correlation here is the inner loop of that larger search
  (THEORY "Where this sits in the real world").
- **Shape only.** The score is pure geometric complementarity (core/skin). Real
  scoring adds electrostatics, desolvation, statistical potentials, and pose
  clustering/refinement.
- **Synthetic, self-docking sample.** To get a *known, verifiable* answer the
  ligand is a displaced copy of the receptor (a real task — homodimer/symmetry
  search — but not a blind hetero-complex prediction). It is labeled synthetic
  everywhere and carries no biological or clinical meaning.
- **Not for any clinical or experimental-design use.** Study material only.
