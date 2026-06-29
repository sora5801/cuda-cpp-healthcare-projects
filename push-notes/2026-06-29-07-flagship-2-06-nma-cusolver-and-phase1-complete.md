# Push 2026-06-29 #07 -- flagship 2.06 nma-cusolver and PHASE 1 COMPLETE

> Push-note (CLAUDE.md §7.1). The fourteenth and final Phase 1 flagship — structural biology — and the
> close-out of Phase 1: **one polished, fully-verified project in every one of the 14 domains.**

## 1. Summary

The structural-biology flagship is done: **2.06 Normal Mode Analysis / Elastic Network Models**, a complete,
verified GPU NMA built around **cuSOLVER** (the dense symmetric eigensolver) — a pattern distinct from every
other flagship's hand-written kernels. It builds the 3N×3N ANM Hessian, diagonalizes it on the GPU, finds
exactly the 6 rigid-body modes, and predicts per-residue mobility. **This completes Phase 1: 14/14 flagships,
one per domain, all 14/301 projects DONE.**

## 2. What changed

- [`projects/02-structural-biology/2.06-normal-mode-analysis-elastic-network-models/`](../projects/02-structural-biology/2.06-normal-mode-analysis-elastic-network-models) — fully implemented:
  - `src/kernels.cu` — **cuSOLVER `Dsyevd`** eigendecomposition wrapper (the library call, documented).
  - `src/reference_cpu.cpp` / `.h` — ANM Hessian build + a transparent **Jacobi** eigensolver + mobility.
  - `src/main.cu` — build Hessian → CPU Jacobi + GPU cuSOLVER → compare eigenvalues → modes + mobility.
  - `build/*.vcxproj` + `CMakeLists.txt` link **cusolver/cublas/cusparse**.
  - `THEORY.md`, `README.md`, `data/` (synthetic Cα fold), `scripts/`, `demo/`.
- `docs/STATUS.md` — `2.06` → **done** (**14/301; Phase 1 complete**). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb)

**2.06 NMA** teaches **dense linear algebra via a CUDA library**: build a symmetric matrix and call cuSOLVER
to diagonalize it. The standout is `src/kernels.cu` + THEORY §4: exactly what `cusolverDnDsyevd` computes,
the bufferSize/workspace pattern, the column-major eigenvector layout — and the built-in physics check that
the Hessian must have **exactly 6 zero modes** (rigid-body invariance).

## 4. How to build & run

```powershell
cd projects/02-structural-biology/2.06-normal-mode-analysis-elastic-network-models
msbuild build/normal-mode-analysis-elastic-network-models.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> 6 zero modes + functional frequencies + mobility + RESULT: PASS
```

## 5. What to study here

Reading path: `THEORY.md` (§2 the Hessian + zero modes, §4 the cuSOLVER call) → `src/kernels.cu` →
`src/reference_cpu.cpp` (Hessian + Jacobi). Then try README **Exercises**: GNM, compare mobility to real
B-factors, animate a low-frequency mode, or a sparse Lanczos solve for large proteins.

## 6. Verification

- ✅ `Release|x64` **and** `Debug|x64` build with **zero errors / zero warnings** (cuSOLVER links in both).
- ✅ Demo **PASS**: deterministic modes + mobility match `expected_output.txt`.
- ✅ **cuSOLVER eigenvalues == CPU Jacobi** to `2.3e-12` (machine precision).
- ✅ Physics check: the ANM Hessian has **exactly 6 zero (rigid-body) modes**; functional modes follow.
- ✅ `verify_project.py` → **DONE** (comment ratio **0.55**, no TODOs).
- *Build note:* the first synthetic fold left a terminus under-connected (one near-zero soft mode dominating
  mobility); tightened the fold + cutoff → clean, well-separated modes.
- **Environment:** RTX 2080 (`sm_75`), CUDA 13.3, VS 2026 (`v145`).

## 7. Phase 1 retrospective — 14 flagships, 13 distinct GPU patterns

| Flagship | Domain | GPU pattern |
|---|---|---|
| 1.12 Tanimoto | drug discovery | independent jobs · constant mem · `__popcll` |
| 3.01 Smith-Waterman | genomics | dependency **wavefront** (anti-diagonal DP) |
| 4.01 CT FBP | imaging | per-pixel **gather** + interpolation |
| 5.01 Monte Carlo dose | radiation | stochastic histories + **atomics** (integer-determinism) |
| 6.04 Lattice-Boltzmann | physiology | nearest-neighbour **stencil** + ping-pong |
| 7.10 Signal conv | medical AI | **shared-memory tiling** + halo |
| 8.03 EEG cuFFT | neuroscience | **cuFFT** library (batched FFT) |
| 9.02 SEIR ensemble | epidemiology | **ensemble RK4** (thread per trajectory) |
| 10.02 PBD soft tissue | biomechanics | **Jacobi constraint projection** |
| 11.09 k-means | biotech | **assign + atomic reduction** |
| 12.01 spectral search | omics | **batched dot-product** scoring |
| 13.02 PBPK | pharmacology | ensemble RK4 + on-device sampling |
| 14.02 reaction-diffusion | frontiers | reaction-diffusion **stencil** (Turing) |
| 2.06 NMA | structural biology | **cuSOLVER** dense eigensolver |

Cross-cutting lessons established and reused: a shared `__host__ __device__` core for exact CPU/GPU parity;
integer/fixed-point accumulation to make atomics deterministic (5.01, 11.09); honest floating-point
reproducibility tolerances for long iterative solvers (10.02, 14.02); and "no black box" documentation of
every library call (cuFFT, cuSOLVER). Every flagship: builds clean Debug+Release, GPU verified against a CPU
reference, full THEORY.md, its own push-note.

## 8. Next push preview

**Standards/template review, then Phase 2.** Per CLAUDE.md §11, I will (a) fold Phase-1 lessons into
`docs/PROJECT_TEMPLATE/` and the standards docs (the shared-header pattern, the determinism tricks, the
FP-tolerance guidance), then (b) begin **Phase 2** — the remaining 287 projects, domain-by-domain,
easiest-first, in parallel batches per CLAUDE.md §10.
