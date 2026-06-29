# GPU Patterns Cookbook — reusable idioms from the flagships

> Distilled from the 14 Phase-1 flagships. When you build a new project, **find the
> closest pattern below, open that flagship, and follow it.** It is the fastest way
> to a correct, consistent, Definition-of-Done project. Pair this with
> `docs/COMMENTING_STANDARD.md` (how to comment) and `docs/BUILD_GUIDE.md` (how to
> build). The flagships are the canonical worked examples.

## 1. The GPU pattern map — pick the closest exemplar

| If your project is… | Pattern | Study this flagship |
|---|---|---|
| score one query vs N items, each independent | independent jobs · constant-memory query | `1.12` Tanimoto, `12.01` spectral search |
| a dynamic-programming recurrence | anti-diagonal **wavefront** | `3.01` Smith-Waterman |
| per-output-pixel/voxel gather with interpolation | **gather** | `4.01` CT backprojection |
| stochastic / Monte-Carlo histories | per-thread RNG + **atomic** scoring | `5.01` Monte Carlo dose |
| a grid PDE / nearest-neighbour update | **stencil** + ping-pong buffers | `6.04` lattice-Boltzmann, `14.02` reaction-diffusion |
| a sliding-window / FIR / conv | **shared-memory tiling** + halo | `7.10` 1-D convolution |
| an FFT / spectral transform | use **cuFFT** | `8.03` EEG spectral |
| the same ODE for many parameter sets | **ensemble RK4** (thread per trajectory) | `9.02` SEIR, `13.02` PBPK |
| iterative constraint relaxation on a mesh | **Jacobi** projection + double buffer | `10.02` PBD soft tissue |
| clustering / centroid accumulation | parallel assign + **atomic reduce** | `11.09` k-means |
| a dense linear-algebra solve (eigen, SVD, lstsq) | use **cuSOLVER**/**cuBLAS** | `2.06` NMA eigensolver |

## 2. The shared `__host__ __device__` core (CPU/GPU parity)

The single most useful idiom. Put the *per-element physics* in **one header** as
`__host__ __device__` inline functions, so the CPU reference and the GPU kernel run
**byte-for-byte identical math** — making verification exact instead of approximate.

```cpp
// foo.h  (included by reference_cpu.cpp via the host compiler AND by kernels.cu via nvcc)
#ifdef __CUDACC__
#define FOO_HD __host__ __device__
#else
#define FOO_HD          // host compiler: the decorators don't exist
#endif

FOO_HD inline double update(/* state + params */) { /* the one true formula */ }
```

The host reference loops `update()`; the kernel calls `update()` from one thread.
Used in `5.01, 6.04, 9.02, 10.02, 13.02, 14.02`. Keep CUDA-only types out of this
header (no `__global__`), so the host compiler can include it.

## 3. Determinism — so the demo's stdout is reproducible

`demo/run_demo` diffs the program's **stdout** against `expected_output.txt`, so
stdout must be **identical every run**. Two rules:

1. **Split streams:** deterministic results → `stdout`; timings and run-varying
   numbers → `stderr` (shown, not diffed). Every flagship does this.
2. **Atomics are not associative in floating point.** If many threads `atomicAdd`
   into shared accumulators (Monte-Carlo tally, k-means centroids), a *float* sum
   depends on the (nondeterministic) order → irreproducible. **Accumulate in
   integers / fixed-point** instead — integer adds commute, so the result is
   deterministic *and* matches the CPU exactly. See `5.01` (integer energy quanta)
   and `11.09` (fixed-point coordinate sums, `km_to_fixed`).

Generate `expected_output.txt` by **running the built program and capturing its
stdout** — never hand-write it.

## 4. Verification tolerance — be honest about floating point

Pick the tolerance to match the numerics, and **document why**:

- **Exact (`== 0`)** when the result is integer or the same exact operations run on
  both sides (popcount, integer DP, fixed-point) — `1.12, 3.01, 5.01, 11.09, 12.01`.
- **~machine precision (`1e-12…1e-14`)** for short double-precision computations —
  `6.04, 9.02, 13.02, 2.06`.
- **A small *physical* tolerance (`1e-3`)** for **long iterative** solvers, where the
  GPU's fused-multiply-add (FMA) and the host compiler diverge by ~`1e-5` over
  thousands of steps **even in double precision**. This is real and worth teaching —
  see `10.02` (THEORY §5) and `14.02`. Verify to a physically-negligible tolerance and
  say so; do not pretend the results are bit-identical.

A second, stronger check where possible: compare against an **analytic/known result**
(Poiseuille parabola `6.04`, `AUC ≈ dose/CL` `13.02`, 6 rigid-body modes `2.06`,
mean AUC `13.02`) — that validates the science, not just CPU==GPU agreement.

## 5. Using a CUDA library without a black box

When a step is a solved problem (FFT, eigensolve, sort, GEMM), **use the library**
(cuFFT, cuSOLVER, cuBLAS, Thrust/CUB) — but document **what it computes, the data
layout it expects, and what hand-rolling would take** (CLAUDE.md §6.1.6). See
`8.03` (cuFFT R2C) and `2.06` (cuSOLVER `Dsyevd`). To link an extra library, add it
to **both** Link sections of the `.vcxproj` and to `CMakeLists.txt`:

```xml
<AdditionalDependencies>cusolver.lib;cublas.lib;cusparse.lib;cudart_static.lib;%(AdditionalDependencies)</AdditionalDependencies>
```
```cmake
find_package(CUDAToolkit REQUIRED)
target_link_libraries(<slug> PRIVATE CUDA::cufft)            # or CUDA::cusolver CUDA::cublas CUDA::cusparse
```

## 6. Synthetic data that makes the demo *interpretable*

Engineer the committed sample so the result is **meaningful and verifiable**:
embed a known answer (the query is a mutated copy of target `7` in `12.01`; five
separable populations in `11.09`; a motif to align in `3.01`), and report a metric
that recovers it (rank-1 hit, recovered cluster sizes, percent identity). Label
synthetic data as synthetic everywhere (CLAUDE.md §8). Initialization matters:
`11.09` uses farthest-first (k-means++) seeding to avoid a local minimum — a real
lesson kept as an exercise.

## 7. The honest-timing rule

Timing is a **teaching artifact, never a benchmark claim** (CLAUDE.md §12). Many
small kernel launches (per-diagonal `3.01`, per-step `6.04/10.02/14.02`) are
**launch-bound** and can be *slower* than the CPU on tiny inputs — say so, and note
the GPU's edge grows with problem size. Where the GPU genuinely wins (`4.01` ~30×,
`9.02` ~24×, `8.03` ~16× + an algorithmic `O(N²)→O(N log N)` win), state it plainly.

---

**Bottom line for a new project:** copy `docs/PROJECT_TEMPLATE/`, find your pattern
in §1, open that flagship, reuse its structure (shared `__host__ __device__` core,
stdout/stderr split, the right tolerance), and build to the Definition of Done
(`tools/verify_project.py`).
