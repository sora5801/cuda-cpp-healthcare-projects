# Push 2026-06-29 #01 -- flagship 9.02 seir-ensemble

> Push-note (CLAUDE.md §7.1). Eighth Phase 1 flagship — epidemiology / public health.

## 1. Summary

The epidemiology flagship is done: **9.02 Large-Scale Compartmental & Metapopulation Models**, a complete,
verified GPU SEIR **ensemble** solver. It introduces an eighth distinct GPU pattern — **parallel ensemble ODE
integration**: the same SEIR ODE solved for thousands of (β, γ) parameter sets, one RK4 trajectory per thread.
Double precision keeps the GPU and CPU in lock-step (matching to ~1e-15), and the results are textbook
epidemiology.

## 2. What changed

- [`projects/09-epidemiology-public-health/9.02-large-scale-compartmental-metapopulation-models/`](../projects/09-epidemiology-public-health/9.02-large-scale-compartmental-metapopulation-models) — fully implemented:
  - `src/seir.h` — **shared host+device** SEIR derivative + RK4 step + per-member integrator.
  - `src/kernels.cu` — `ensemble_kernel` (one thread per member runs the full RK4 loop) + wrapper.
  - `src/reference_cpu.cpp` / `.h` — ensemble config, (idx→β,γ) sweep mapping, serial reference.
  - `src/main.cu` — load → CPU + GPU integrate → per-member compare → print samples + ensemble summary.
  - `THEORY.md`, `README.md`, `data/`, `scripts/`, `demo/`.
- `docs/STATUS.md` — `9.02` → **done** (8/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb)

**9.02 SEIR ensemble** teaches the **uncertainty-quantification pattern**: solving an ODE for many parameter
samples at once. Each GPU thread runs an entire RK4 time loop in registers for one (β, γ) pair and writes a
summary — no inter-thread communication, no global traffic during integration. The standout file is
`src/seir.h` (and THEORY §4): one `__host__ __device__` integrator powering both CPU and GPU, so double-
precision RK4 matches to round-off, and why R0 = β/γ governs whether each member becomes an epidemic.

## 4. How to build & run

```powershell
cd projects/09-epidemiology-public-health/9.02-large-scale-compartmental-metapopulation-models
msbuild build/large-scale-compartmental-metapopulation-models.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> sample trajectories + ensemble summary + RESULT: PASS
```

## 5. What to study here

Reading path: `THEORY.md` (§2 SEIR + R0, §4 thread-per-member) → `src/seir.h` → `src/kernels.cu` →
`src/reference_cpu.cpp`. Then try README **Exercises**: Latin-hypercube/Sobol sampling, adaptive RK45, or
metapopulation coupling (a cuSPARSE batched sparse mat-vec per step).

## 6. Verification

- ✅ `Release|x64` **and** `Debug|x64` build with **zero errors / zero warnings**.
- ✅ Demo **PASS**: deterministic sample trajectories + summary match `expected_output.txt`.
- ✅ **GPU == CPU to ~machine precision** (`worst per-member diff = 1.4e-15`; double-precision RK4).
- ✅ Epidemiologically correct: R0=2.64 → 16% peak / 84% attack; R0=4.93 → 30% peak (day 77) / 99% attack;
  2689/4096 members with R0>1.
- ✅ `verify_project.py` → **DONE** (comment ratio **0.62**, no TODOs).
- **GPU win:** CPU ~85 ms vs GPU ~3.5 ms (~24×) for 4096 members; grows with ensemble size.
- **Environment:** RTX 2080 (`sm_75`), CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- **Independent SEIR members** (no spatial coupling); the metapopulation case (patches linked by a mobility
  matrix → batched cuSPARSE SpMV per step) is described in THEORY and is Exercise 5.
- Fixed-step RK4 (no adaptive/stiff handling); deterministic ODE (no demographic stochasticity); parameter
  ranges are illustrative, not fitted to a disease.

## 8. Next push preview

Next flagship: **10.02 Real-time soft-tissue deformation** (biomechanics) — a ninth pattern: a **mass-spring /
position-based dynamics (PBD)** solver with Jacobi constraint iterations on a deformable mesh.
