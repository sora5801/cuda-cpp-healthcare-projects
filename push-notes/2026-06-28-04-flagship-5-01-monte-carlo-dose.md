# Push 2026-06-28 #04 -- flagship 5.01 monte-carlo-dose

> Push-note (CLAUDE.md §7.1). Fourth Phase 1 flagship — radiation therapy / medical physics.

## 1. Summary

The medical-physics flagship is done: **5.01 Monte Carlo Dose Calculation (simplified slab)**, a complete,
verified GPU Monte Carlo photon-transport simulation. It introduces a fourth distinct GPU pattern —
**massively parallel stochastic histories with per-thread RNG and atomic scoring** — and a key
determinism lesson: by depositing **integer** energy quanta, atomic adds commute, so the GPU tally is
reproducible and equals the CPU tally exactly. A reduced-scope teaching model, clearly labeled.

## 2. What changed

- [`projects/05-radiation-therapy-medphys/5.01-monte-carlo-dose-calculation/`](../projects/05-radiation-therapy-medphys/5.01-monte-carlo-dose-calculation) — fully implemented:
  - `src/mc_physics.h` — **shared host+device** RNG (splitmix64) + photon transport (`RNG_HD` macro).
  - `src/kernels.cu` — `dose_kernel` (one thread per history, grid-stride, `atomicAdd` scoring) + wrapper.
  - `src/reference_cpu.cpp` / `.h` — loader + serial MC running the identical histories.
  - `src/main.cu` — load → CPU + GPU MC → exact tally compare → print depth-dose histogram.
  - `THEORY.md`, `README.md`, `data/` (parameter file), `scripts/`, `demo/`.
- `docs/STATUS.md` — `5.01` → **done** (4/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb)

**5.01 Monte Carlo dose** teaches the MC skeleton every clinical dose engine shares: independent particle
histories (one GPU thread each, grid-stride), per-thread reproducible RNG, and `atomicAdd` into a shared
depth-dose tally. The standout lesson is in `src/mc_physics.h` + THEORY §4: a shared `__host__ __device__`
RNG lets the CPU reproduce the GPU's exact histories, and **integer** energy quanta make the atomic tally
commute — so float-associativity non-determinism is avoided and GPU==CPU bit-for-bit. Warp **divergence**
(different photons take different branches) is discussed as the headline MC challenge.

## 4. How to build & run

```powershell
cd projects/05-radiation-therapy-medphys/5.01-monte-carlo-dose-calculation
msbuild build/monte-carlo-dose-calculation.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> depth-dose histogram + RESULT: PASS (GPU tally == CPU)
```

## 5. What to study here

Reading path: `THEORY.md` (§2 exponential free path, §4 RNG choice + integer-atomics determinism) →
`src/mc_physics.h` → `src/kernels.cu` → `src/reference_cpu.cpp`. Then try README **Exercises**: swap in
cuRAND (and switch to statistical verification), add an electron-buildup region, or make the slab layered.

## 6. Verification

- ✅ `Release|x64` **and** `Debug|x64` build with **zero errors / zero warnings**.
- ✅ Demo **PASS**: deterministic depth-dose histogram matches `expected_output.txt`.
- ✅ **GPU tally == CPU tally exactly** (`0 bin mismatches`; integer atomics commute).
- ✅ Physically sensible: dose highest at the entrance, falls off with depth; 69.9% deposited.
- ✅ `verify_project.py` → **DONE** (comment ratio **0.67**, no TODOs).
- **GPU win:** CPU MC ~11.1 ms vs GPU MC ~1.0 ms (~11×) for 262,144 histories; grows with history count.
- **Environment:** RTX 2080 (`sm_75`), CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- **Reduced-scope teaching model:** 1-D, single material, integer quanta, one absorb/forward-scatter branch.
  No real cross sections, no Klein-Nishina angular sampling, no electron transport, no CT geometry.
- **No buildup region** (peak at the surface) because charged-particle transport is omitted; real photon
  depth-dose peaks at `d_max`.
- Uses a **shared deterministic RNG** (not cuRAND) on purpose, for exact CPU/GPU verification.

## 8. Next push preview

Next flagship: **6.04 Lattice-Boltzmann (D2Q9) fluid solver** (physiology) — a fifth pattern: a
stream-and-collide **stencil** over a grid, the workhorse of GPU computational fluid dynamics for blood/airflow.
