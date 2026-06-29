# Push 2026-06-28 #05 -- flagship 6.04 lattice-boltzmann

> Push-note (CLAUDE.md §7.1). Fifth Phase 1 flagship — computational physiology.

## 1. Summary

The physiology flagship is done: **6.04 Lattice-Boltzmann Blood/Airflow Solver**, a complete, verified
GPU computational-fluid-dynamics solver. It introduces a fifth distinct GPU pattern — the **nearest-neighbour
stencil** (collide + stream, one thread per lattice node, ping-pong buffers) — and reproduces textbook
**Poiseuille (parabolic) channel flow**, matching the CPU to machine precision.

## 2. What changed

- [`projects/06-physiology-systems-biology/6.04-lattice-boltzmann-blood-airflow-solver/`](../projects/06-physiology-systems-biology/6.04-lattice-boltzmann-blood-airflow-solver) — fully implemented:
  - `src/lbm_d2q9.h` — **shared host+device** D2Q9 collide+stream (BGK, bounce-back walls, body force).
  - `src/kernels.cu` — `lbm_step_kernel` (1 thread/node, 2-D grid) + host ping-pong time loop.
  - `src/reference_cpu.cpp` / `.h` — serial reference + velocity moments.
  - `src/main.cu` — load → CPU + GPU LBM → compare velocity fields → print the channel profile.
  - `THEORY.md` (kinetic theory, BGK, Chapman-Enskog, D3Q19), `README.md`, `data/`, `scripts/`, `demo/`.
- `docs/STATUS.md` — `6.04` → **done** (5/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb)

**6.04 Lattice-Boltzmann** teaches the **stencil** pattern that drives GPU CFD: each node updates from its
nearest neighbours only (collide locally, stream to neighbours), so one thread per node with **double-buffered
(ping-pong)** reads makes every node independent within a step — no atomics, no `__syncthreads`. The most
interesting thing to study is `src/lbm_d2q9.h` (and THEORY §4): why two buffers eliminate the streaming race,
and why structure-of-arrays layout gives coalesced reads. The demo develops a perfect Poiseuille parabola.

## 4. How to build & run

```powershell
cd projects/06-physiology-systems-biology/6.04-lattice-boltzmann-blood-airflow-solver
msbuild build/lattice-boltzmann-blood-airflow-solver.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> parabolic velocity profile + RESULT: PASS (GPU==CPU)
```

## 5. What to study here

Reading path: `THEORY.md` (§2 the LBM equation + BGK, §4 ping-pong + coalescing) → `src/lbm_d2q9.h` →
`src/kernels.cu` → `src/reference_cpu.cpp`. Then try README **Exercises**: add a cylinder obstacle (vortex
street), shared-memory tiling, MRT collision, or a 3-D D3Q19 stencil.

## 6. Verification

- ✅ `Release|x64` **and** `Debug|x64` build with **zero errors / zero warnings**.
- ✅ Demo **PASS**: deterministic velocity profile matches `expected_output.txt`.
- ✅ **GPU velocity field == CPU to machine precision** (`max diff = 2.25e-16`).
- ✅ Physics: clean symmetric **Poiseuille parabola** (≈0 at walls, `u_max=0.00719` at centerline).
- ✅ `verify_project.py` → **DONE** (comment ratio **0.63**, no TODOs).
- **Environment:** RTX 2080 (`sm_75`), CUDA 13.3, VS 2026 (`v145`). 16×24 lattice, 6000 steps; CPU ~185 ms
  vs GPU ~130 ms (launch-bound on this small grid; throughput scales with grid size).

## 7. Known limitations / TODOs

- **2-D D2Q9, single-relaxation BGK**, straight channel (real solvers are 3-D D3Q19/D3Q27, often MRT, in
  vessel geometries).
- **One kernel launch per timestep** (no shared-memory tiling) ⇒ launch-bound on small grids; the GPU win
  grows with grid size. Simple equilibrium-shift forcing (production uses Guo forcing).

## 8. Next push preview

Next flagship: **7.10 Physiological signal/waveform analysis (1-D convolution)** (medical AI) — a sixth
pattern: shared-memory **tiled 1-D convolution** (FIR filtering of ECG/EEG-like waveforms).
