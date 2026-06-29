# Push 2026-06-29 #02 -- flagship 10.02 pbd-soft-tissue

> Push-note (CLAUDE.md §7.1). Ninth Phase 1 flagship — biomechanics / surgery.

## 1. Summary

The biomechanics flagship is done: **10.02 Real-Time Soft-Tissue Deformation for Surgical Simulation**, a
complete, verified GPU **Position-Based Dynamics (PBD)** solver. It introduces a ninth distinct GPU pattern —
**parallel constraint projection** (Jacobi): one thread per particle, double-buffered across iterations. It
also surfaces a genuine **floating-point reproducibility** lesson: over thousands of iterations the GPU and
CPU drift at ~1e-5 (FMA differences), so we verify to a physically-negligible 1e-3 and explain why.

## 2. What changed

- [`projects/10-biomechanics-devices/10.02-real-time-soft-tissue-deformation-for-surgical-simulation/`](../projects/10-biomechanics-devices/10.02-real-time-soft-tissue-deformation-for-surgical-simulation) — fully implemented:
  - `src/pbd.h` — **shared host+device** Vec3 math + predict / Jacobi constraint projection / finalize.
  - `src/kernels.cu` — three kernels (predict, project ×iters with ping-pong, finalize) + time loop.
  - `src/reference_cpu.cpp` / `.h` — mesh init (pinned top row) + the serial reference.
  - `src/main.cu` — load → build mesh → CPU + GPU simulate → compare positions → print drape + samples.
  - `THEORY.md`, `README.md`, `data/`, `scripts/`, `demo/`.
- `docs/STATUS.md` — `10.02` → **done** (9/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb)

**10.02 PBD** teaches **parallel constraint projection**: a distance-constraint network (structural + shear
springs) relaxed with a Jacobi scheme where every particle reads its neighbours' read-only positions and
writes its own correction — so **double-buffering** (ping-pong) eliminates the race and no atomics are
needed. The standout files are `src/pbd.h` (the constraint math) and THEORY §5: a candid treatment of why
double-precision CPU and GPU still drift ~1e-5 over thousands of iterations (FMA), a reproducibility lesson
most tutorials skip.

## 4. How to build & run

```powershell
cd projects/10-biomechanics-devices/10.02-real-time-soft-tissue-deformation-for-surgical-simulation
msbuild build/real-time-soft-tissue-deformation-for-surgical-simulation.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> draped mesh samples + drape depth + RESULT: PASS
```

## 5. What to study here

Reading path: `THEORY.md` (§2 distance constraints, §4 ping-pong Jacobi, §5 FP reproducibility) →
`src/pbd.h` → `src/kernels.cu` → `src/reference_cpu.cpp`. Then try README **Exercises**: XPBD compliance,
graph-coloured Gauss-Seidel, a collision probe, or bending constraints.

## 6. Verification

- ✅ `Release|x64` **and** `Debug|x64` build with **zero errors / zero warnings**.
- ✅ Demo **PASS**: deterministic final positions + drape depth match `expected_output.txt`.
- ✅ **GPU matches CPU** to `2.2e-05` on positions of magnitude ~10 (tol `1e-3`; ~6 significant figures).
- ✅ Physically correct: pinned top edge holds, sheet drapes symmetrically ~12.6 units, stable (no blow-up).
- ✅ `verify_project.py` → **DONE** (comment ratio **0.54**, no TODOs).
- **Environment:** RTX 2080 (`sm_75`), CUDA 13.3, VS 2026 (`v145`). 24×24 mesh, 20 iters × 300 steps; CPU
  ~219 ms vs GPU ~224 ms (launch-bound on this tiny mesh; the win grows with mesh size).

## 7. Known limitations / TODOs

- **Grid sheet, distance constraints only** (no tetra volume, bending, or collisions); real organs use tetra
  meshes + XPBD/FEM/MPM.
- **One kernel launch per predict/iter/finalize** ⇒ launch-bound on small meshes.
- **FP reproducibility:** CPU/GPU drift ~1e-5 over thousands of iterations (FMA); verified to 1e-3 (THEORY §5).

## 8. Next push preview

Next flagship: **11.09 Flow-cytometry clustering (GPU k-means)** (biotech) — a tenth pattern: **k-means**
with a parallel assignment step + an atomic centroid-accumulation reduction over high-dimensional cell events.
