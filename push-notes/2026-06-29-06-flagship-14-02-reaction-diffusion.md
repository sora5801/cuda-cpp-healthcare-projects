# Push 2026-06-29 #06 -- flagship 14.02 reaction-diffusion

> Push-note (CLAUDE.md §7.1). Thirteenth Phase 1 flagship — emerging frontiers.

## 1. Summary

The emerging-frontiers flagship is done: **14.02 Spatial Reaction-Diffusion (Gray-Scott)**, a complete,
verified GPU PDE solver that grows **Turing patterns** from a tiny seed. It is a thirteenth GPU pattern
instance — a 2-D reaction-diffusion **stencil** (cf. lattice-Boltzmann 6.04) — and a deliberately
**reduced-scope teaching version** (CLAUDE.md §11) of the catalog's 🔴 particle-based molecular-resolution
reaction-diffusion frontier project (the full PBRD approach is described in THEORY).

## 2. What changed

- [`projects/14-emerging-frontiers/14.02-spatial-whole-cell-reaction-diffusion-at-molecular-resolution/`](../projects/14-emerging-frontiers/14.02-spatial-whole-cell-reaction-diffusion-at-molecular-resolution) — fully implemented:
  - `src/rd.h` — **shared host+device** 5-point Laplacian + Gray-Scott per-cell update.
  - `src/kernels.cu` — `rd_step_kernel` (one thread per cell) + host ping-pong time loop.
  - `src/reference_cpu.cpp` / `.h` — field init (central seed) + serial reference.
  - `src/main.cu` — load → seed → CPU + GPU simulate → compare fields → print pattern metrics.
  - `THEORY.md`, `README.md`, `data/`, `scripts/`, `demo/`.
- `docs/STATUS.md` — `14.02` → **done** (13/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb)

**14.02 reaction-diffusion** teaches a PDE **stencil**: two chemicals diffuse (5-point Laplacian) and react
(the autocatalytic `U·V²` term), and from a seed they self-organize into a Turing labyrinth. The standout
file is `src/rd.h` + THEORY §2/§4: the Gray-Scott equations, why `(F,k)` select the pattern, and why
double-buffering (ping-pong) makes the per-cell stencil race-free on the GPU.

## 4. How to build & run

```powershell
cd projects/14-emerging-frontiers/14.02-spatial-whole-cell-reaction-diffusion-at-molecular-resolution
msbuild build/spatial-whole-cell-reaction-diffusion-at-molecular-resolution.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> pattern metrics + RESULT: PASS (GPU field == CPU)
```

## 5. What to study here

Reading path: `THEORY.md` (§2 Gray-Scott + Turing, §4 ping-pong stencil) → `src/rd.h` → `src/kernels.cu` →
`src/reference_cpu.cpp`. Then try README **Exercises**: sweep the (F,k) phase diagram (spots/stripes/mazes),
add shared-memory tiling, or write the V field to an image.

## 6. Verification

- ✅ `Release|x64` **and** `Debug|x64` build with **zero errors / zero warnings**.
- ✅ Demo **PASS**: deterministic pattern metrics match `expected_output.txt`.
- ✅ **GPU field == CPU** to `6.9e-08` (double; stable labyrinth regime keeps FP drift tiny).
- ✅ Pattern forms: from a tiny seed, V self-organizes into a Turing labyrinth (~8600 of 16384 cells active).
- ✅ `verify_project.py` → **DONE** (comment ratio **0.60**, no TODOs).
- **GPU win:** CPU ~1075 ms vs GPU ~67 ms (~16×) for 8000 steps on 128²; grows with grid size.
- **Environment:** RTX 2080 (`sm_75`), CUDA 13.3, VS 2026 (`v145`).
- *Build note:* the first parameter pick (`F=0.0367/k=0.0649`, small seed) decayed to a flat field; swept
  regimes and chose the robust `F=0.0545/k=0.062` labyrinth.

## 7. Known limitations / TODOs

- **Reduced-scope teaching version**: continuum 2-D Gray-Scott PDE, not the catalog's particle-based
  molecular-resolution RD (every molecule tracked; a multi-GPU frontier — see THEORY).
- Explicit Euler (conditionally stable); periodic boundaries; abstract model, not real biochemistry; one
  kernel launch per step (launch-bound on small grids).

## 8. Next push preview

**2.06 Normal Mode Analysis / Elastic Network Model** (structural biology) — the FINAL flagship, using
**cuSOLVER** for the eigendecomposition of the Hessian (a new CUDA library). After it, Phase 1 is complete
and I'll do a standards/template review before Phase 2.
