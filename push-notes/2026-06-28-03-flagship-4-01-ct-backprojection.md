# Push 2026-06-28 #03 -- flagship 4.01 ct-backprojection

> Push-note (CLAUDE.md §7.1). Third Phase 1 flagship — medical imaging.

## 1. Summary

The medical-imaging flagship is done: **4.01 CT Reconstruction (Filtered Backprojection)**, a complete,
verified 2-D parallel-beam FBP reconstructor. It introduces a third distinct GPU pattern — the **per-pixel
backprojection gather** on a 2-D thread grid — and is the first flagship with a clear GPU **win** (~30×).
The reconstruction is quantitatively correct: a unit-density disc reconstructs to ≈ 1.0.

## 2. What changed

- [`projects/04-medical-imaging/4.01-ct-reconstruction-filtered-backprojection/`](../projects/04-medical-imaging/4.01-ct-reconstruction-filtered-backprojection) — fully implemented:
  - `src/kernels.cu` — `backproject_kernel` (one thread per pixel, 2-D grid, linear detector interp) + wrapper.
  - `src/reference_cpu.cpp` / `.h` — geometry (`CTProblem`), Ram-Lak ramp filter, serial backprojection.
  - `src/main.cu` — load → host ramp-filter → CPU + GPU backproject → verify → print image samples.
  - `THEORY.md` (Radon transform, Fourier-slice, FBP, FDK), `README.md`, `data/` (analytic disc-phantom
    sinogram), `scripts/`, `demo/`.
- `docs/STATUS.md` — `4.01` → **done** (3/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb)

**4.01 CT FBP** teaches **backprojection as a gather**: each output pixel independently samples one
interpolated value from every projection and sums them, so each pixel is one GPU thread on a 2-D grid — no
atomics, no shared memory. The ramp filter (the "Filtered" in FBP) runs once on the host so both
reconstructions start identical; the GPU teaching point is the backprojection. The most interesting thing to
look at is `src/kernels.cu` (and THEORY §4): why host-precomputed `cos`/`sin` keep CPU and GPU in agreement,
and why this kernel is genuinely bandwidth-bound and GPU-favorable.

## 4. How to build & run

```powershell
cd projects/04-medical-imaging/4.01-ct-reconstruction-filtered-backprojection
msbuild build/ct-reconstruction-filtered-backprojection.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> image samples + RESULT: PASS (GPU matches CPU)
```

## 5. What to study here

Reading path: `THEORY.md` (§2 Radon/Fourier-slice, §4 the gather) → `src/kernels.cu` →
`src/reference_cpu.cpp` (ramp filter + backprojection). Then try README **Exercises**: bind the sinogram to
a texture and use `tex2D`, move the ramp filter to cuFFT, or extend to 3-D FDK.

## 6. Verification

- ✅ `Release|x64` **and** `Debug|x64` build with **zero errors / zero warnings**.
- ✅ Demo **PASS**: deterministic image samples match `expected_output.txt`.
- ✅ **GPU == CPU** (`max_abs_err = 1.2e-05`, tol `1e-3`).
- ✅ Physically correct: center pixel ≈ 1.0 (main disc density); flat inside discs, ≈ 0 outside.
- ✅ `verify_project.py` → **DONE** (comment ratio **0.61**, no TODOs).
- **Real GPU win:** CPU backproject ~7.75 ms vs GPU ~0.25 ms (~30×) on the 120×183 → 128² sample.
- **Environment:** RTX 2080 (`sm_75`), CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- **2-D parallel-beam only**; clinical scanners are fan/cone beam (FDK is the 3-D extension, in THEORY).
- Ramp filter is a **spatial** convolution for clarity (production uses an FFT); manual linear interp (no
  texture hardware) so the math stays visible; values are arbitrary density units, not calibrated HU.

## 8. Next push preview

Next flagship: **5.01 Monte Carlo dose (slab geometry)** (radiation/medical physics) — a fourth pattern:
massively parallel stochastic simulation with **cuRAND** per-thread RNG and atomic dose scoring.
