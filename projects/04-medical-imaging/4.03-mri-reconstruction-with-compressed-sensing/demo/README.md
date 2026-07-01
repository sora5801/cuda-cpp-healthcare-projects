# Demo — 4.3 MRI Reconstruction with Compressed Sensing

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/kspace_sample.txt` (a 32×32 slice with
   only ~34% of k-space acquired).
3. **Reconstruct** the image two ways on the CPU — the naive *zero-filled* inverse
   FFT (the "before" picture, full of aliasing) and *FISTA* compressed sensing
   (the "after" picture) — then run the **same FISTA on the GPU with cuFFT**.
4. **Verify** two things and print a clear `PASS`/`FAIL`:
   - the GPU (cuFFT) image agrees with the CPU (radix-2 FFT) image within tolerance
     (correctness / portability), and
   - CS actually *helps*: its error vs. the ground truth is smaller than the
     zero-filled baseline's (the science).
5. **Time** the CPU and GPU FISTA loops (CUDA events) — a *teaching artifact*, not
   a benchmark claim.

The program splits its output deliberately (PATTERNS.md §3):

- **stdout** is byte-for-byte deterministic (every printed number is derived from
  the deterministic CPU path) and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries timing and the run-varying GPU/CPU error, so it is shown but
  never diffed.

## Expected result

```
4.3 -- MRI Reconstruction with Compressed Sensing
under-sampled Cartesian CS-MRI (single slice, single coil), FISTA + cuFFT
image: 32x32   sampled k-space: 348/1024 (34.0%)   lambda=0.0100   iters=60
recon image RMS (CPU): 0.130112   peak: 0.923697
error vs truth (RMS): zero-filled=0.051109  CS-reconstructed=0.005779
CS improvement: 8.84x lower error than zero-filling
RESULT: PASS (GPU cuFFT recon matches CPU FISTA within tol; CS beats zero-fill)
```

The headline: from just **34%** of k-space, compressed sensing reconstructs the
image with **~8.8× lower error** than the naive zero-filled reconstruction — the
whole point of CS-MRI. On `stderr` you will also see the CPU vs. GPU FISTA timings
and the GPU-vs-CPU image agreement (RMS difference ≈ `3e-8`, far inside tolerance).

> On this tiny 32×32 slice the GPU is *slower* than the CPU because each iteration's
> two FFTs are dominated by kernel-launch overhead — exactly the launch-bound regime
> PATTERNS.md §7 warns about. The GPU's advantage appears at clinical sizes
> (256³ volumes × ~32 coils), where the FFTs are enormous.
