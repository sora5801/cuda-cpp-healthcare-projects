# THEORY — 7.10 Physiological Signal & Waveform Analysis

> For a reader who knows C++ but is new to CUDA and to signal processing.
> See [README.md](README.md) for the tour and build. _Educational only._

## 1. The science

Physiological waveforms — ECG (heart), EEG (brain), arterial blood pressure,
photoplethysmography — are continuous, high-frequency time series. Clinicians and
algorithms denoise them, extract features (R-peaks, QRS width, frequency bands),
and classify them (arrhythmia, seizure). The foundational operation for all of
this is **convolution** with a filter: low-pass to remove noise, band-pass to
isolate a rhythm, or — in modern systems — a *learned* convolution kernel inside a
1-D CNN (ResNet/TCN/WaveNet). This project implements that 1-D convolution and the
GPU optimization that makes it fast.

## 2. The math

A finite-impulse-response (FIR) filter convolves the signal `x` with taps `h`:

```
y[n] = Σ_{k=0}^{K-1} h[k] · x[n − HALO + k],   HALO = (K−1)/2
```

(centered; samples outside the signal are treated as zero). For a low-pass
filter, `h` is a smooth, positive, normalized kernel (we use a **Gaussian**,
`h[k] ∝ exp(−½((k−c)/σ)²)`, normalized so `Σ h = 1` ⇒ unity DC gain). Convolution
in time is multiplication in frequency, so a smooth `h` whose Fourier transform
rolls off at high frequency **attenuates noise** while preserving slow structure.

## 3. The algorithm

```
build h (K taps)
for each output index n:        # INDEPENDENT -> parallel
    y[n] = sum_k h[k] * x[n - HALO + k]   # zero-padded
```

**Complexity.** `Θ(n·K)` multiply-adds. The naive memory traffic is `n·K` input
reads, but consecutive outputs reuse `K−1` of their inputs, so the *unique* data
is only `n`. That reuse is exactly what shared-memory tiling captures.

## 4. The GPU mapping

**Decomposition.** One thread per output sample; a 1-D grid of `BLOCK`-thread
blocks. The naive kernel would read `K` inputs per thread from global memory —
`K×` redundant. Instead each block **tiles**:

```
shared tile[ BLOCK + 2*HALO ]            # the block's inputs + a halo each side
  thread t loads tile[t + HALO] = x[blockStart + t]      # the "main" sample
  first HALO threads also load the left & right halo samples (zero past the ends)
  __syncthreads()                         # tile complete before anyone reads
  y[n] = sum_k h[k] * tile[t + k]         # all reads now hit fast shared memory
```

- **Shared memory** (on-chip, ~100× faster than global) holds the reused window;
  each input is read from global memory **once** per block instead of `K` times.
- **Constant memory** holds the filter taps `h`: read by every thread, never
  written → the constant cache broadcasts a tap to a whole warp.
- **`__syncthreads()`** is essential: a thread must not read the tile until all
  threads (including the halo loaders) have finished writing it — the one place
  this project needs a barrier.
- **Dynamic shared memory:** the tile size depends on `K`, so it is sized at
  launch (`<<<grid, block, shmemBytes>>>`).

**CPU/GPU parity.** Both sum the taps in the same order, so the results match to
~`1e-7` (only float rounding / FMA differ).

## 5. Numerical considerations

- **Precision.** Single precision is standard for waveform DSP and CNN inference.
- **Determinism.** No cross-thread reduction (each thread writes its own `y[n]`),
  so the result is reproducible and matches the CPU.
- **Boundaries.** Zero-padding is the simplest choice; reflect/replicate padding
  avoids edge artifacts and is a one-line change.

## 6. How we verify correctness

`main.cu` convolves with `conv1d_cpu` (a plain double loop) and `conv1d_gpu` (the
tiled kernel) and compares the outputs (`max_abs_err ≈ 1e-7`). As a physical
sanity check it also reports the RMS of `x − filtered` (the energy removed): it is
non-zero, confirming the low-pass actually attenuates the injected noise.

## 7. Where this sits in the real world

The exact tiled convolution here is what **cuDNN** / **PyTorch `conv1d`** execute
(with many more optimizations: vectorized loads, register blocking, im2col or
implicit-GEMM formulations, Tensor Cores for low precision). The difference
between this FIR filter and a waveform **CNN** is only that the CNN *learns* the
taps and stacks many such convolutions with nonlinearities. Foundation models like
ECG-FM pretrain these 1-D convolutional/transformer stacks on hundreds of
thousands of patients; the inner loop is still the tiled 1-D convolution you see
here.

## References

- Oppenheim & Schafer, *Discrete-Time Signal Processing* — FIR filtering fundamentals.
- Bai, Kolter & Koltun (2018), *Temporal Convolutional Networks* — 1-D conv for sequences.
- NVIDIA CUDA C++ Programming Guide — shared memory, constant memory, `__syncthreads`.
- cuDNN documentation — production convolution primitives.
