# THEORY — 8.03 EEG/MEG Spectral Processing (cuFFT)

> For a reader who knows C++ but is new to CUDA and to EEG. See [README.md](README.md)
> for the tour and build. _Educational only._

## 1. The science

The brain's electrical activity, measured as EEG (scalp) or MEG (magnetic), is
dominated by rhythmic oscillations. Their power in canonical frequency **bands**
is clinically meaningful: **delta** (0.5–4 Hz, deep sleep), **theta** (4–8 Hz,
drowsiness/memory), **alpha** (8–13 Hz, relaxed wakefulness), **beta** (13–30 Hz,
active thinking), **gamma** (30–100 Hz, binding/attention). Quantitative EEG (qEEG),
sleep staging, seizure detection, and brain-computer interfaces all start by
transforming the signal to the frequency domain and reading off band powers.

## 2. The math

For a length-`N` real signal `x[t]`, the **Discrete Fourier Transform** is

```
X[k] = Σ_{t=0}^{N-1} x[t] · e^{-2πi k t / N},   k = 0..N-1
```

For a *real* signal `X` is Hermitian-symmetric (`X[N-k] = conj(X[k])`), so only
`k = 0..N/2` are independent — what the real-to-complex (R2C) transform returns.
The **power spectrum** is `P[k] = |X[k]|² / N²`, and bin `k` corresponds to
frequency `f_k = k·fs/N`. **Band power** is the sum of `P[k]` over the bins in a
band's frequency range. The **Fast Fourier Transform (FFT)** computes the same
`X[k]` in `O(N log N)` instead of the DFT's `O(N²)` by recursively factoring the
transform (Cooley-Tukey).

## 3. The algorithm

```
for each channel c:                      # INDEPENDENT -> batch
    X_c = FFT(x_c)                        # cuFFT (R2C); naive DFT in the reference
    P_c[k] = |X_c[k]|^2 / N^2
for each channel, band:
    band_power = sum of P_c[k] over bins with f_k in [band_lo, band_hi)
```

## 4. The GPU mapping

**Use the library.** The FFT is a solved problem with a world-class GPU
implementation — **cuFFT** — so we use it rather than hand-rolling butterflies.
The flagship lesson is doing this **without a black box**:

```
cufftPlan1d(&plan, N, CUFFT_R2C, n_ch);   // n_ch independent length-N real FFTs,
                                          //   laid out contiguously (idist=N, odist=N/2+1)
cufftExecR2C(plan, d_in, d_out);          // computes X_c[k] for every channel, in O(N log N)
```

`d_in` is `n_ch·N` `cufftReal` (== float); `d_out` is `n_ch·(N/2+1)` `cufftComplex`
(== float2: `.x` real, `.y` imag). One `cufftExecR2C` does **all channels at once**
— the natural batched mapping for multi-channel EEG.

**The only custom kernel** is `power_kernel`: one thread per output bin, computing
`|X|²/N²`. Band integration is a cheap host post-step (a handful of sums).

**What it would take by hand.** A correct batched, mixed-radix FFT with good
memory coalescing and twiddle-factor handling is hundreds of tuned lines; cuFFT
encapsulates decades of that work. We treat it as a known building block — but we
*document what it computes and the exact layout it expects*, which is the
difference between "using a library" and "cargo-culting a call".

## 5. Numerical considerations

- **Precision.** cuFFT here is single precision (`cufftReal`/`cufftComplex`); the
  reference DFT is double. They compute the same transform, so band powers agree to
  ~`1e-6` relative — we verify with an `allclose(atol, rtol)` test, which is the
  right tool when comparing float vs double over an integrated quantity.
- **Determinism.** cuFFT with a fixed plan and input is deterministic; the power
  kernel has no cross-thread reduction. So the reported band powers are reproducible.
- **Spectral leakage.** With `fs == N` our test tones sit exactly on bins (no
  leakage); real signals need a **window** (Hann) and often **Welch averaging**
  (Exercise 1) to control leakage and variance.

## 6. How we verify correctness

`main.cu` computes band powers two ways — cuFFT (GPU) and a naive `O(N²)` DFT
(CPU) — and checks they agree within tolerance. The naive DFT is transparently the
textbook formula, so agreement validates that we are *driving cuFFT correctly*
(right plan, right layout, right normalization). As a physical check, each
synthetic channel was built with a known dominant rhythm and the demo recovers it
(ch0→alpha, ch1→beta, …), confirming the whole pipeline, not just CPU/GPU parity.

## 7. Where this sits in the real world

MNE-Python, FieldTrip, and EEGLAB wrap this spectral core with windowing, Welch/
multitaper estimation, artifact rejection, and montage handling. The catalog entry
also spans **source localization** — the *inverse problem* of estimating which
brain locations produced the sensor signals — solved with beamformers (LCMV) or
Bayesian methods over a precomputed leadfield matrix; those are large
matrix factorizations that also benefit from the GPU (cuSOLVER/cuBLAS) and are a
natural follow-on (Exercise 5). The batched FFT you call here is the front end of
all of it.

## References

- Cooley & Tukey (1965) — the FFT algorithm.
- Niedermeyer & da Silva, *Electroencephalography* — EEG bands and clinical meaning.
- NVIDIA **cuFFT** documentation — plans, R2C layout, batching.
- Van Veen et al. (1997) — LCMV beamforming for source localization.
