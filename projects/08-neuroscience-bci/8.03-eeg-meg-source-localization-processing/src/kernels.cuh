// ===========================================================================
// src/kernels.cuh  --  GPU spectral-processing interface (cuFFT + power kernel)
// ---------------------------------------------------------------------------
// Project 8.03 : EEG/MEG Spectral Processing (cuFFT)
//
// THE BIG IDEA (seventh flagship pattern: USING A CUDA LIBRARY)
//   The Fast Fourier Transform is a solved problem with a superb GPU library --
//   cuFFT. The lesson here is how to use a library kernel WITHOUT it being a
//   black box: kernels.cu documents exactly what cufftExecR2C computes, the
//   batched layout it expects, and what it would take to hand-roll. We add only
//   a tiny custom kernel for the magnitude-squared (the power spectrum).
//
//   cuFFT batches one real-to-complex FFT PER CHANNEL in a single call -- the
//   natural mapping for multi-channel EEG.
//
//   kernels.cu defines the kernel + wrapper. main.cu calls spectrum_gpu().
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // EegData (pure C++, safe in .cu)

// Device kernel: power[i] = |X[i]|^2 / N^2.  X is the cuFFT output (float2 ==
// cufftComplex: .x real, .y imag). One thread per (channel, frequency-bin).
__global__ void power_kernel(const float2* __restrict__ X, int total, float invN2,
                             float* __restrict__ power);

// Host wrapper: batched real-to-complex FFT of all channels via cuFFT, then the
// power kernel. Returns the per-channel power spectrum (size n_ch*(n/2+1)) and
// the GPU time of the FFT + power step.
void spectrum_gpu(const EegData& d, std::vector<float>& power, float* kernel_ms);
