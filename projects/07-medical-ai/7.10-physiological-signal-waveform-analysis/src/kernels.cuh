// ===========================================================================
// src/kernels.cuh  --  GPU 1-D convolution interface (shared-memory tiled)
// ---------------------------------------------------------------------------
// Project 7.10 : Physiological Signal & Waveform Analysis
//
// THE BIG IDEA (sixth flagship pattern: SHARED-MEMORY TILING)
//   Each output sample y[n] needs K neighbouring inputs. Adjacent output
//   samples share almost all of those inputs, so the naive "one thread per
//   output, read K inputs from global memory" re-reads each input ~K times. The
//   optimized kernel loads a BLOCK of inputs (plus a HALO of K-1 extra samples)
//   into fast on-chip SHARED MEMORY once, then every thread reads its window
//   from there -- the canonical tiling optimization. The small filter lives in
//   CONSTANT memory (broadcast to all threads).
//
//   kernels.cu defines the kernel. main.cu calls conv1d_gpu().
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // Signal (pure C++, safe in .cu)

// Threads per block (= output samples per block / tile width).
static constexpr int CONV_BLOCK = 256;
// Maximum supported filter length (sets the constant-memory buffer size).
static constexpr int CONV_K_MAX = 64;

// Device kernel: thread computes one output sample from a shared-memory tile.
//   x    : [n] device input signal
//   K    : filter length ; halo = (K-1)/2
//   y    : [n] device output (filtered signal)
// The filter taps are read from a __constant__ array defined in kernels.cu.
// Launched with dynamic shared memory of (blockDim.x + 2*halo) floats.
__global__ void conv1d_kernel(const float* __restrict__ x, int n, int K, int halo,
                              float* __restrict__ y);

// Host wrapper: upload filter (to constant memory) + signal, launch the tiled
// kernel, copy the result back, report kernel time.
void conv1d_gpu(const Signal& s, const std::vector<float>& h,
                std::vector<float>& y, float* kernel_ms);
