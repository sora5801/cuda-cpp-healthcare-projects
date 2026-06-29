// ===========================================================================
// src/reference_cpu.h  --  Signal model + FIR filter + CPU convolution
// ---------------------------------------------------------------------------
// Project 7.10 : Physiological Signal & Waveform Analysis
//
// WHAT THIS PROJECT COMPUTES
//   A 1-D CONVOLUTION of a physiological waveform (ECG/EEG-like) with an FIR
//   filter -- the operation at the heart of waveform analysis. The same 1-D
//   convolution is (a) classical signal filtering (denoising, band-pass) and
//   (b) the conv layer of every 1-D waveform CNN (ResNet/TCN/WaveNet). Here we
//   low-pass-filter a noisy synthetic ECG with a Gaussian FIR kernel.
//
//   y[n] = sum_{k=0}^{K-1} h[k] * x[n - HALO + k] ,  HALO = (K-1)/2  (zero-padded)
//
// WHY A GPU
//   Each output sample is independent -> one thread per sample. The naive kernel
//   re-reads the input K times per output from global memory; the optimized
//   kernel TILES a block of the signal into SHARED MEMORY once and reads from
//   there -- the canonical shared-memory-tiling lesson (kernels.cu). Clinical
//   pipelines filter thousands of multi-hour recordings, which is GPU-bound.
//
//   Pure C++ header (no CUDA). kernels.cu reuses Signal.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

// A loaded waveform: n samples (uniformly sampled in time).
struct Signal {
    int n = 0;
    std::vector<float> x;   // [n] samples
};

// Load a signal from the text format (data/README.md): "n" then n float samples.
Signal load_signal(const std::string& path);

// Build a normalized Gaussian low-pass FIR kernel of length K (odd), std sigma.
// Returns h with sum(h)=1 (so a flat signal passes through unchanged).
std::vector<float> make_gaussian_filter(int K, double sigma);

// CPU reference: 1-D convolution y = x (*) h, zero-padded at the boundaries.
// The trusted baseline the GPU tiled kernel is checked against. y sized to n.
void conv1d_cpu(const Signal& s, const std::vector<float>& h, std::vector<float>& y);
