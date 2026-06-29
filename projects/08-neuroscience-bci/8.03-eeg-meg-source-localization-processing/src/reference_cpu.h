// ===========================================================================
// src/reference_cpu.h  --  EEG model, naive-DFT reference, band integration
// ---------------------------------------------------------------------------
// Project 8.03 : EEG/MEG Spectral Processing (cuFFT)
//
// WHAT THIS PROJECT COMPUTES
//   The power spectrum of a multi-channel EEG window and its EEG BAND POWERS
//   (delta/theta/alpha/beta/gamma) -- the bread-and-butter of quantitative EEG.
//   The transform is a Fast Fourier Transform; on the GPU we use the cuFFT
//   library (batched, one FFT per channel) and a tiny kernel for |X|^2. The CPU
//   reference uses a naive O(N^2) DFT -- slow but OBVIOUSLY correct -- so we can
//   verify cuFFT's batched FFT.
//
// WHY A GPU / cuFFT
//   Real montages are 64-306 channels at 1-10 kHz over long recordings; the FFTs
//   are independent across channels and windows, which cuFFT batches efficiently.
//   This flagship's lesson is USING A LIBRARY KERNEL WITHOUT IT BEING A BLACK BOX
//   (kernels.cu explains exactly what cufftExecR2C computes and its layout).
//
//   (Per the catalog, the full project also covers source localization -- the
//   inverse problem -- which we describe in THEORY "real world"; the flagship
//   focuses on the spectral PROCESSING with cuFFT.)
//
//   Pure C++ header (no CUDA). kernels.cu reuses EegData.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

// A multi-channel EEG window: n_ch channels, n samples each, sampled at fs Hz.
//   x[c*n + t] = sample t of channel c.
struct EegData {
    int n_ch = 0;
    int n = 0;            // samples per channel (FFT length)
    double fs = 0.0;      // sampling rate (Hz)
    std::vector<float> x; // [n_ch * n]
};

// Number of EEG bands and their names/edges (Hz). Index order = BAND_NAMES.
constexpr int N_BANDS = 5;
extern const char* const BAND_NAMES[N_BANDS];     // delta theta alpha beta gamma
extern const double BAND_LO[N_BANDS];
extern const double BAND_HI[N_BANDS];

// Load an EegData from the text format (data/README.md):
//   header: "<n_ch> <n> <fs>"  then n_ch rows of n float samples.
EegData load_eeg(const std::string& path);

// CPU reference: power spectrum via a naive DFT, normalized by N^2.
//   power[c*(n/2+1) + k] = |X_c[k]|^2 / N^2 ,  k = 0..n/2.
// The trusted baseline the cuFFT result is checked against.
void dft_power_cpu(const EegData& d, std::vector<double>& power);

// Integrate the power spectrum into the 5 EEG bands per channel.
//   bands[c*N_BANDS + b] = sum of power over frequency bins in band b.
// Shared by both paths so the comparison is apples-to-apples.
void band_powers(const EegData& d, const std::vector<double>& power, std::vector<double>& bands);
