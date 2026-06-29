// ===========================================================================
// src/main.cu  --  Entry point: load EEG, FFT, band powers, verify, report
// ---------------------------------------------------------------------------
// Project 8.03 : EEG/MEG Spectral Processing (cuFFT)
//
// 5-step shape:
//   1. Load the multi-channel EEG window (data/sample).
//   2. CPU reference: naive-DFT power spectrum (reference_cpu.cpp).
//   3. GPU: cuFFT batched FFT + power kernel (kernels.cu).
//   4. VERIFY: per-channel BAND POWERS agree (cuFFT vs DFT) within tolerance.
//   5. REPORT: deterministic band powers + dominant band per channel.
//
// Code tour: start here, then kernels.cuh -> kernels.cu (the cuFFT call), then
// reference_cpu.cpp. The science/GPU-mapping is in ../THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // spectrum_gpu, EegData
#include "reference_cpu.h"    // load_eeg, dft_power_cpu, band_powers, BAND_NAMES
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "8.3";
static const char* PROJECT_NAME = "EEG/MEG Spectral Processing (cuFFT)";

// cuFFT is single precision; the DFT reference is double. Band powers integrate
// many bins, so an allclose(atol, rtol) test is the right comparison.
static constexpr double ATOL = 1.0e-6;
static constexpr double RTOL = 1.0e-3;

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/eeg_sample.txt";
    EegData d;
    try {
        d = load_eeg(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference: naive-DFT power (timed) ------------------------
    std::vector<double> power_cpu, bands_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    dft_power_cpu(d, power_cpu);
    const double cpu_ms = cpu_timer.stop_ms();
    band_powers(d, power_cpu, bands_cpu);

    // ---- 3. GPU: cuFFT + power kernel (timed) -----------------------------
    std::vector<float> power_gpu_f;
    float gpu_kernel_ms = 0.0f;
    spectrum_gpu(d, power_gpu_f, &gpu_kernel_ms);
    // Reuse the SAME band-integration on the GPU spectrum (cast to double).
    const std::vector<double> power_gpu(power_gpu_f.begin(), power_gpu_f.end());
    std::vector<double> bands_gpu;
    band_powers(d, power_gpu, bands_gpu);

    // ---- 4. Verify (band powers agree) ------------------------------------
    double worst = 0.0;
    bool pass = true;
    for (std::size_t i = 0; i < bands_cpu.size(); ++i) {
        const double diff = std::fabs(bands_cpu[i] - bands_gpu[i]);
        if (diff > ATOL + RTOL * std::fabs(bands_cpu[i])) pass = false;
        const double rel = diff / (std::fabs(bands_cpu[i]) + 1e-12);
        if (rel > worst) worst = rel;
    }

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("EEG spectral analysis: %d channels, %d samples, fs=%.0f Hz (cuFFT batched FFT)\n",
                d.n_ch, d.n, d.fs);
    std::printf("band power per channel [delta theta alpha beta gamma] -> dominant:\n");
    for (int c = 0; c < d.n_ch; ++c) {
        int dom = 0;
        for (int b = 1; b < N_BANDS; ++b)
            if (bands_gpu[c * N_BANDS + b] > bands_gpu[c * N_BANDS + dom]) dom = b;
        std::printf("  ch%-2d:", c);
        for (int b = 0; b < N_BANDS; ++b) std::printf(" %.6f", bands_gpu[c * N_BANDS + b]);
        std::printf("  -> %s\n", BAND_NAMES[dom]);
    }
    std::printf("RESULT: %s (cuFFT band powers match CPU DFT within rtol=1e-3)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d ch x %d samples)\n", path.c_str(), d.n_ch, d.n);
    std::fprintf(stderr, "[timing] CPU naive DFT: %.3f ms   GPU cuFFT+power: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the naive DFT is O(N^2); cuFFT is O(N log N) and "
                         "batched. The gap explodes with N and channel count.\n");
    std::fprintf(stderr, "[verify] worst relative band-power error = %.3e\n", worst);

    return pass ? 0 : 1;
}
