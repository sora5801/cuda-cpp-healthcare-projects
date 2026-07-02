// ===========================================================================
// src/kernels.cu  --  GPU ensemble forecast (one thread per member) + EnKF driver
// ---------------------------------------------------------------------------
// Project 6.27 : Parameter Estimation & Data Assimilation for Physiological Models
//
// This file holds the GPU FORECAST kernel and the driver that runs the whole
// Ensemble Kalman Filter using it. The forecast reuses the shared
// __host__ __device__ integrator (windkessel.h), so each device thread runs the
// identical RK4 the CPU reference runs -> the two ensembles agree to round-off.
// The ANALYSIS step is the shared host function enkf_analysis (reference_cpu.cpp),
// called between forecasts, so there is exactly one copy of the Kalman math.
//
// READ THIS AFTER: kernels.cuh (the interface + the pattern), windkessel.h.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include "windkessel.h"          // wk_forecast_member (shared HD forecast)

// Threads per block. 128 is a good default for this register-heavy, compute-bound
// kernel on sm_75..sm_89: a multiple of the 32-lane warp, enough warps to hide
// latency, and it keeps register pressure (each thread holds a WK_NSTATE state +
// RK4 scratch) from capping occupancy too hard.
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// forecast_kernel: thread idx advances ensemble member idx by ONE window.
//   Launch config (set in forecast_gpu):
//     grid  = ceil(m / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: idx = blockIdx.x*blockDim.x + threadIdx.x owns the member
//   whose augmented state occupies ens[idx*WK_NSTATE + 0..2] = [P, logR, logC].
//
//   The whole RK4 window runs in the thread's registers/local memory (the state is
//   only WK_NSTATE doubles); the sole global-memory traffic is the load at the
//   start and the store at the end. No shared memory, no atomics -- members are
//   fully independent, the textbook "ensemble ODE" pattern (docs/PATTERNS.md §1).
//   Divergence is negligible: every member runs the same number of RK4 sub-steps.
// ---------------------------------------------------------------------------
__global__ void forecast_kernel(double* __restrict__ ens, int m,
                                double t0, double dt, int substeps,
                                double T, double t_sys, double Q_peak) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m) return;                          // guard the ragged last block

    // Pull this member's state into registers, integrate, write it back.
    double x[WK_NSTATE];
    double* g = ens + static_cast<std::size_t>(idx) * WK_NSTATE;
    #pragma unroll
    for (int j = 0; j < WK_NSTATE; ++j) x[j] = g[j];

    wk_forecast_member(x, t0, dt, substeps, T, t_sys, Q_peak);

    #pragma unroll
    for (int j = 0; j < WK_NSTATE; ++j) g[j] = x[j];
}

// ---------------------------------------------------------------------------
// forecast_gpu: forecast the whole ensemble one window on the device.
//   Canonical CUDA shape: (1) upload the ensemble, (2) launch one thread/member,
//   (3) download it. We time ONLY the kernel (CUDA events) and add it to *accum_ms
//   so the driver reports the summed forecast cost across all windows.
//
//   NOTE (teaching vs. throughput): re-uploading/downloading the ensemble every
//   window is the simplest correct thing and keeps the analysis on the host, but
//   it pays a PCIe round-trip per window. A throughput build would keep the
//   ensemble resident on the device across windows; we keep it simple and say so
//   (THEORY §5). The point here is the parallel FORECAST, not the transfer.
// ---------------------------------------------------------------------------
void forecast_gpu(const EnKFConfig& c, std::vector<double>& ensemble, double t0,
                  float* accum_ms) {
    const int m = c.m;
    const std::size_t bytes = static_cast<std::size_t>(m) * WK_NSTATE * sizeof(double);

    // (1) Device buffer + upload. d_ prefix marks a DEVICE pointer (CLAUDE.md §12).
    double* d_ens = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ens, bytes));                 // can fail: out of memory
    CUDA_CHECK(cudaMemcpy(d_ens, ensemble.data(), bytes, cudaMemcpyHostToDevice));

    // (2) Launch: enough blocks to cover m members (round-up division).
    const int blocks = (m + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    forecast_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_ens, m, t0, c.dt, c.substeps,
                                                    c.T, c.t_sys, c.Q_peak);
    *accum_ms += timer.stop_ms();                          // GPU-measured kernel time
    CUDA_CHECK_LAST("forecast_kernel");                    // launch + execution errors

    // (3) Download the advanced ensemble and free the buffer.
    CUDA_CHECK(cudaMemcpy(ensemble.data(), d_ens, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_ens));
}

// ---------------------------------------------------------------------------
// run_enkf_gpu: the full filter with the forecast on the GPU.
//   This is a line-for-line twin of run_enkf_cpu (reference_cpu.cpp): same initial
//   ensemble (same seed), same per-window analysis seeds, same shared enkf_analysis.
//   The ONLY difference is that forecast_gpu replaces the serial forecast_cpu loop.
//   Because both forecasts call the identical shared integrator, the resulting
//   ensembles match to round-off -- exactly what main.cu verifies.
// ---------------------------------------------------------------------------
EnKFResult run_enkf_gpu(const EnKFConfig& c, const std::vector<double>& observations,
                        std::vector<double>& ensemble_out, float* forecast_ms) {
    std::vector<double> ens = build_initial_ensemble(c);   // shared with the CPU path
    double t = 0.0;
    double sq_err = 0.0;
    *forecast_ms = 0.0f;

    for (int k = 0; k < c.n_obs; ++k) {
        // FORECAST on the GPU (one thread per member).
        forecast_gpu(c, ens, t, forecast_ms);
        t += enkf_window_len(c);

        // Post-forecast ensemble-mean pressure vs. the observation (fit quality).
        double Pbar = 0.0;
        for (int i = 0; i < c.m; ++i) Pbar += ens[static_cast<std::size_t>(i) * WK_NSTATE];
        Pbar /= c.m;
        const double d = Pbar - observations[static_cast<std::size_t>(k)];
        sq_err += d * d;

        // ANALYSIS via the SHARED host function -- identical seed schedule as the CPU.
        const uint64_t obs_seed = c.seed * 0x100000001B3ULL + static_cast<uint64_t>(k) + 1ULL;
        enkf_analysis(c, ens, observations[static_cast<std::size_t>(k)], obs_seed);
    }

    EnKFResult r = summarize_ensemble(c, ens);
    r.final_rmse = std::sqrt(sq_err / c.n_obs);
    ensemble_out = ens;
    return r;
}
