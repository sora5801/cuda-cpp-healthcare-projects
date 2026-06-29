// ===========================================================================
// src/kernels.cu  --  Ensemble SEIR kernel (one thread per trajectory)
// ---------------------------------------------------------------------------
// Project 9.02 : Large-Scale Compartmental & Metapopulation Models
//
// GPU twin of integrate_cpu(): each thread runs the same RK4 loop (seir.h) for
// one parameter set. main.cu compares the per-member results. See ../THEORY.md.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// ensemble_kernel: thread idx owns ensemble member idx.
//   It reads its (beta, gamma) from the parameter sweep, then runs the FULL RK4
//   time loop in registers/local memory and writes one MemberResult. There is no
//   inter-thread communication -- pure embarrassing parallelism over members.
//   (Divergence is mild: all members run the same number of steps; only the
//   peak-tracking branch differs.)
// ---------------------------------------------------------------------------
__global__ void ensemble_kernel(EnsembleConfig c, MemberResult* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ensemble_size(c)) return;          // guard the ragged last block

    double beta, gamma;
    member_params(c, idx, beta, gamma);
    out[idx] = integrate_member(c.N, c.N - c.I0, 0.0, c.I0, 0.0,
                                beta, c.sigma, gamma, c.dt, c.steps);
}

void integrate_gpu(const EnsembleConfig& c, std::vector<MemberResult>& results, float* kernel_ms) {
    const int M = ensemble_size(c);
    results.assign(M, MemberResult{});

    MemberResult* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<std::size_t>(M) * sizeof(MemberResult)));

    const int blocks = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    ensemble_kernel<<<blocks, THREADS_PER_BLOCK>>>(c, d_out);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("ensemble_kernel");

    CUDA_CHECK(cudaMemcpy(results.data(), d_out,
                          static_cast<std::size_t>(M) * sizeof(MemberResult),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));
}
